"""Gestion des erreurs et reprise (agent « recuperation »).

Politique déterministe, sans modèle, pour qu'elle reste prévisible :
- une erreur passagère (réseau, délai, limite de débit, 5xx) → on réessaie ;
- une erreur définitive (clé refusée, solde insuffisant, modèle inconnu,
  budget épuisé) → l'étape échoue tout de suite : réessayer coûterait sans
  rien changer, il faut une intervention humaine ;
- un résultat refusé par le vérificateur → on refait avec le motif du refus,
  et à la dernière tentative on passe à un autre agent, seulement s'il n'a
  PAS plus de permissions risquées que l'agent d'origine.
"""

from dataclasses import dataclass

from jarvis.providers.base import BudgetExceeded, LLMError


class StepTimeout(Exception):
    """La tentative a dépassé son délai."""


KIND_LABELS = {
    "transient": "erreur passagère",
    "timeout": "délai dépassé",
    "permanent": "erreur définitive",
    "budget": "budget épuisé",
    "rejected": "résultat refusé",
    "approval": "en attente d'approbation",
}


def classify(exc: BaseException) -> str:
    if isinstance(exc, BudgetExceeded):
        return "budget"
    if isinstance(exc, StepTimeout):
        return "timeout"
    if isinstance(exc, LLMError):
        return "transient" if exc.transient else "permanent"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    return "permanent"


@dataclass
class Decision:
    retry: bool
    reason: str
    needs_human: bool = False


def decide(kind: str, attempt: int, max_attempts: int) -> Decision:
    """Que faire après l'échec de la tentative ``attempt`` (numérotée à partir de 0) ?"""
    remaining = max_attempts - attempt - 1
    if kind in ("budget", "permanent"):
        return Decision(False, f"{KIND_LABELS[kind]} : réessayer ne changerait rien", needs_human=True)
    if remaining <= 0:
        return Decision(False, f"{KIND_LABELS.get(kind, kind)} après {max_attempts} tentatives", needs_human=True)
    return Decision(True, f"{KIND_LABELS.get(kind, kind)} : nouvelle tentative ({remaining} restante(s))")


RISKY = ("écriture", "réseau", "sensible")


def safe_alternative(step: dict, current: str, tried: list[str], agents, tools) -> str | None:
    """Un autre agent pour la dernière tentative, sans élévation de privilèges :
    ses outils risqués doivent déjà être permis à l'agent d'origine."""
    original = agents.get(step["agent"])

    def risky(agent) -> set[str]:
        return {t for t in agent.tools if (tools.get(t) and tools.get(t).risk in RISKY)}

    allowed = risky(original)
    candidates = [agents.fallback] + [a.name for a in agents.ranked_for(step)]
    for name in candidates:
        if name in tried or name == current or name not in agents.names():
            continue
        if risky(agents.get(name)) <= allowed:
            return name
    return None
