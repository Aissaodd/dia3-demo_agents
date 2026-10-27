"""Couche des fournisseurs de modèles, indépendante de l'orchestrateur.

- ``build_router(cfg)`` construit un ``LLMRouter`` : un fournisseur par défaut,
  et éventuellement un autre par rôle (planner, verifier, codeur…) choisi
  EXPLICITEMENT dans .env (JARVIS_MODEL_PLANNER=deepseek:deepseek-v4-pro).
- Il n'y a aucun repli automatique d'un fournisseur vers un autre : si le
  fournisseur choisi échoue, l'erreur remonte. Chaque appel enregistre le
  fournisseur et le modèle réellement utilisés.
"""

from jarvis.providers.base import (BudgetExceeded, LLM, LLMError, Usage, extract_json, looks_secret,
                                   redact)
from jarvis.providers.echo import EchoLLM
from jarvis.providers.remote import AnthropicLLM, ClaudeCLI, DeepSeekLLM, GroqLLM

PROVIDERS: dict[str, type[LLM]] = {
    "deepseek": DeepSeekLLM,
    "groq": GroqLLM,
    "anthropic": AnthropicLLM,
    "claude": ClaudeCLI,
    "echo": EchoLLM,
}
# Alias : « simulation » est un echo, aucun appel réseau ni payant.
ALIASES = {"simulation": "echo", "simulate": "echo", "offline": "echo"}

# Clés à renseigner pour chaque fournisseur (affichées dans le tableau de bord).
PROVIDER_KEYS = {
    "deepseek": ("DEEPSEEK_KEY", "DEEPSEEK_API_KEY"),
    "groq": ("GROQ_KEY", "GROQ_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "claude": (),
    "echo": (),
}


def parse_spec(spec: str, default_provider: str) -> tuple[str, str]:
    """« provider:modele », « provider » ou « modele » → (provider, modele)."""
    spec = (spec or "").strip()
    if ":" in spec:
        prov, model = spec.split(":", 1)
        prov = ALIASES.get(prov.strip().lower(), prov.strip().lower())
        if prov in PROVIDERS:
            return prov, model.strip()
    low = ALIASES.get(spec.lower(), spec.lower())
    if low in PROVIDERS:
        return low, ""
    return default_provider, spec


def build_llm(cfg_or_provider, model: str = "", **kw) -> LLM:
    """Un fournisseur. Accepte une Config (compatibilité) ou un nom de fournisseur."""
    if hasattr(cfg_or_provider, "llm_provider"):
        cfg = cfg_or_provider
        provider, model = cfg.llm_provider, cfg.llm_model
        kw.setdefault("delay", cfg.retry_delay)
        kw.setdefault("timeout", cfg.llm_timeout)
    else:
        provider = cfg_or_provider
    provider = ALIASES.get(provider, provider)
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise LLMError(f"fournisseur inconnu : {provider} ({', '.join(PROVIDERS)})")
    if cls is EchoLLM:
        kw.pop("timeout", None)
        kw.pop("delay", None)
        return EchoLLM(**kw)
    return cls(model, **kw)


class LLMRouter:
    """Choisit le fournisseur selon le rôle de l'appel. Interface identique à ``LLM``."""

    def __init__(self, default: LLM, by_role: dict[str, LLM] | None = None):
        self.default = default
        self.by_role = dict(by_role or {})

    @property
    def name(self) -> str:
        return self.default.name

    @property
    def calls(self) -> int:
        seen, total = set(), 0
        for llm in [self.default, *self.by_role.values()]:
            if id(llm) not in seen:
                seen.add(id(llm))
                total += llm.calls
        return total

    def for_role(self, role: str | None) -> LLM:
        if role and role in self.by_role:
            return self.by_role[role]
        if role and role.startswith("agent:") and "agent" in self.by_role:
            return self.by_role["agent"]
        return self.default

    def complete(self, messages, role=None) -> str:
        return self.for_role(role).complete(messages, role=role)

    def complete_with_usage(self, messages, role=None):
        return self.for_role(role).complete_with_usage(messages, role=role)

    def describe(self) -> dict[str, str]:
        out = {"défaut": self.default.label}
        out.update({r: llm.label for r, llm in self.by_role.items()})
        return out

    @property
    def paid(self) -> bool:
        return any(l.paid for l in [self.default, *self.by_role.values()])


def as_router(llm) -> LLMRouter:
    return llm if isinstance(llm, LLMRouter) else LLMRouter(llm)


def build_router(cfg) -> LLMRouter:
    default = build_llm(cfg)
    cache = {(cfg.llm_provider, cfg.llm_model): default}
    by_role = {}
    for role, spec in cfg.role_models.items():
        prov, model = parse_spec(spec, cfg.llm_provider)
        key = (prov, model)
        if key not in cache:
            cache[key] = build_llm(prov, model, delay=cfg.retry_delay, timeout=cfg.llm_timeout)
        by_role[role] = cache[key]
    return LLMRouter(default, by_role)


__all__ = ["LLM", "LLMError", "BudgetExceeded", "Usage", "EchoLLM", "DeepSeekLLM", "GroqLLM", "AnthropicLLM",
           "ClaudeCLI", "PROVIDERS", "PROVIDER_KEYS", "LLMRouter", "as_router", "build_llm", "build_router",
           "extract_json", "redact", "looks_secret", "parse_spec"]
