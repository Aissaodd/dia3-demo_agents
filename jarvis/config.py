"""Réglages de Jarvis, lus depuis l'environnement (.env).

Aucune clé n'est stockée ici : les fournisseurs les lisent eux-mêmes dans
l'environnement au moment de se connecter.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

APPROVAL_MODES = ("sensible", "ecriture", "jamais")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _role_models() -> dict[str, str]:
    """JARVIS_MODEL_PLANNER=deepseek:deepseek-v4-pro → {"planner": "deepseek:deepseek-v4-pro"}.

    JARVIS_MODEL_AGENT=... vaut pour tous les agents, JARVIS_MODEL_AGENT_CODEUR=... pour un seul.
    """
    out = {}
    for k, v in os.environ.items():
        if k.startswith("JARVIS_MODEL_") and v.strip():
            role = k[len("JARVIS_MODEL_"):].lower()
            if role.startswith("agent_"):
                role = "agent:" + role[len("agent_"):]
            out[role] = v.strip()
    return out


@dataclass
class Config:
    # Fournisseur : deepseek, groq, anthropic, claude (CLI Claude Code), echo (hors ligne).
    llm_provider: str = "deepseek"
    llm_model: str = ""                 # vide = modèle par défaut du fournisseur
    role_models: dict[str, str] = field(default_factory=dict)
    llm_timeout: float = 120.0          # secondes par appel de modèle
    db_path: Path = PROJECT_DIR / "data" / "jarvis.db"
    # Seul dossier que les outils fichiers peuvent lire ou écrire.
    workspace: Path = PROJECT_DIR / "workspace"
    logs_dir: Path = PROJECT_DIR / "data" / "logs"
    max_steps: int = 8                  # étapes max par plan
    max_step_attempts: int = 3          # tentatives par étape avant échec
    max_tool_calls: int = 6             # appels d'outils max par tentative
    retry_delay: float = 2.0            # secondes, doublées à chaque tentative
    lease_seconds: int = 180            # verrou d'exécution d'une mission (renouvelé à chaque appel)
    max_parallel: int = 3               # étapes indépendantes exécutées en même temps
    step_timeout: float = 600.0         # secondes max par tentative d'étape
    mission_timeout: float = 3600.0     # secondes max par exécution de mission
    # Budget par mission (cumulé sur toutes les reprises).
    max_llm_calls: int = 80
    max_tokens: int = 500_000
    max_cost_usd: float = 0.50          # ne compte que les appels dont le prix est connu
    max_web_calls: int = 15             # pages web lues par mission
    # Approbation humaine : sensible (défaut) = actions destructrices ; ecriture = toute
    # écriture ou accès réseau ; jamais = aucune (déconseillé).
    approval_mode: str = "sensible"
    disabled_tools: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(PROJECT_DIR / ".env")
        cfg = cls()
        cfg.llm_provider = os.environ.get("JARVIS_LLM", cfg.llm_provider).strip().lower()
        cfg.llm_model = os.environ.get("JARVIS_MODEL", cfg.llm_model).strip()
        cfg.role_models = _role_models()
        if os.environ.get("JARVIS_DB"):
            cfg.db_path = Path(os.environ["JARVIS_DB"])
        if os.environ.get("JARVIS_WORKSPACE"):
            cfg.workspace = Path(os.environ["JARVIS_WORKSPACE"])
        cfg.llm_timeout = _float("JARVIS_LLM_TIMEOUT", cfg.llm_timeout)
        cfg.max_steps = max(1, _int("JARVIS_MAX_STEPS", cfg.max_steps))
        cfg.max_step_attempts = max(1, _int("JARVIS_MAX_ATTEMPTS", cfg.max_step_attempts))
        cfg.max_tool_calls = max(0, _int("JARVIS_MAX_TOOL_CALLS", cfg.max_tool_calls))
        cfg.retry_delay = _float("JARVIS_RETRY_DELAY", cfg.retry_delay)
        cfg.max_parallel = max(1, _int("JARVIS_MAX_PARALLEL", cfg.max_parallel))
        cfg.step_timeout = _float("JARVIS_STEP_TIMEOUT", cfg.step_timeout)
        cfg.mission_timeout = _float("JARVIS_MISSION_TIMEOUT", cfg.mission_timeout)
        cfg.max_llm_calls = _int("JARVIS_MAX_LLM_CALLS", cfg.max_llm_calls)
        cfg.max_tokens = _int("JARVIS_MAX_TOKENS", cfg.max_tokens)
        cfg.max_cost_usd = _float("JARVIS_MAX_COST_USD", cfg.max_cost_usd)
        cfg.max_web_calls = _int("JARVIS_MAX_WEB_CALLS", cfg.max_web_calls)
        mode = os.environ.get("JARVIS_APPROVAL", cfg.approval_mode).strip().lower()
        cfg.approval_mode = mode if mode in APPROVAL_MODES else "sensible"
        cfg.disabled_tools = {
            t.strip() for t in os.environ.get("JARVIS_DISABLED_TOOLS", "").split(",") if t.strip()
        }
        return cfg

    def warnings(self) -> list[str]:
        """Problèmes de configuration à signaler (sans jamais afficher de clé)."""
        from jarvis.providers import PROVIDER_KEYS, PROVIDERS, parse_spec
        from jarvis.providers.remote import DeepSeekLLM

        out = []
        specs = {"défaut": f"{self.llm_provider}:{self.llm_model}"}
        specs.update(self.role_models)
        for role, spec in specs.items():
            prov, model = parse_spec(spec, self.llm_provider)
            if prov not in PROVIDERS and prov not in ("simulation", "simulate", "offline"):
                out.append(f"[{role}] fournisseur inconnu : {prov}")
                continue
            keys = PROVIDER_KEYS.get(prov, ())
            if keys and not any(os.environ.get(k) for k in keys):
                out.append(f"[{role}] {prov} : clé manquante ({' ou '.join(keys)})")
            if prov == "deepseek" and model in DeepSeekLLM.retired:
                out.append(f"[{role}] le modèle {model} a été retiré par DeepSeek le 2026-07-24 : "
                           f"utilise {DeepSeekLLM.retired[model]}")
        if self.approval_mode == "jamais":
            out.append("Approbations désactivées (JARVIS_APPROVAL=jamais) : les écrasements de fichiers "
                       "se feront sans confirmation.")
        return out
