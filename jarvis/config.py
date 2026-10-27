"""Réglages de Jarvis, lus depuis l'environnement (.env)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent


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


@dataclass
class Config:
    # Fournisseur LLM : "deepseek", "claude" (CLI Claude Code) ou "echo" (hors ligne, tests).
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    db_path: Path = PROJECT_DIR / "data" / "jarvis.db"
    # Seul dossier que les outils fichiers peuvent lire ou écrire.
    workspace: Path = PROJECT_DIR / "workspace"
    logs_dir: Path = PROJECT_DIR / "data" / "logs"
    max_steps: int = 8              # étapes max par plan
    max_step_attempts: int = 3      # tentatives par étape avant échec
    max_tool_calls: int = 6         # appels d'outils max par tentative
    retry_delay: float = 2.0        # secondes, doublées à chaque tentative
    lease_seconds: int = 180        # verrou d'exécution d'une mission
    disabled_tools: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(PROJECT_DIR / ".env")
        cfg = cls()
        cfg.llm_provider = os.environ.get("JARVIS_LLM", cfg.llm_provider).strip().lower()
        cfg.llm_model = os.environ.get("JARVIS_MODEL", cfg.llm_model).strip()
        if os.environ.get("JARVIS_DB"):
            cfg.db_path = Path(os.environ["JARVIS_DB"])
        if os.environ.get("JARVIS_WORKSPACE"):
            cfg.workspace = Path(os.environ["JARVIS_WORKSPACE"])
        cfg.max_steps = _int("JARVIS_MAX_STEPS", cfg.max_steps)
        cfg.max_step_attempts = max(1, _int("JARVIS_MAX_ATTEMPTS", cfg.max_step_attempts))
        cfg.max_tool_calls = _int("JARVIS_MAX_TOOL_CALLS", cfg.max_tool_calls)
        cfg.retry_delay = _float("JARVIS_RETRY_DELAY", cfg.retry_delay)
        cfg.disabled_tools = {
            t.strip() for t in os.environ.get("JARVIS_DISABLED_TOOLS", "").split(",") if t.strip()
        }
        return cfg
