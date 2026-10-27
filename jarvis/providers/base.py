"""Socle commun des fournisseurs de modèles.

Chaque fournisseur implémente ``_complete(messages, role)`` et renvoie le
texte (et, s'il les connaît, les jetons consommés). Le socle s'occupe du
reste : réessais des seules erreurs passagères, mesure de la latence,
calcul du coût quand le prix du modèle est connu, et masquage des secrets
dans les messages d'erreur.
"""

import json
import os
import re
import threading
import time
from dataclasses import dataclass


class LLMError(Exception):
    """Erreur d'appel de modèle. ``transient`` = réessayer peut marcher."""

    def __init__(self, message: str, transient: bool = False, retry_after: float | None = None):
        super().__init__(redact(message))
        self.transient = transient
        self.retry_after = retry_after


class BudgetExceeded(LLMError):
    """Budget de la mission épuisé : définitif tant qu'on ne l'augmente pas."""

    def __init__(self, message: str):
        super().__init__(message, transient=False)


@dataclass
class Usage:
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int = 0
    cost_usd: float | None = None   # None = prix inconnu (jamais inventé)


# --- prix (USD par million de jetons : entrée, sortie) ----------------------
# Relevés sur les pages officielles le 2026-10-27. Pour DeepSeek, on prend le
# tarif « heures pleines » (le plus cher) : le budget reste une borne haute.
# Les prix changent : surcharge-les avec JARVIS_PRICES='{"modele": [entree, sortie]}'.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-flash": (0.30, 1.20),
    "deepseek-v4-flash": (0.30, 1.20),      # ancien nom, servi et facturé comme Flash
    "deepseek-v4-pro": (1.32, 3.96),
    "openai/gpt-oss-120b": (0.15, 0.60),    # Groq
    "openai/gpt-oss-20b": (0.075, 0.30),    # Groq
}


def load_prices() -> dict[str, tuple[float, float]]:
    prices = dict(DEFAULT_PRICES)
    raw = os.environ.get("JARVIS_PRICES", "").strip()
    if raw:
        try:
            for model, pair in json.loads(raw).items():
                prices[str(model)] = (float(pair[0]), float(pair[1]))
        except (ValueError, TypeError, IndexError, AttributeError):
            pass  # réglage illisible : on garde les prix par défaut
    return prices


# --- secrets ----------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"),          # DeepSeek, OpenAI, Anthropic (sk-ant-…)
    re.compile(r"\bgsk_[A-Za-z0-9]{12,}"),            # Groq
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|mot de passe)\s*[=:]\s*\S{6,}"),
]
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)


def _env_secrets() -> list[str]:
    return [v for k, v in os.environ.items() if _SECRET_ENV.search(k) and v and len(v) >= 8]


def redact(text) -> str:
    """Retire les clés et mots de passe d'un texte avant de l'afficher ou de l'enregistrer."""
    if text is None:
        return ""
    text = str(text)
    for value in _env_secrets():
        if value in text:
            text = text.replace(value, "[secret masqué]")
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[secret masqué]", text)
    return text


def looks_secret(text: str) -> bool:
    return redact(text) != text


# --- fournisseur de base ------------------------------------------------------
class LLM:
    name = "base"
    paid = False          # True = chaque appel est facturé
    default_model = ""

    def __init__(self, model: str = "", retries: int = 2, delay: float = 2.0, timeout: float = 120,
                 prices: dict | None = None):
        self.model = model or self.default_model
        self.retries = retries
        self.delay = delay
        self.timeout = timeout
        self.prices = prices if prices is not None else load_prices()
        self.calls = 0
        self._lock = threading.Lock()
        self.sleep = time.sleep

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}" if self.model else self.name

    def complete(self, messages: list[dict], role: str | None = None) -> str:
        return self.complete_with_usage(messages, role)[0]

    def complete_with_usage(self, messages: list[dict], role: str | None = None) -> tuple[str, Usage]:
        last: LLMError | None = None
        for attempt in range(self.retries + 1):
            with self._lock:
                self.calls += 1
            t0 = time.perf_counter()
            try:
                out = self._complete(messages, role)
                text, tin, tout = out if isinstance(out, tuple) else (out, None, None)
                if not text or not str(text).strip():
                    raise LLMError("réponse vide du modèle", transient=True)
                usage = Usage(self.name, self.model, tin, tout, int((time.perf_counter() - t0) * 1000))
                usage.cost_usd = self.cost(tin, tout)
                return str(text), usage
            except LLMError as exc:
                last = exc
            except Exception as exc:  # erreur inattendue d'un SDK : on ne réessaie pas à l'aveugle
                last = LLMError(f"{type(exc).__name__}: {exc}", transient=False)
            if not last.transient or attempt >= self.retries:
                break
            wait = last.retry_after if last.retry_after is not None else self.delay * (2 ** attempt)
            self.sleep(min(max(wait, 0), 30))
        raise last or LLMError("échec inconnu")

    def cost(self, tin, tout) -> float | None:
        if not self.paid:
            return 0.0
        price = self.prices.get(self.model)
        if price is None or tin is None or tout is None:
            return None
        return (tin * price[0] + tout * price[1]) / 1_000_000

    def _complete(self, messages: list[dict], role: str | None = None):
        raise NotImplementedError


def http_error(status: int | None, message: str, retry_after=None) -> LLMError:
    """Classe une erreur HTTP : 408/409/429/5xx sont passagères, le reste non."""
    transient = status is None or status in (408, 409, 429) or status >= 500
    try:
        ra = float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        ra = None
    hints = {401: "clé refusée", 402: "solde insuffisant", 403: "accès refusé", 404: "modèle ou URL inconnus",
             429: "limite de débit atteinte"}
    prefix = f"HTTP {status}" + (f" ({hints[status]})" if status in hints else "") if status else "réseau"
    return LLMError(f"{prefix} : {message[:400]}", transient=transient, retry_after=ra)


def extract_json(text: str):
    """Le premier objet JSON trouvé dans une réponse de modèle, ou None."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    for cand in candidates:
        try:
            return json.loads(cand)
        except ValueError:
            pass
        start = cand.find("{")
        while start != -1:
            depth = 0
            in_str = esc = False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cand[start : i + 1])
                        except ValueError:
                            break
            start = cand.find("{", start + 1)
    return None
