"""Couche des fournisseurs de modèles, indépendante de l'orchestrateur.

- ``build_router(cfg)`` construit un ``LLMRouter`` : un fournisseur par défaut,
  et éventuellement un autre par rôle (planner, verifier, codeur…) choisi
  EXPLICITEMENT dans .env (JARVIS_MODEL_PLANNER=deepseek:deepseek-v4-pro).
- Plusieurs clés : si un fournisseur échoue (surcharge, limite de débit, clé
  refusée, solde épuisé), Jarvis essaie ses autres clés (GEMINI_API_KEY_2,
  _3…), puis les fournisseurs listés dans JARVIS_FALLBACK, dans l'ordre.
  Sans clé supplémentaire ni JARVIS_FALLBACK, il n'y a aucun repli. Chaque
  appel enregistre le fournisseur et le modèle réellement utilisés.
"""

import os

from jarvis.providers.base import (BudgetExceeded, FallbackLLM, LLM, LLMError, Usage, extract_json,
                                   looks_secret, redact)
from jarvis.providers.echo import EchoLLM
from jarvis.providers.remote import AnthropicLLM, ClaudeCLI, DeepSeekLLM, GeminiLLM, GroqLLM

PROVIDERS: dict[str, type[LLM]] = {
    "deepseek": DeepSeekLLM,
    "groq": GroqLLM,
    "gemini": GeminiLLM,
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
    "gemini": ("GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"),
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
        kw.setdefault("retries", cfg.llm_retries)
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
        kw.pop("retries", None)
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


def provider_keys(provider: str) -> list[str]:
    """Toutes les clés d'un fournisseur, dans l'ordre : GEMINI_API_KEY, GEMINI_API_KEY_2… _9."""
    seen, out = set(), []
    for name in PROVIDER_KEYS.get(provider, ()):
        for var in [name] + [f"{name}_{i}" for i in range(2, 10)]:
            value = os.environ.get(var, "").strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
    return out


def build_chain(cfg, provider: str, model: str = "") -> LLM:
    """Le fournisseur demandé avec chacune de ses clés, puis ceux de JARVIS_FALLBACK."""
    provider = ALIASES.get(provider, provider)
    base_kw = {"delay": cfg.retry_delay, "timeout": cfg.llm_timeout, "retries": cfg.llm_retries}
    if provider == "echo":
        return build_llm(provider, model, **base_kw)
    specs = [(provider, model)] + [parse_spec(s, provider) for s in getattr(cfg, "fallback", [])]
    links, seen = [], set()
    for i, (prov, mod) in enumerate(specs):
        if prov == "echo":
            continue                      # jamais de repli silencieux vers des réponses simulées
        keys = provider_keys(prov) if PROVIDER_KEYS.get(prov) else []
        for n, key in enumerate(keys or [""], 1):
            ident = (prov, mod, key)
            if ident in seen:
                continue
            seen.add(ident)
            kw = dict(base_kw)
            if key:
                kw.update(api_key=key, key_index=n)
            try:
                links.append(build_llm(prov, mod, **kw))
            except LLMError:
                if i == 0 and not links:
                    raise                 # fournisseur principal inutilisable : erreur explicite
                # repli mal configuré : ignoré ici, signalé par Config.warnings()
    if len(links) == 1:
        return links[0]
    # Un seul réessai avant de changer de clé ; le dernier garde tous les siens.
    for link in links[:-1]:
        link.retries = min(link.retries, 1)
    return FallbackLLM(links)


def build_router(cfg) -> LLMRouter:
    default = build_chain(cfg, cfg.llm_provider, cfg.llm_model)
    cache = {(cfg.llm_provider, cfg.llm_model): default}
    by_role = {}
    for role, spec in cfg.role_models.items():
        prov, model = parse_spec(spec, cfg.llm_provider)
        key = (prov, model)
        if key not in cache:
            cache[key] = build_chain(cfg, prov, model)
        by_role[role] = cache[key]
    return LLMRouter(default, by_role)


__all__ = ["LLM", "LLMError", "BudgetExceeded", "Usage", "EchoLLM", "FallbackLLM", "build_chain", "provider_keys", "DeepSeekLLM", "GroqLLM", "GeminiLLM", "AnthropicLLM",
           "ClaudeCLI", "PROVIDERS", "PROVIDER_KEYS", "LLMRouter", "as_router", "build_llm", "build_router",
           "extract_json", "redact", "looks_secret", "parse_spec"]
