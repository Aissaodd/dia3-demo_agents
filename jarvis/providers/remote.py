"""Fournisseurs distants : DeepSeek, Groq (API compatibles OpenAI), Anthropic,
et Claude Code en ligne de commande (abonnement Claude).

URL et noms de modèles vérifiés sur la documentation officielle le 2026-10-27 :
- DeepSeek : https://api.deepseek.com, modèles deepseek-flash et deepseek-v4-pro
  (deepseek-chat et deepseek-reasoner ont été retirés le 2026-07-24) ;
- Groq : https://api.groq.com/openai/v1, ex. openai/gpt-oss-120b ;
- Anthropic : https://api.anthropic.com/v1/messages ;
- Gemini : https://generativelanguage.googleapis.com/v1beta/openai/, ex. gemini-flash-latest.
"""

import json
import os
import shutil
import subprocess
import tempfile

from jarvis.providers.base import LLM, LLMError, http_error


def _key(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    raise LLMError(f"{names[0]} manquante dans .env")


class OpenAICompatLLM(LLM):
    paid = True
    base_url = ""
    key_names: tuple[str, ...] = ()
    retired: dict[str, str] = {}

    def __init__(self, model: str = "", api_key: str = "", key_index: int = 1, **kw):
        super().__init__(model, **kw)
        from openai import OpenAI

        self.key_index = key_index
        # max_retries=0 : c'est notre socle qui décide quoi réessayer.
        self.client = OpenAI(api_key=api_key or _key(*self.key_names), base_url=self.base_url,
                             timeout=self.timeout, max_retries=0)

    def _complete(self, messages, role=None):
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is None and type(exc).__name__ not in ("APIConnectionError", "APITimeoutError"):
                raise LLMError(f"{type(exc).__name__}: {exc}")
            headers = getattr(getattr(exc, "response", None), "headers", None) or {}
            msg = str(exc)
            if self.model in self.retired:
                msg += f" — le modèle « {self.model} » a été retiré, utilise {self.retired[self.model]}"
            raise http_error(status, msg, headers.get("retry-after"))
        if not resp.choices:
            raise LLMError("réponse sans contenu", transient=True)
        usage = getattr(resp, "usage", None)
        return (resp.choices[0].message.content or "",
                getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))


class DeepSeekLLM(OpenAICompatLLM):
    name = "deepseek"
    base_url = "https://api.deepseek.com"
    key_names = ("DEEPSEEK_KEY", "DEEPSEEK_API_KEY")
    default_model = "deepseek-flash"
    retired = {"deepseek-chat": "deepseek-flash", "deepseek-reasoner": "deepseek-v4-pro"}


class GroqLLM(OpenAICompatLLM):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
    key_names = ("GROQ_KEY", "GROQ_API_KEY")
    default_model = "openai/gpt-oss-120b"


class GeminiLLM(OpenAICompatLLM):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    key_names = ("GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY")
    default_model = "gemini-flash-latest"


class AnthropicLLM(LLM):
    """API Messages d'Anthropic, appelée avec httpx (pas de SDK supplémentaire)."""

    name = "anthropic"
    paid = True
    default_model = "claude-sonnet-5-5"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "", max_tokens: int = 4096, api_key: str = "", key_index: int = 1, **kw):
        super().__init__(model, **kw)
        self.key = api_key or _key("ANTHROPIC_API_KEY")
        self.key_index = key_index
        self.max_tokens = max_tokens

    def _complete(self, messages, role=None):
        import httpx

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if convo and convo[-1]["role"] == m["role"]:   # l'API veut des tours alternés
                convo[-1]["content"] += "\n\n" + m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        if convo and convo[0]["role"] != "user":
            convo.insert(0, {"role": "user", "content": "(début de la conversation)"})
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": convo}
        if system:
            body["system"] = system
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        try:
            r = httpx.post(self.url, json=body, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise http_error(None, f"{type(exc).__name__}: {exc}")
        if r.status_code != 200:
            raise http_error(r.status_code, r.text, r.headers.get("retry-after"))
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return text, usage.get("input_tokens"), usage.get("output_tokens")


class ClaudeCLI(LLM):
    """Claude Code en mode headless (abonnement Claude, pas de facturation à l'appel).

    Isolé : --tools "" coupe TOUS ses outils (pas de lecture de fichiers, pas de
    commandes), pas de serveurs MCP, pas de session sauvegardée, et il tourne
    dans un dossier temporaire pour ne charger aucun CLAUDE.md du projet. Les
    actions passent uniquement par le registre d'outils de Jarvis.
    """

    name = "claude"
    paid = False

    def __init__(self, model: str = "", **kw):
        kw.setdefault("timeout", 300)
        super().__init__(model if model.startswith("claude") or model in ("sonnet", "opus", "haiku") else "",
                         **kw)
        self.exe = shutil.which("claude")
        if not self.exe:
            raise LLMError("CLI 'claude' introuvable (installe Claude Code puis 'claude login')")

    def _complete(self, messages, role=None):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages if m["role"] != "system")
        cmd = [self.exe, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
               "--no-session-persistence"]
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        try:
            proc = subprocess.run(cmd, input=convo, capture_output=True, text=True, encoding="utf-8",
                                  timeout=self.timeout, cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired:
            raise LLMError("délai dépassé pour claude -p", transient=True)
        try:
            data = json.loads(proc.stdout)
        except ValueError:
            data = None
        if proc.returncode != 0 or (isinstance(data, dict) and data.get("is_error")):
            detail = (data or {}).get("result") if isinstance(data, dict) else proc.stderr
            raise LLMError(f"claude -p a échoué : {str(detail).strip()[:300]}")
        if not isinstance(data, dict):
            return proc.stdout
        usage = data.get("usage") or {}
        return data.get("result", ""), usage.get("input_tokens"), usage.get("output_tokens")
