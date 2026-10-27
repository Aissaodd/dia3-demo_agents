"""Accès aux modèles de langage.

Tous les fournisseurs exposent la même méthode ``complete(messages)``, donc
on change de modèle sans toucher aux agents. Les erreurs passagères sont
réessayées ici ; les autres remontent à l'orchestrateur.
"""

import json
import os
import re
import shutil
import subprocess
import time


class LLMError(Exception):
    pass


class LLM:
    name = "base"

    def __init__(self, retries: int = 2, delay: float = 2.0):
        self.retries = retries
        self.delay = delay
        self.calls = 0

    def complete(self, messages: list[dict]) -> str:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self.calls += 1
                text = self._complete(messages)
                if not text or not text.strip():
                    raise LLMError("réponse vide du modèle")
                return text
            except LLMError as exc:
                last = exc
            except Exception as exc:  # erreurs réseau / SDK
                last = LLMError(f"{type(exc).__name__}: {exc}")
            if attempt < self.retries:
                time.sleep(self.delay * (2 ** attempt))
        raise last or LLMError("échec inconnu")

    def _complete(self, messages: list[dict]) -> str:
        raise NotImplementedError


class DeepSeekLLM(LLM):
    """DeepSeek, via son API compatible OpenAI."""

    name = "deepseek"
    base_url = "https://api.deepseek.com"

    def __init__(self, model: str, **kw):
        super().__init__(**kw)
        from openai import OpenAI

        key = os.environ.get("DEEPSEEK_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise LLMError("DEEPSEEK_KEY manquante dans .env")
        self.client = OpenAI(api_key=key, base_url=self.base_url)
        self.model = model or "deepseek-chat"

    def _complete(self, messages):
        resp = self.client.chat.completions.create(messages=messages, model=self.model)
        return resp.choices[0].message.content or ""


class ClaudeCLI(LLM):
    """Claude Code en mode headless (abonnement Claude), comme Harvey.

    Pas de --dangerously-skip-permissions : le modèle répond, il n'agit pas.
    Les actions passent uniquement par le registre d'outils de Jarvis.
    """

    name = "claude"

    def __init__(self, model: str = "", timeout: int = 300, **kw):
        super().__init__(**kw)
        self.exe = shutil.which("claude")
        if not self.exe:
            raise LLMError("CLI 'claude' introuvable (installe Claude Code puis 'claude login')")
        self.model = model if model.startswith("claude") else ""
        self.timeout = timeout

    def _complete(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages if m["role"] != "system"
        )
        cmd = [self.exe, "-p", "--output-format", "json"]
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        try:
            proc = subprocess.run(
                cmd, input=convo, capture_output=True, text=True,
                encoding="utf-8", timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise LLMError("délai dépassé pour claude -p")
        if proc.returncode != 0:
            raise LLMError(f"claude -p a échoué : {proc.stderr.strip()[:300]}")
        try:
            return json.loads(proc.stdout).get("result", "")
        except ValueError:
            return proc.stdout


class EchoLLM(LLM):
    """Fournisseur hors ligne : rejoue des réponses écrites à l'avance.

    Sert aux tests et à essayer le tableau de bord sans clé d'API. Sans
    script, il produit un plan d'une étape et une réponse simple.
    """

    name = "echo"

    def __init__(self, script: list | None = None, **kw):
        kw.setdefault("retries", 0)
        super().__init__(**kw)
        self.script = list(script or [])
        self.seen: list[list[dict]] = []

    def _complete(self, messages):
        self.seen.append(messages)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        system = messages[0]["content"] if messages else ""
        last = messages[-1]["content"]
        if "PLANIFICATEUR" in system:
            return json.dumps({"steps": [{
                "title": "Répondre à l'objectif",
                "instructions": "Réponds directement à l'objectif.",
                "agent": "generaliste",
                "success_criteria": "Une réponse non vide.",
            }]})
        if "VÉRIFICATEUR" in system:
            return json.dumps({"ok": True, "reason": "mode echo"})
        return json.dumps({"action": "final", "answer": f"[echo] {last[:200]}"})


def build_llm(cfg) -> LLM:
    if cfg.llm_provider == "deepseek":
        return DeepSeekLLM(cfg.llm_model, delay=cfg.retry_delay)
    if cfg.llm_provider == "claude":
        return ClaudeCLI(cfg.llm_model, delay=cfg.retry_delay)
    if cfg.llm_provider == "echo":
        return EchoLLM()
    raise LLMError(f"fournisseur inconnu : {cfg.llm_provider} (deepseek, claude ou echo)")


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
