"""Fournisseur hors ligne : aucun réseau, aucun coût.

Sert aux tests et au mode simulation. Trois façons de répondre, dans l'ordre :
un script (liste de réponses, ou d'exceptions à lever), une fonction
``responder(messages, role)``, ou des réponses par défaut qui font tourner
tout le moteur (plan d'une étape, verdict favorable, réponse « [echo] … »).
"""

import json
import threading

from jarvis.providers.base import LLM


class EchoLLM(LLM):
    name = "echo"
    paid = False

    def __init__(self, script: list | None = None, responder=None, **kw):
        kw.setdefault("retries", 0)
        kw.setdefault("prices", {})
        super().__init__("", **kw)
        self.script = list(script or [])
        self.responder = responder
        self.seen: list[list[dict]] = []
        self.roles: list[str | None] = []
        self._script_lock = threading.Lock()

    def _complete(self, messages, role=None):
        with self._script_lock:
            self.seen.append(messages)
            self.roles.append(role)
            item = self.script.pop(0) if self.script else None
        if item is None and self.responder is not None:
            item = self.responder(messages, role)
        if item is None:
            item = self._default(messages, role)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)

    @staticmethod
    def _default(messages, role):
        system = messages[0]["content"] if messages else ""
        last = messages[-1]["content"] if messages else ""
        if role == "planner" or "PLANIFICATEUR" in system:
            return {"steps": [{
                "title": "Répondre à l'objectif",
                "instructions": "Réponds directement à l'objectif.",
                "agent": "generaliste",
                "success_criteria": "Une réponse non vide.",
            }]}
        if role == "verifier" or "VÉRIFICATEUR" in system:
            return {"ok": True, "reason": "mode echo (aucune vérification réelle)"}
        if role == "router" or "AIGUILLEUR" in system:
            return {"type": "conversation"}
        if role == "chat":
            return f"[echo] {last[:200]}"
        if role == "synthesizer":
            return f"[echo] synthèse\nLEÇON: (mode echo)"
        return {"action": "final", "answer": f"[echo] {last[:200]}"}
