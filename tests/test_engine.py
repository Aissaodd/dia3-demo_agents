"""Tests du moteur de missions, avec un modèle simulé (aucune clé d'API)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.agents import build_default_agents  # noqa: E402
from jarvis.config import Config  # noqa: E402
from jarvis.llm import EchoLLM, LLMError, extract_json  # noqa: E402
from jarvis.memory import LeaseHeld, MissionStore  # noqa: E402
from jarvis.orchestrator import Orchestrator  # noqa: E402
from jarvis.tools import ToolError, Workspace, build_default_registry, calculate  # noqa: E402


def make(tmp_path, script, **cfg_kw):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws", retry_delay=0, **cfg_kw)
    store = MissionStore(cfg.db_path)
    llm = EchoLLM(script)
    orch = Orchestrator(cfg, store, llm)
    orch.sleep = lambda s: None
    return orch, store, llm


def plan_of(*steps):
    return {"steps": [{"title": t, "instructions": t, "agent": a, "success_criteria": "ok"} for t, a in steps]}


def test_mission_with_tool_and_synthesis(tmp_path):
    orch, store, llm = make(tmp_path, [
        plan_of(("Calculer", "analyste"), ("Rédiger", "redacteur")),
        {"action": "tool", "tool": "calculatrice", "args": {"expression": "6*7"}},
        {"action": "final", "answer": "Le résultat est 42."},
        {"ok": True, "reason": "correct"},
        {"action": "tool", "tool": "ecrire_fichier", "args": {"path": "r.md", "content": "42"}},
        {"action": "final", "answer": "Fichier r.md écrit."},
        {"ok": True, "reason": "fichier présent"},
        "Réponse : 42, enregistrée dans r.md.\nLEÇON: utiliser la calculatrice.",
    ])
    mid = orch.submit("Calcule 6*7 et écris-le")
    m = orch.run(mid)
    assert m["status"] == "done"
    assert m["result"].startswith("Réponse : 42")
    assert (tmp_path / "ws" / "r.md").read_text(encoding="utf-8") == "42"
    kinds = [e["kind"] for e in store.get_events(mid)]
    assert kinds.count("tool") == 2 and "mission_done" in kinds
    assert store.search("calculatrice")  # la leçon est en mémoire
    assert m["llm_calls"] == 8 and m["lease_until"] == 0


def test_rejected_then_retried_with_feedback(tmp_path):
    orch, store, llm = make(tmp_path, [
        plan_of(("Étape", "analyste")),
        {"action": "final", "answer": "brouillon"},
        {"ok": False, "reason": "il manque le total"},
        {"action": "final", "answer": "total = 10"},
        {"ok": True, "reason": "ok"},
    ])
    mid = orch.submit("x")
    assert orch.run(mid)["status"] == "done"
    step = store.get_steps(mid)[0]
    assert step["attempts"] == 2 and step["output"] == "total = 10"
    # le motif du refus est renvoyé à l'agent
    assert "il manque le total" in llm.seen[3][-1]["content"]


def test_fallback_agent_then_failure_then_resume(tmp_path):
    orch, store, llm = make(tmp_path, [
        plan_of(("Étape", "chercheur")),
        LLMError("réseau"),                          # tentative 1 : erreur
        {"action": "final", "answer": "a"}, {"ok": False, "reason": "faux"},  # tentative 2
        {"action": "final", "answer": "b"}, {"ok": False, "reason": "faux"},  # tentative 3 (secours)
    ])
    mid = orch.submit("x")
    m = orch.run(mid)
    assert m["status"] == "failed"
    step = store.get_steps(mid)[0]
    assert step["status"] == "failed" and step["agent_used"] == "generaliste"
    assert "fallback" in [e["kind"] for e in store.get_events(mid)]

    # Reprise : le plan est conservé, seule l'étape échouée repart.
    llm.script = [{"action": "final", "answer": "c"}, {"ok": True, "reason": "ok"}]
    m = orch.run(mid)
    assert m["status"] == "done" and m["result"] == "c"


def test_resume_after_crash_skips_done_steps(tmp_path):
    orch, store, llm = make(tmp_path, [])
    mid = orch.submit("x")
    store.save_plan(mid, plan_of(("A", "analyste"), ("B", "analyste"))["steps"])
    a, b = store.get_steps(mid)
    store.update_step(a["id"], status="done", output="résultat A")
    store.update_step(b["id"], status="running", attempts=1)  # coupé en plein vol
    store.update_mission(mid, status="running")

    llm.script = [{"action": "final", "answer": "résultat B"}, {"ok": True, "reason": "ok"}, "Synthèse"]
    m = orch.run(mid)
    assert m["status"] == "done" and m["result"] == "Synthèse"
    # B a reçu le résultat de A en contexte, et aucun nouveau plan n'a été fait
    assert "résultat A" in llm.seen[0][1]["content"]
    assert len(llm.seen) == 3


def test_lease_blocks_second_worker(tmp_path):
    orch, store, _ = make(tmp_path, [])
    mid = orch.submit("x")
    store.update_mission(mid, worker="autre:1", lease_until=10**12)
    with pytest.raises(LeaseHeld):
        orch.run(mid)


def test_cancel(tmp_path):
    orch, store, _ = make(tmp_path, [plan_of(("A", "analyste"))])
    mid = orch.submit("x")
    orch._plan(mid, "x")
    store.request_cancel(mid)
    orch.store.claim = lambda *a, **k: None  # garde la demande d'annulation
    assert orch.run(mid)["status"] == "cancelled"


def test_unreadable_plan_falls_back_to_one_step(tmp_path):
    orch, store, _ = make(tmp_path, ["je ne sais pas faire de JSON",
                                      "réponse en texte libre", {"ok": True, "reason": ""}])
    mid = orch.submit("Dis bonjour")
    m = orch.run(mid)
    assert m["status"] == "done" and m["result"] == "réponse en texte libre"
    assert store.get_steps(mid)[0]["agent"] == "generaliste"


def test_tool_permissions(tmp_path):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws", disabled_tools={"lire_page_web"})
    reg = build_default_registry(cfg)
    with pytest.raises(ToolError, match="non autorisé"):
        reg.call("ecrire_fichier", {"path": "a", "content": "b"}, allowed={"calculatrice"})
    with pytest.raises(ToolError, match="désactivé"):
        reg.call("lire_page_web", {"url": "https://example.com"}, allowed={"lire_page_web"})
    with pytest.raises(ToolError, match="inconnus"):
        reg.call("calculatrice", {"expression": "1", "x": 2}, allowed={"calculatrice"})
    # un agent qui appelle un outil interdit reçoit une erreur, pas une exception
    agent = build_default_agents().get("analyste")
    llm = EchoLLM([{"action": "tool", "tool": "ecrire_fichier", "args": {"path": "a", "content": "b"}},
                   {"action": "final", "answer": "fini"}])
    res = agent.run(llm, reg, "t", "")
    assert res.output == "fini" and res.tool_calls[0]["ok"] is False
    assert not (tmp_path / "ws" / "a").exists()


def test_workspace_sandbox(tmp_path):
    ws = Workspace(tmp_path / "ws")
    for bad in ("../x.txt", "/etc/passwd", "..\\x.txt", "sub/../../x"):
        with pytest.raises(ToolError):
            ws.write(bad, "x")
    ws.write("sub/ok.txt", "ok")
    assert ws.read("sub/ok.txt") == "ok"


def test_fetch_refuses_private_addresses():
    from jarvis.tools import fetch_url

    for url in ("http://127.0.0.1/", "http://localhost:8080/", "file:///etc/passwd", "http://10.0.0.1/"):
        with pytest.raises(ToolError):
            fetch_url(url)


def test_calculator_is_safe():
    assert calculate("(12*3)/4") == "9.0"
    assert calculate("sqrt(16) + pi") .startswith("7.14")
    for bad in ("__import__('os')", "open('x')", "9**99999", "(1).__class__"):
        with pytest.raises(ToolError):
            calculate(bad)


def test_extract_json():
    assert extract_json('Voici :\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('blabla {"a": "}"} fin') == {"a": "}"}
    assert extract_json("rien") is None
