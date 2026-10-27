"""Tests JARVIS NEXUS : fournisseurs, parallélisme, budgets, approbations, mémoire,
reprise et sécurité. Tout tourne hors ligne (aucune clé d'API, aucun appel payant)."""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import recovery  # noqa: E402
from jarvis.agents import Agent, build_default_agents, plan  # noqa: E402
from jarvis.config import Config  # noqa: E402
from jarvis.memory import MissionStore  # noqa: E402
from jarvis.orchestrator import Orchestrator  # noqa: E402
from jarvis.providers import (BudgetExceeded, EchoLLM, LLMError, LLMRouter, build_llm, parse_spec,  # noqa: E402
                              redact)
from jarvis.providers.base import LLM, http_error  # noqa: E402
from jarvis.tools import ToolContext, ToolError, Workspace, build_default_registry  # noqa: E402
from jarvis.verification import verify  # noqa: E402


def make(tmp_path, script=None, responder=None, **cfg_kw):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws", retry_delay=0, **cfg_kw)
    store = MissionStore(cfg.db_path)
    llm = EchoLLM(script, responder=responder)
    orch = Orchestrator(cfg, store, llm)
    orch.sleep = lambda s: None
    return orch, store, llm


def step(title, agent="analyste", deps=None, criteria="ok"):
    s = {"title": title, "instructions": title, "agent": agent, "success_criteria": criteria}
    if deps is not None:
        s["depends_on"] = deps
    return s


# --- fournisseurs ---------------------------------------------------------

def test_parse_spec_and_router_by_role():
    assert parse_spec("deepseek:deepseek-v4-pro", "groq") == ("deepseek", "deepseek-v4-pro")
    assert parse_spec("echo", "deepseek") == ("echo", "")
    assert parse_spec("simulation", "deepseek") == ("echo", "")
    assert parse_spec("mon-modele", "groq") == ("groq", "mon-modele")
    a, b = EchoLLM(["A"]), EchoLLM(["B"])
    router = LLMRouter(a, {"planner": b})
    assert router.complete([{"role": "user", "content": "x"}], role="planner") == "B"
    assert router.complete([{"role": "user", "content": "x"}], role="agent:analyste") == "A"
    assert router.calls == 2


def test_missing_key_is_explicit_and_no_silent_fallback(monkeypatch):
    for k in ("DEEPSEEK_KEY", "DEEPSEEK_API_KEY", "GROQ_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for prov in ("deepseek", "groq", "anthropic"):
        with pytest.raises(LLMError, match="manquante"):
            build_llm(prov)
    with pytest.raises(LLMError, match="inconnu"):
        build_llm("fournisseur_bidon")


def test_retries_only_transient_errors():
    class Flaky(LLM):
        name, paid = "flaky", True

        def __init__(self, errors):
            super().__init__("m", retries=2, delay=0, prices={"m": (1.0, 2.0)})
            self.errors = list(errors)
            self.sleep = lambda s: None

        def _complete(self, messages, role=None):
            if self.errors:
                raise self.errors.pop(0)
            return "ok", 1000, 500

    llm = Flaky([http_error(429, "lent", retry_after="1"), http_error(503, "panne")])
    text, usage = llm.complete_with_usage([{"role": "user", "content": "x"}])
    assert text == "ok" and llm.calls == 3
    assert usage.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    perm = Flaky([http_error(401, "mauvaise clé")])
    with pytest.raises(LLMError, match="401"):
        perm.complete([{"role": "user", "content": "x"}])
    assert perm.calls == 1   # une erreur définitive n'est jamais réessayée


def test_unknown_price_is_not_invented():
    class P(LLM):
        name, paid = "p", True

        def _complete(self, messages, role=None):
            return "ok", 10, 10

    _, usage = P("modele-sans-prix", prices={}).complete_with_usage([{"role": "user", "content": "x"}])
    assert usage.cost_usd is None


def test_secrets_are_redacted(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_KEY", "sk-abcdef1234567890abcdef")
    assert "sk-abcdef" not in redact("clé : sk-abcdef1234567890abcdef refusée")
    assert "gsk_" not in redact("gsk_ABCDEFGHIJKLMNOPQRSTUV")
    assert "hunter2xyz" not in redact("password=hunter2xyz")
    store = MissionStore(tmp_path / "j.db")
    mid = store.create_mission("x")
    store.log(mid, "error", "HTTP 401 avec sk-abcdef1234567890abcdef", {"k": "sk-abcdef1234567890abcdef"})
    ev = store.get_events(mid)[0]
    assert "sk-abcdef" not in ev["message"] and "sk-abcdef" not in ev["data"]


# --- orchestration --------------------------------------------------------

def test_independent_steps_run_in_parallel_and_feed_dependents(tmp_path):
    active, peak, lock = [0], [0], threading.Lock()

    def responder(messages, role):
        if role == "planner":
            return {"understanding": "trois étapes", "steps": [
                step("Chercher A", "chercheur", []), step("Chercher B", "chercheur", []),
                step("Assembler", "redacteur", [1, 2])]}
        if role == "verifier":
            return {"ok": True, "reason": "ok"}
        if role == "synthesizer":
            return "Synthèse finale\nLEÇON: paralléliser les recherches."
        task = messages[1]["content"]
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.3)
        with lock:
            active[0] -= 1
        if "Assembler" in task:
            assert "résultat-A" in task and "résultat-B" in task
            return {"action": "final", "answer": "assemblé"}
        return {"action": "final", "answer": "résultat-A" if "Chercher A" in task else "résultat-B"}

    orch, store, _ = make(tmp_path, responder=responder, max_parallel=3)
    mid = orch.submit("trois étapes")
    m = orch.run(mid)
    assert m["status"] == "done", m["error"]
    assert peak[0] == 2                      # A et B en même temps, C après
    assert [s["deps"] for s in store.get_steps(mid)] == [[], [], [0, 1]]
    assert m["result"] == "Synthèse finale"


def test_plan_drops_forward_dependencies_and_picks_agent_by_skill():
    llm = EchoLLM([{"steps": [step("Écrire un script python", "agent_inexistant", [2, 5]),
                              step("Relire", "analyste", [1, 2])]}])
    steps = plan(llm, build_default_agents(), "obj", "", 5)
    assert steps[0]["depends_on"] == [] and steps[1]["depends_on"] == [0]
    assert steps[0]["agent"] == "codeur"     # choisi par compétences


def test_permanent_error_stops_without_retry(tmp_path):
    orch, store, _ = make(tmp_path, [{"steps": [step("A")]}, LLMError("HTTP 401 (clé refusée)")])
    mid = orch.submit("x")
    m = orch.run(mid)
    s = store.get_steps(mid)[0]
    assert m["status"] == "failed" and s["attempts"] == 1 and s["error_kind"] == "permanent"
    assert "intervention" in m["error"]


def test_budget_is_enforced(tmp_path):
    orch, store, _ = make(tmp_path, max_llm_calls=2)
    mid = orch.submit("x")
    m = orch.run(mid)   # plan (1) + agent (2) puis le vérificateur dépasserait le budget
    assert m["status"] == "failed" and "budget" in m["error"]
    assert m["llm_calls"] == 2


def test_step_timeout_then_retry(tmp_path):
    calls = {"n": 0}

    def responder(messages, role):
        if role and role.startswith("agent:"):
            calls["n"] += 1
            if calls["n"] == 1:
                time.sleep(0.25)       # la 1re tentative dépasse son délai…
                return {"action": "tool", "tool": "heure", "args": {}}
            return {"action": "final", "answer": "fini"}
        return None

    orch, store, _ = make(tmp_path, responder=responder, step_timeout=0.2)
    mid = orch.submit("x")
    m = orch.run(mid)
    assert m["status"] == "done"
    kinds = [e["kind"] for e in store.get_events(mid)]
    assert "step_error" in kinds and any("délai" in e["message"] for e in store.get_events(mid))


def test_emergency_stop(tmp_path):
    orch, store, _ = make(tmp_path)
    store.set_emergency_stop(True)
    mid = orch.submit("x")
    m = orch.run(mid)
    assert m["status"] == "pending" and "urgence" in m["error"] and m["llm_calls"] == 0
    store.set_emergency_stop(False)
    assert orch.run(mid)["status"] == "done"


def test_malformed_agent_reply_and_unknown_action(tmp_path):
    orch, store, _ = make(tmp_path, [
        {"steps": [step("A")]},
        {"action": "danser"},                 # action inconnue → on redemande
        "{ pas du json",                     # texte libre → pris comme réponse
        {"ok": True, "reason": "ok"},
    ])
    mid = orch.submit("x")
    m = orch.run(mid)
    assert m["status"] == "done" and m["result"] == "{ pas du json"


# --- approbations ---------------------------------------------------------

def overwrite_script(tmp_path):
    return [{"steps": [step("Mettre à jour notes.md", "redacteur", [])]},
            {"action": "tool", "tool": "ecrire_fichier", "args": {"path": "notes.md", "content": "v2"}}]


def test_overwrite_waits_for_approval_then_runs_once(tmp_path):
    orch, store, llm = make(tmp_path, overwrite_script(tmp_path))
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "ws" / "notes.md").write_text("v1", encoding="utf-8")
    mid = orch.submit("x")
    m = orch.run(mid)
    assert m["status"] == "waiting"
    pending = store.list_approvals("pending", mid)
    assert len(pending) == 1 and "écrase" in pending[0]["reason"]
    assert (tmp_path / "ws" / "notes.md").read_text(encoding="utf-8") == "v1"   # rien n'a bougé

    store.decide_approval(pending[0]["id"], approve=True)
    llm.script = [{"action": "tool", "tool": "ecrire_fichier", "args": {"path": "notes.md", "content": "v2"}},
                  {"action": "final", "answer": "notes.md mis à jour"}, {"ok": True, "reason": "ok"}]
    m = orch.run(mid)
    assert m["status"] == "done"
    assert (tmp_path / "ws" / "notes.md").read_text(encoding="utf-8") == "v2"
    assert store.list_approvals("used", mid)          # consommée : ne resservira pas
    assert not store.list_approvals("approved", mid)


def test_denied_action_is_reported_to_agent(tmp_path):
    orch, store, llm = make(tmp_path, overwrite_script(tmp_path))
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "ws" / "notes.md").write_text("v1", encoding="utf-8")
    mid = orch.submit("x")
    orch.run(mid)
    store.decide_approval(store.list_approvals("pending", mid)[0]["id"], approve=False)
    llm.script = [{"action": "tool", "tool": "ecrire_fichier", "args": {"path": "notes.md", "content": "v2"}},
                  {"action": "final", "answer": "Refusé par l'utilisateur, notes.md inchangé."},
                  {"ok": True, "reason": "ok"}]
    assert orch.run(mid)["status"] == "done"
    assert (tmp_path / "ws" / "notes.md").read_text(encoding="utf-8") == "v1"
    tool_events = [e for e in store.get_events(mid) if e["kind"] == "tool"]
    assert "refusée par l'utilisateur" in tool_events[0]["data"]


def test_cancel_denies_pending_approvals(tmp_path):
    orch, store, _ = make(tmp_path, overwrite_script(tmp_path))
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "ws" / "notes.md").write_text("v1", encoding="utf-8")
    mid = orch.submit("x")
    orch.run(mid)
    store.request_cancel(mid)
    assert store.get_mission(mid)["status"] == "cancelled"
    assert not store.list_approvals("pending", mid)


# --- vérification et reprise ---------------------------------------------

def test_missing_file_is_rejected_without_model(tmp_path):
    ws = Workspace(tmp_path / "ws")
    llm = EchoLLM()
    s = {"title": "Rapport", "instructions": "Rédige le rapport et enregistre-le dans rapport.md",
         "success_criteria": ""}
    v = verify(llm, s, "Voilà, c'est fait.", ws)
    assert not v.ok and "rapport.md" in v.reason and llm.seen == []
    ws.write("rapport.md", "# Rapport")
    assert verify(llm, s, "Rapport enregistré dans rapport.md", ws).ok


def test_safe_alternative_never_escalates_privileges(tmp_path):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws")
    tools = build_default_registry(cfg)
    agents = build_default_agents()
    agents.register(Agent("generaliste", "secours trop puissant", "x", ("ecrire_fichier", "lire_page_web")))
    alt = recovery.safe_alternative({"agent": "analyste", "title": "calcul", "instructions": "calcul"},
                                    "analyste", ["analyste"], agents, tools)
    assert alt != "generaliste" and alt not in ("redacteur", "codeur", "chercheur")


def test_classification():
    assert recovery.classify(BudgetExceeded("x")) == "budget"
    assert recovery.classify(LLMError("x", transient=True)) == "transient"
    assert recovery.classify(LLMError("x")) == "permanent"
    assert recovery.classify(recovery.StepTimeout()) == "timeout"
    assert not recovery.decide("permanent", 0, 3).retry
    assert recovery.decide("transient", 0, 3).retry and not recovery.decide("transient", 2, 3).retry


# --- mémoire --------------------------------------------------------------

def test_memory_kinds_dedup_and_sanitizing(tmp_path):
    store = MissionStore(tmp_path / "j.db")
    assert store.remember("knowledge", "Python 3.13 est sorti en octobre 2024", source="python.org")
    assert store.remember("knowledge", "  python 3.13 est SORTI en octobre 2024 ") is None   # doublon
    store.remember("hypothesis", "Python 3.14 sortira peut-être en octobre 2025", confidence=0.3)
    store.remember("preference", "Contact : jean@example.com, 06 12 34 56 78, password=abc123456")
    pref = store.list_memories("preference")[0]["text"]
    assert "example.com" not in pref and "06 12" not in pref and "abc123456" not in pref
    hits = store.search("python octobre")
    assert hits[0]["kind"] == "knowledge"          # le vérifié passe avant l'hypothèse
    with pytest.raises(ValueError):
        store.remember("secret", "x")


def test_memorize_tool_permissions(tmp_path):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws")
    store = MissionStore(cfg.db_path)
    reg = build_default_registry(cfg, store)
    ok = {"memoriser"}
    with pytest.raises(ToolError, match="refusé"):
        reg.call("memoriser", {"kind": "preference", "text": "x"}, ok)       # pas d'élévation
    with pytest.raises(ToolError, match="source"):
        reg.call("memoriser", {"kind": "knowledge", "text": "fait"}, ok)
    assert reg.call("memoriser", {"kind": "hypothesis", "text": "peut-être", "confidence": 0.4}, ok) == "mémorisé"


def test_legacy_lessons_are_migrated(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    MissionStore(db)
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM settings WHERE key = 'lessons_migrated'")
        c.execute("INSERT INTO lessons (mission_id, ts, text) VALUES (1, 0, 'toujours citer les sources')")
    store = MissionStore(db)
    assert any("citer les sources" in m["text"] for m in store.list_memories("lesson"))


def test_identical_mission_reuses_verified_result_in_planning(tmp_path):
    orch, store, llm = make(tmp_path)
    first = orch.submit("Quelle heure est-il à Paris ?")
    orch.run(first)
    second = orch.submit("quelle heure est-il à paris ?")
    orch.run(second)
    planner_msgs = [m for m, r in zip(llm.seen, llm.roles) if r == "planner"]
    assert "DÉJÀ RÉALISÉE" in planner_msgs[-1][1]["content"]


# --- outils ----------------------------------------------------------------

def test_protected_files_and_types(tmp_path):
    ws = Workspace(tmp_path / "ws")
    for bad in (".env", "sub/.env.local", "cle.pem", ".git/config", "data.db"):
        with pytest.raises(ToolError, match="protégé"):
            ws.write(bad, "x")
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws")
    reg = build_default_registry(cfg)
    with pytest.raises(ToolError, match="attendu"):
        reg.call("calculatrice", {"expression": ["1"]}, {"calculatrice"})
    with pytest.raises(ToolError, match="inconnu"):
        reg.call("shell", {"cmd": "rm -rf /"}, {"shell"})


def test_web_call_limit_per_mission(tmp_path):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws")
    store = MissionStore(cfg.db_path)
    reg = build_default_registry(cfg, store)
    mid = store.create_mission("x")
    for _ in range(2):
        store.log(mid, "tool", "lecture", {"tool": "lire_page_web"})
    ctx = ToolContext(mid, None, store, "sensible", max_web_calls=2)
    with pytest.raises(ToolError, match="limite"):
        reg.call("lire_page_web", {"url": "https://example.com"}, {"lire_page_web"}, ctx)


def test_write_policy_requires_approval_for_new_files(tmp_path):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws")
    reg = build_default_registry(cfg)
    with pytest.raises(ToolError, match="approbation"):   # sans contexte : refus par défaut
        reg.call("ecrire_fichier", {"path": "a.md", "content": "x"}, {"ecrire_fichier"},
                 ToolContext(approval_mode="ecriture"))


def test_analyze_code_never_executes(tmp_path):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws")
    reg = build_default_registry(cfg)
    Workspace(tmp_path / "ws").write("m.py", "import os\ndef f(x):\n    return eval(x)\nos.remove('x')\n")
    out = reg.call("analyser_code", {"path": "m.py"}, {"analyser_code"})
    assert "fonction f" in out and "eval" in out
    Workspace(tmp_path / "ws").write("bad.py", "def (:\n")
    assert "SYNTAXE" in reg.call("analyser_code", {"path": "bad.py"}, {"analyser_code"})


def test_usage_is_recorded_per_call(tmp_path):
    orch, store, _ = make(tmp_path)
    mid = orch.submit("x")
    orch.run(mid)
    rows = store.usage_summary(mission_id=mid)
    assert sum(r["calls"] for r in rows) == store.get_mission(mid)["llm_calls"] > 0
    assert {r["provider"] for r in rows} == {"echo"}


def test_calculator_explains_caret():
    from jarvis.tools import calculate
    with pytest.raises(ToolError, match=r"\*\*"):
        calculate("2^10")
    assert calculate("2**10") == "1024"
