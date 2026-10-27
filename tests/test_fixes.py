"""Régressions des défauts trouvés à l'audit (phase 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import cli  # noqa: E402
from jarvis.config import Config  # noqa: E402
from jarvis.llm import EchoLLM  # noqa: E402
from jarvis.memory import MissionStore  # noqa: E402
from jarvis.orchestrator import Orchestrator  # noqa: E402


def make(tmp_path, script, **kw):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws", retry_delay=0, **kw)
    store = MissionStore(cfg.db_path)
    orch = Orchestrator(cfg, store, EchoLLM(script))
    orch.sleep = lambda s: None
    return orch, store


def test_unreadable_verdict_is_not_a_success(tmp_path):
    # Le vérificateur répond n'importe quoi à chaque fois : la mission ne doit PAS réussir.
    orch, store = make(tmp_path, [
        {"steps": [{"title": "A", "instructions": "A", "agent": "analyste", "success_criteria": "x"}]},
    ] + ["bla"] * 30, max_step_attempts=3)
    mid = orch.submit("x")
    assert orch.run(mid)["status"] == "failed"


def test_worker_start_failure_marks_mission_failed(tmp_path, monkeypatch):
    db = tmp_path / "j.db"
    monkeypatch.setenv("JARVIS_DB", str(db))
    monkeypatch.setenv("JARVIS_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("JARVIS_LLM", "fournisseur_inexistant")
    store = MissionStore(db)
    mid = store.create_mission("objectif")
    assert cli.main(["run", str(mid)]) == 1
    m = store.get_mission(mid)
    assert m["status"] == "failed" and "Démarrage impossible" in m["error"]


def test_cancel_without_worker_is_immediate(tmp_path):
    store = MissionStore(tmp_path / "j.db")
    mid = store.create_mission("x")
    store.request_cancel(mid)
    assert store.get_mission(mid)["status"] == "cancelled"
