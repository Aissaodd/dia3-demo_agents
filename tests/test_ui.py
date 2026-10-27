"""L'interface se charge sans erreur et reflète la base (Streamlit AppTest, sans navigateur)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from jarvis.memory import MissionStore  # noqa: E402

PAGES = ["chat", "missions", "approvals", "memory", "settings"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DB", str(tmp_path / "ui.db"))
    monkeypatch.setenv("JARVIS_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("JARVIS_LLM", "echo")
    store = MissionStore(tmp_path / "ui.db")
    mid = store.create_mission("Mission de démonstration")
    store.save_plan(mid, [{"title": "A", "instructions": "faire A", "agent": "analyste", "depends_on": []}])
    store.update_mission(mid, status="done", result="Résultat vérifié")
    store.remember("lesson", "Toujours vérifier les sources.")
    store.request_approval(mid, None, "ecrire_fichier", {"path": "x.md", "content": "y"}, "écrase x.md")
    return store


def run(path: Path) -> "AppTest":
    import jarvis.ui as ui
    ui.get_store.clear()
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_entry_loads(env):
    at = run(ROOT / "jarvis" / "dashboard.py")
    assert any("action(s) attendent" in w.value for w in at.sidebar.warning)


@pytest.mark.parametrize("page", PAGES)
def test_each_page_loads(env, page):
    run(ROOT / "jarvis" / "app_pages" / f"{page}.py")


def test_chat_conversation_roundtrip(env):
    at = run(ROOT / "jarvis" / "app_pages" / "chat.py")
    at.chat_input[0].set_value("/chat bonjour").run()
    assert not at.exception, at.exception
    texts = [m.markdown[0].value for m in at.chat_message if m.markdown]
    assert "/chat bonjour" in texts and any(t.startswith("[echo]") for t in texts)
    at.run()   # rafraîchissement : rien n'est doublé
    convs = env.list_conversations()
    assert len(env.get_messages(convs[0]["id"])) == 2


def test_emergency_stop_button(env):
    at = run(ROOT / "jarvis" / "dashboard.py")
    at.sidebar.button[0].click().run()
    assert env.emergency_stop()
