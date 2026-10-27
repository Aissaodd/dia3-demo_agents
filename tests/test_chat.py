"""Chat connecté au moteur : aiguillage, idempotence, suite de mission (hors ligne)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.chat import ChatEngine  # noqa: E402
from jarvis.config import Config  # noqa: E402
from jarvis.memory import MissionStore  # noqa: E402
from jarvis.orchestrator import Orchestrator  # noqa: E402
from jarvis.providers import EchoLLM, LLMError  # noqa: E402


def make(tmp_path, script=None, responder=None):
    cfg = Config(db_path=tmp_path / "j.db", workspace=tmp_path / "ws", retry_delay=0)
    store = MissionStore(cfg.db_path)
    launched = []
    llm = EchoLLM(script, responder=responder)
    chat = ChatEngine(cfg, store, llm, launcher=launched.append)
    return chat, store, launched, llm, cfg


def test_conversation_gets_a_direct_reply(tmp_path):
    chat, store, launched, _, _ = make(tmp_path, [{"type": "conversation"}, "Bonjour, je suis JARVIS."])
    conv = store.create_conversation()
    chat.submit(conv, "Salut !", "k1")
    msgs = store.get_messages(conv)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "Bonjour, je suis JARVIS." and not launched
    assert store.list_conversations()[0]["title"] == "Salut !"


def test_mission_is_created_once_even_if_resubmitted(tmp_path):
    chat, store, launched, _, _ = make(tmp_path, [
        {"type": "mission", "objective": "Écrire un rapport dans r.md", "continues": False}])
    conv = store.create_conversation()
    chat.submit(conv, "fais-moi un rapport dans r.md", "k1")
    chat.submit(conv, "fais-moi un rapport dans r.md", "k1")      # rafraîchissement / double clic
    msg_id = store.get_messages(conv)[0]["id"]
    chat.process(msg_id)                                           # retraitement
    assert len(launched) == 1 and len(store.list_missions()) == 1
    assert len(store.get_messages(conv)) == 2
    m = store.list_missions()[0]
    assert m["conversation_id"] == conv and m["objective"] == "Écrire un rapport dans r.md"


def test_explicit_prefixes_skip_the_router(tmp_path):
    chat, store, launched, llm, _ = make(tmp_path)
    conv = store.create_conversation()
    chat.submit(conv, "/mission calcule 2**10", "k1")
    assert launched and llm.roles == []                             # aucun appel de modèle
    assert store.list_missions()[0]["objective"] == "calcule 2**10"


def test_follow_up_continues_previous_mission(tmp_path):
    chat, store, launched, llm, cfg = make(tmp_path)
    conv = store.create_conversation()
    chat.submit(conv, "/mission écris un poème", "k1")
    first = launched[0]
    orch = Orchestrator(cfg, store, EchoLLM())
    orch.run(first)
    chat.submit(conv, "/suite raccourcis-le", "k2")
    second = launched[1]
    assert store.get_mission(second)["parent_id"] == first
    orch.run(second)
    assert store.get_mission(second)["status"] == "done"


def test_parent_result_reaches_the_planner(tmp_path):
    chat, store, launched, _, cfg = make(tmp_path)
    conv = store.create_conversation()
    chat.submit(conv, "/mission écris un poème", "k1")
    llm = EchoLLM()
    orch = Orchestrator(cfg, store, llm)
    orch.run(launched[0])
    chat.submit(conv, "/suite raccourcis-le", "k2")
    orch.run(launched[1])
    planner_msgs = [m for m, r in zip(llm.seen, llm.roles) if r == "planner"]
    assert "SUITE de la mission" in planner_msgs[-1][1]["content"]


def test_unreadable_route_means_no_execution(tmp_path):
    chat, store, launched, _, _ = make(tmp_path, ["je sais pas", "D'accord."])
    conv = store.create_conversation()
    chat.submit(conv, "supprime tout", "k1")
    assert not launched and store.get_messages(conv)[-1]["content"] == "D'accord."


def test_model_failure_is_reported_not_hidden(tmp_path):
    chat, store, launched, _, _ = make(tmp_path, [LLMError("HTTP 401 (clé refusée)")])
    conv = store.create_conversation()
    chat.submit(conv, "bonjour", "k1")
    last = store.get_messages(conv)[-1]
    assert last["role"] == "assistant" and "401" in last["content"]
    calls = store.usage_summary()
    assert calls and calls[0]["errors"] == 1


def test_pending_messages_are_detected(tmp_path):
    chat, store, _, _, _ = make(tmp_path)
    conv = store.create_conversation()
    store.add_message(conv, "user", "message resté sans réponse", client_key="k1")
    pending = chat.pending_messages(conv)
    assert len(pending) == 1
    chat.process(pending[0]["id"])
    assert not chat.pending_messages(conv)
