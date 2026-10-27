"""Chat central : converse avec JARVIS, ou lance une mission et suis-la en direct."""

import uuid

import streamlit as st

from jarvis.chat import ChatEngine
from jarvis.cli import launch_worker
from jarvis.ui import (ACTIVE, approvals_block, badge, effective_status, fmt_usage, get_store,
                       step_timeline)

cfg, store = get_store()


@st.cache_resource
def get_engine(db_path: str) -> ChatEngine:
    return ChatEngine(cfg, store)


engine = get_engine(str(store.db_path))

# --- conversation courante ---------------------------------------------
convs = store.list_conversations()
if "conv" not in st.session_state or st.session_state.conv not in [c["id"] for c in convs]:
    st.session_state.conv = convs[0]["id"] if convs else store.create_conversation()
    convs = store.list_conversations()

with st.sidebar:
    st.divider()
    if st.button("Nouvelle conversation", icon=":material/add:", width="stretch"):
        st.session_state.conv = store.create_conversation()
        st.rerun()
    titles = {c["id"]: c["title"] for c in convs}
    st.selectbox("Conversations", list(titles), key="conv", format_func=lambda i: titles.get(i, f"#{i}"))
    st.caption("`/mission …` force une mission · `/suite …` prolonge la précédente · `/chat …` force "
               "une simple discussion.")

conv = st.session_state.conv


@st.fragment(run_every="2s")
def live_mission(mid: int):
    """Carte de mission : rafraîchie tant qu'elle tourne, figée ensuite."""
    m = store.get_mission(mid)
    if m is None:
        return
    status = effective_status(store, m)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**Mission #{mid}**")
            badge(status)
            st.caption(fmt_usage(m))
            st.space("stretch")
            if status in ("running", "planning", "pending") and not m["cancel_requested"]:
                if st.button("Annuler", icon=":material/stop:", key=f"c{mid}"):
                    store.request_cancel(mid)
                    st.rerun(scope="fragment")
            if status in ("failed", "cancelled", "stale"):
                if st.button("Reprendre", icon=":material/replay:", key=f"r{mid}"):
                    launch_worker(cfg, mid)
                    store.log(mid, "resume_request", "Reprise demandée depuis le chat")
                    st.rerun(scope="fragment")
        step_timeline(store, mid)
        approvals_block(store, cfg, mid, key=f"chat{mid}")
        if m["error"]:
            st.error(m["error"], icon=":material/error:")
        if m["result"]:
            st.markdown(m["result"])


def static_mission(mid: int):
    m = store.get_mission(mid)
    if m is None:
        return
    status = effective_status(store, m)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**Mission #{mid}**")
            badge(status)
            st.caption(fmt_usage(m))
        with st.expander("Étapes et vérifications", icon=":material/account_tree:"):
            step_timeline(store, mid)
        if m["error"]:
            st.error(m["error"], icon=":material/error:")
        if m["result"]:
            st.markdown(m["result"])


# --- entrée -------------------------------------------------------------
prompt = st.chat_input("Parlez à JARVIS, ou décrivez une mission…", submit_mode="disable")
if prompt:
    # Une clé par envoi : le même envoi traité deux fois (rafraîchissement) ne double rien.
    key = f"ui:{conv}:{uuid.uuid4().hex}"
    with st.spinner("JARVIS réfléchit…"):
        engine.submit(conv, prompt, key)
else:
    # Message resté sans réponse (page fermée pendant le traitement) : on le termine, une seule fois.
    for m in engine.pending_messages(conv):
        engine.process(m["id"])

messages = store.get_messages(conv)
if not messages:
    st.markdown("## Bonjour. Que puis-je faire pour vous ?")
    st.caption("Exemples : « Résume https://… en 5 points et enregistre-le dans resume.md » · "
               "« Analyse le fichier script.py du workspace » · « Qu'avons-nous fait la semaine dernière ? »")

for msg in messages:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg["mission_id"]:
            st.caption(msg["content"])
            m = store.get_mission(msg["mission_id"])
            if m and (effective_status(store, m) in ACTIVE or m["status"] == "waiting"):
                live_mission(msg["mission_id"])
            else:
                static_mission(msg["mission_id"])
        else:
            st.markdown(msg["content"])
