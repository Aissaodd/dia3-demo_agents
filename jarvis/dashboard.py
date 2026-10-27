"""Interface JARVIS NEXUS : python -m jarvis dashboard (ou streamlit run jarvis/dashboard.py).

Elle lit la base et lance des workers : fermer la page n'arrête pas les missions.
Pages : Chat, Missions, Approbations, Mémoire, Configuration.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from jarvis.ui import get_store  # noqa: E402

st.set_page_config(page_title="JARVIS NEXUS", page_icon=":material/smart_toy:", layout="wide")
cfg, store = get_store()

PAGES = ROOT / "jarvis" / "app_pages"
n_pending = len(store.list_approvals("pending"))
page = st.navigation([
    st.Page(PAGES / "chat.py", title="Chat", icon=":material/chat:", default=True),
    st.Page(PAGES / "missions.py", title="Missions", icon=":material/rocket_launch:"),
    st.Page(PAGES / "approvals.py", title=f"Approbations ({n_pending})" if n_pending else "Approbations",
            icon=":material/gavel:"),
    st.Page(PAGES / "memory.py", title="Mémoire", icon=":material/memory:"),
    st.Page(PAGES / "settings.py", title="Configuration", icon=":material/tune:"),
], position="top")

with st.sidebar:
    st.markdown("### :primary[JARVIS] NEXUS")
    st.caption(f"Modèle : **{cfg.llm_provider}**" + (f" · {cfg.llm_model}" if cfg.llm_model else "")
               + (f" · {len(cfg.role_models)} rôle(s) dédié(s)" if cfg.role_models else ""))
    stats = store.stats()
    with st.container(horizontal=True):
        st.metric("Actives", stats["running"] + stats["planning"] + stats["pending"])
        st.metric("En attente", stats["waiting"])
    if n_pending:
        st.warning(f"{n_pending} action(s) attendent votre décision.", icon=":material/gavel:")
    for w in cfg.warnings():
        st.warning(w, icon=":material/warning:")
    st.divider()
    if store.emergency_stop():
        st.error("Arrêt d'urgence actif : aucune mission ne s'exécute.", icon=":material/emergency:")
        if st.button("Lever l'arrêt d'urgence", icon=":material/power_settings_new:", width="stretch"):
            store.set_emergency_stop(False)
            st.rerun()
    elif st.button("Arrêt d'urgence", icon=":material/emergency:", type="primary", width="stretch",
                   help="Arrête toutes les missions avant leur prochain appel de modèle."):
        store.set_emergency_stop(True)
        st.rerun()

page.run()
