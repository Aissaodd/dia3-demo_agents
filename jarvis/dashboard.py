"""Tableau de bord Jarvis : streamlit run jarvis/dashboard.py (ou python -m jarvis dashboard).

Il ne fait que lire la base et lancer des workers : fermer la page
n'arrête pas les missions en cours.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from jarvis.agents import build_default_agents  # noqa: E402
from jarvis.cli import launch_worker  # noqa: E402
from jarvis.config import Config  # noqa: E402
from jarvis.memory import MissionStore  # noqa: E402
from jarvis.tools import build_default_registry  # noqa: E402

LABELS = {
    "pending": ("En attente", "gray"), "planning": ("Planification", "blue"),
    "running": ("En cours", "blue"), "done": ("Terminée", "green"),
    "failed": ("Échec", "red"), "cancelled": ("Annulée", "orange"),
    "stale": ("Interrompue", "orange"), "skipped": ("Ignorée", "gray"),
}
STEP_ICONS = {"pending": ":material/radio_button_unchecked:", "running": ":material/progress_activity:",
              "done": ":material/check_circle:", "failed": ":material/error:", "skipped": ":material/block:"}


@st.cache_resource
def get_store() -> tuple[Config, MissionStore]:
    cfg = Config.from_env()
    return cfg, MissionStore(cfg.db_path)


cfg, store = get_store()


def effective_status(m: dict) -> str:
    """Une mission « en cours » dont le verrou a expiré a perdu son worker."""
    if m["status"] in ("planning", "running") and not store.is_live(m):
        return "stale"
    return m["status"]


def badge(status: str):
    text, color = LABELS.get(status, (status, "gray"))
    st.badge(text, color=color)


def fmt_time(ts) -> str:
    return time.strftime("%d/%m %H:%M:%S", time.localtime(ts)) if ts else ""


st.set_page_config(page_title="Jarvis — Missions", page_icon=":material/smart_toy:", layout="wide")
st.title("Jarvis", anchor=False)
st.caption(f"Moteur de missions · modèle : **{cfg.llm_provider}** ({cfg.llm_model})")

with st.sidebar:
    st.subheader("Nouvelle mission", anchor=False)
    with st.form("new_mission", clear_on_submit=True, border=False):
        objective = st.text_area("Objectif", placeholder="Ex. : résume la page https://… en 5 points "
                                 "et enregistre le résumé dans resume.md", height=140)
        if st.form_submit_button("Lancer", type="primary", icon=":material/rocket_launch:",
                                 width="stretch") and objective.strip():
            mid = store.create_mission(objective)
            launch_worker(cfg, mid)
            st.session_state.selected = mid
            st.toast(f"Mission #{mid} lancée")

tab_missions, tab_team = st.tabs(["Missions", "Agents et outils"])


@st.fragment(run_every="3s")
def missions_view():
    stats = store.stats()
    missions = store.list_missions()
    stale = sum(1 for m in missions if effective_status(m) == "stale")
    with st.container(horizontal=True):
        st.metric("Missions", sum(stats.values()), border=True)
        st.metric("En cours", stats["running"] + stats["planning"] - stale, border=True)
        st.metric("Terminées", stats["done"], border=True)
        st.metric("Échecs / interrompues", stats["failed"] + stale, border=True)

    if not missions:
        st.info("Aucune mission pour l'instant. Décris un objectif dans le panneau de gauche.",
                icon=":material/info:")
        return

    rows = [{
        "#": m["id"],
        "Statut": LABELS.get(effective_status(m), ("?",))[0],
        "Objectif": m["objective"],
        "Progression": (m["n_done"] / m["n_steps"]) if m["n_steps"] else 0.0,
        "Appels LLM": m["llm_calls"],
        "Créée": fmt_time(m["created_at"]),
    } for m in missions]
    with st.container(border=True):
        st.markdown("**Historique**")
        st.dataframe(rows, hide_index=True, height=min(38 * (len(rows) + 1), 300),
                     column_config={
                         "Progression": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                         "Objectif": st.column_config.TextColumn(width="large"),
                     })

    ids = [m["id"] for m in missions]
    if st.session_state.get("selected") not in ids:
        st.session_state.selected = ids[0]
    mid = st.selectbox("Mission à suivre", ids, key="selected",
                       format_func=lambda i: f"#{i} — {next(m['objective'] for m in missions if m['id'] == i)[:90]}")
    mission_detail(mid)


def mission_detail(mid: int):
    m = store.get_mission(mid)
    status = effective_status(m)
    steps = store.get_steps(mid)

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f"Mission #{mid}", anchor=False)
            badge(status)
            st.space("stretch")
            if status in ("failed", "cancelled", "stale", "pending"):
                if st.button("Reprendre", icon=":material/replay:", key=f"resume{mid}"):
                    launch_worker(cfg, mid)
                    store.log(mid, "resume_request", "Reprise demandée depuis le tableau de bord")
                    st.toast("Reprise lancée")
            if status in ("running", "planning") and not m["cancel_requested"]:
                if st.button("Annuler", icon=":material/stop:", key=f"cancel{mid}"):
                    store.request_cancel(mid)
                    st.toast("Annulation demandée")
        st.markdown(m["objective"])
        if m["error"]:
            st.error(m["error"], icon=":material/error:")
        if status == "stale":
            st.warning("Le processus qui exécutait cette mission s'est arrêté. "
                       "« Reprendre » repart de la première étape non terminée.", icon=":material/warning:")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Étapes**")
        if not steps:
            st.caption("Plan pas encore établi.")
        for s in steps:
            title = f"{s['position'] + 1}. {s['title']}  ·  {s['agent_used'] or s['agent']}"
            if s["attempts"] > 1:
                title += f"  ·  {s['attempts']} tentatives"
            with st.expander(title, icon=STEP_ICONS.get(s["status"]), expanded=s["status"] in ("running", "failed")):
                st.markdown(f"**Consigne** : {s['instructions']}")
                if s["success_criteria"]:
                    st.markdown(f"**Critères** : {s['success_criteria']}")
                if s["output"]:
                    st.markdown("**Résultat**")
                    st.text(s["output"][:5000])
                if s["verdict"]:
                    st.caption(f"Vérificateur : {s['verdict']}")
                if s["error"] and s["status"] != "done":
                    st.caption(f"Dernier motif d'échec : {s['error']}")
        if m["result"]:
            with st.container(border=True):
                st.markdown("**Résultat final**")
                st.markdown(m["result"])

    with right:
        st.markdown("**Journal**")
        events = store.get_events(mid, limit=200)
        st.dataframe(
            [{"Heure": time.strftime("%H:%M:%S", time.localtime(e["ts"])), "Type": e["kind"],
              "Événement": e["message"]} for e in events],
            hide_index=True, height=520,
            column_config={"Événement": st.column_config.TextColumn(width="large")},
        )


def team_view():
    agents = build_default_agents()
    tools = build_default_registry(cfg, store)
    st.markdown("**Agents** — interchangeables : l'orchestrateur ne connaît que leur nom.")
    st.dataframe([{"Agent": a.name, "Rôle": a.description, "Outils autorisés": ", ".join(a.tools)}
                  for a in agents.all()], hide_index=True)
    st.markdown("**Registre des outils**")
    st.dataframe([{"Outil": t.name, "Type": t.risk,
                   "État": "désactivé" if t.name in tools.disabled else "actif",
                   "Description": t.description} for t in tools.all()], hide_index=True)
    st.caption(f"Les outils fichiers sont limités à `{cfg.workspace}`. "
               "Désactiver un outil : JARVIS_DISABLED_TOOLS=lire_page_web dans .env")


with tab_missions:
    missions_view()
with tab_team:
    team_view()
