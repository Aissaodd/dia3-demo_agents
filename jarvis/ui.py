"""Éléments partagés de l'interface Streamlit. Tout ce qui est affiché vient de la
base : l'interface ne simule jamais une activité qui n'existe pas."""

import json
import time

import streamlit as st

from jarvis.cli import launch_worker
from jarvis.config import Config
from jarvis.memory import MissionStore

LABELS = {
    "pending": ("En attente", "gray"), "planning": ("Planification", "blue"),
    "running": ("En cours", "blue"), "waiting": ("Approbation requise", "yellow"),
    "done": ("Terminée", "green"), "failed": ("Échec", "red"), "cancelled": ("Annulée", "orange"),
    "stale": ("Interrompue", "orange"), "skipped": ("Ignorée", "gray"),
}
ACTIVE = ("pending", "planning", "running")
STEP_ICONS = {"pending": ":material/radio_button_unchecked:", "waiting": ":material/pause_circle:",
              "skipped": ":material/block:"}


@st.cache_resource
def get_store() -> tuple[Config, MissionStore]:
    cfg = Config.from_env()
    return cfg, MissionStore(cfg.db_path)


def effective_status(store: MissionStore, m: dict) -> str:
    """Une mission « en cours » dont le verrou a expiré a perdu son worker."""
    if m["status"] in ("planning", "running") and not store.is_live(m):
        return "stale"
    return m["status"]


def badge(status: str):
    text, color = LABELS.get(status, (status, "gray"))
    st.badge(text, color=color)


def fmt_time(ts) -> str:
    return time.strftime("%d/%m %H:%M:%S", time.localtime(ts)) if ts else ""


def fmt_usage(m: dict) -> str:
    tokens = (m.get("input_tokens") or 0) + (m.get("output_tokens") or 0)
    cost = f" · {m['cost_usd']:.4f} $" if m.get("cost_usd") else ""
    unknown = " (+ coût inconnu)" if m.get("cost_unknown_calls") else ""
    return f"{m['llm_calls']} appels · {tokens:,} jetons{cost}{unknown}".replace(",", " ")


def decide(store: MissionStore, cfg: Config, approval_id: int, approve: bool):
    """Décide une approbation ; relance la mission quand plus rien n'attend."""
    try:
        a = store.decide_approval(approval_id, approve)
    except (KeyError, ValueError) as exc:
        st.toast(str(exc), icon=":material/info:")
        return
    if not store.list_approvals("pending", a["mission_id"]):
        launch_worker(cfg, a["mission_id"])
    st.toast("Action approuvée" if approve else "Action refusée")


def approvals_block(store: MissionStore, cfg: Config, mission_id: int | None = None, key: str = ""):
    for a in store.list_approvals("pending", mission_id):
        with st.container(border=True):
            st.markdown(f":orange[:material/gavel:] **{a['tool']}** — {a['reason']}  ·  mission #{a['mission_id']}")
            try:
                args = json.loads(a["args"])
            except ValueError:
                args = a["args"]
            st.json(args, expanded=False)
            with st.container(horizontal=True):
                if st.button("Approuver", icon=":material/thumb_up:", type="primary", key=f"ok{key}{a['id']}"):
                    decide(store, cfg, a["id"], True)
                    st.rerun()
                if st.button("Refuser", icon=":material/thumb_down:", key=f"no{key}{a['id']}"):
                    decide(store, cfg, a["id"], False)
                    st.rerun()


def step_timeline(store: MissionStore, mission_id: int, expanded_output: bool = False):
    """Les étapes réelles de la mission, en chronologie (agent, dépendances, verdict)."""
    steps = store.get_steps(mission_id)
    if not steps:
        st.caption("Plan pas encore établi.")
        return
    for s in steps:
        deps = f" ← {', '.join(str(d + 1) for d in s['deps'])}" if s["deps"] else ""
        label = f"{s['position'] + 1}. {s['title']} · :violet[{s['agent_used'] or s['agent']}]{deps}"
        if s["attempts"] > 1:
            label += f" · {s['attempts']} tentatives"
        state = {"running": "running", "done": "complete", "failed": "error"}.get(s["status"])
        box = (st.status(label, state=state, type="step", expanded=expanded_output and state == "error")
               if state else st.expander(label, icon=STEP_ICONS.get(s["status"]), type="step"))
        with box:
            st.caption(s["instructions"])
            if s["success_criteria"]:
                st.caption(f"Critères : {s['success_criteria']}")
            if s["output"]:
                st.text(s["output"][:4000])
            if s["verdict"]:
                st.caption(f":material/verified: Vérificateur : {s['verdict']}")
            if s["error"] and s["status"] != "done":
                st.caption(f":red[:material/error:] {s['error'][:500]}")
