"""Missions : lancement, historique, suivi en direct, agents mobilisés, chronologie, erreurs."""

import time

import streamlit as st

from jarvis.cli import launch_worker
from jarvis.ui import (LABELS, approvals_block, badge, effective_status, fmt_time, fmt_usage, get_store,
                       step_timeline)

cfg, store = get_store()

with st.form("new_mission", clear_on_submit=True, border=True):
    objective = st.text_area("Nouvelle mission", height=90,
                             placeholder="Ex. : résume la page https://… en 5 points et enregistre le résumé "
                                         "dans resume.md")
    if st.form_submit_button("Lancer", type="primary", icon=":material/rocket_launch:") and objective.strip():
        mid = store.create_mission(objective)
        launch_worker(cfg, mid)
        st.session_state.selected = mid
        st.toast(f"Mission #{mid} lancée")


@st.fragment(run_every="3s")
def missions_view():
    stats = store.stats()
    missions = store.list_missions()
    stale = sum(1 for m in missions if effective_status(store, m) == "stale")
    with st.container(horizontal=True):
        st.metric("Missions", sum(stats.values()), border=True)
        st.metric("En cours", stats["running"] + stats["planning"] - stale, border=True)
        st.metric("Approbation requise", stats["waiting"], border=True)
        st.metric("Terminées", stats["done"], border=True)
        st.metric("Échecs / interrompues", stats["failed"] + stale, border=True)

    if not missions:
        st.info("Aucune mission pour l'instant. Décrivez un objectif ci-dessus ou dans le chat.",
                icon=":material/info:")
        return

    rows = [{
        "#": m["id"],
        "Statut": LABELS.get(effective_status(store, m), ("?",))[0],
        "Objectif": m["objective"],
        "Progression": (m["n_done"] / m["n_steps"]) if m["n_steps"] else 0.0,
        "Appels": m["llm_calls"],
        "Jetons": (m["input_tokens"] or 0) + (m["output_tokens"] or 0),
        "Coût ($)": m["cost_usd"] or 0.0,
        "Créée": fmt_time(m["created_at"]),
    } for m in missions]
    with st.expander("Historique", icon=":material/schedule:", expanded=True):
        st.dataframe(rows, hide_index=True, height=min(38 * (len(rows) + 1), 300),
                     column_config={
                         "Progression": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                        format="percent"),
                         "Objectif": st.column_config.TextColumn(width="large"),
                         "Coût ($)": st.column_config.NumberColumn(format="%.4f"),
                     })

    ids = [m["id"] for m in missions]
    if st.session_state.get("selected") not in ids:
        st.session_state.selected = ids[0]
    objectives = {m["id"]: m["objective"] for m in missions}
    mid = st.selectbox("Mission à suivre", ids, key="selected",
                       format_func=lambda i: f"#{i} — {objectives[i][:90]}")
    mission_detail(mid)


def mission_detail(mid: int):
    m = store.get_mission(mid)
    status = effective_status(store, m)
    steps = store.get_steps(mid)

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f"Mission #{mid}", anchor=False)
            badge(status)
            st.caption(fmt_usage(m))
            st.space("stretch")
            if status in ("failed", "cancelled", "stale", "pending"):
                if st.button("Reprendre", icon=":material/replay:", key=f"resume{mid}"):
                    launch_worker(cfg, mid)
                    store.log(mid, "resume_request", "Reprise demandée depuis le tableau de bord")
                    st.toast("Reprise lancée")
            if status in ("running", "planning", "pending", "waiting") and not m["cancel_requested"]:
                if st.button("Annuler", icon=":material/stop:", key=f"cancel{mid}"):
                    store.request_cancel(mid)
                    st.toast("Annulation demandée")
        st.markdown(m["objective"])
        if m.get("parent_id"):
            st.caption(f"Suite de la mission #{m['parent_id']}")
        if m["error"]:
            st.error(m["error"], icon=":material/error:")
        if status == "stale":
            st.warning("Le processus qui exécutait cette mission s'est arrêté. « Reprendre » repart des "
                       "étapes non terminées.", icon=":material/warning:")
        agents = sorted({s["agent_used"] or s["agent"] for s in steps})
        if agents:
            with st.container(horizontal=True):
                st.caption("Agents mobilisés :")
                for a in agents:
                    st.badge(a, icon=":material/smart_toy:", color="violet")
        approvals_block(store, cfg, mid, key="missions")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Étapes et vérification**")
        step_timeline(store, mid, expanded_output=True)
        if m["result"]:
            with st.container(border=True):
                st.markdown("**Résultat final**")
                st.markdown(m["result"])

    with right:
        events = store.get_events(mid, limit=300)
        errors = [e for e in events if e["kind"] in ("step_error", "step_rejected", "step_failed", "error",
                                                       "budget", "timeout", "mission_failed", "synth_error")]
        if errors:
            with st.expander(f"Erreurs et refus ({len(errors)})", icon=":material/error:"):
                for e in errors:
                    st.caption(f"{time.strftime('%H:%M:%S', time.localtime(e['ts']))} · {e['message']}")
        usage = store.usage_summary(mission_id=mid)
        if usage:
            with st.expander("Appels de modèles", icon=":material/monitoring:"):
                st.dataframe([{"Rôle": u["role"], "Modèle": f"{u['provider']}:{u['model'] or '-'}",
                               "Appels": u["calls"], "Erreurs": u["errors"],
                               "Jetons": (u["input_tokens"] or 0) + (u["output_tokens"] or 0),
                               "Latence moy. (ms)": round(u["avg_latency_ms"] or 0)} for u in usage],
                             hide_index=True)
        st.markdown("**Chronologie**")
        st.dataframe(
            [{"Heure": time.strftime("%H:%M:%S", time.localtime(e["ts"])), "Type": e["kind"],
              "Événement": e["message"]} for e in events],
            hide_index=True, height=480,
            column_config={"Événement": st.column_config.TextColumn(width="large")},
        )


missions_view()
