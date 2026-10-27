"""Approbations : les actions sensibles attendent une décision humaine."""

import json

import streamlit as st

from jarvis.ui import approvals_block, fmt_time, get_store

cfg, store = get_store()

st.caption(f"Politique actuelle : **{cfg.approval_mode}** (JARVIS_APPROVAL). Une approbation vaut pour une "
           "action précise et ne sert qu'une fois ; la mission reprend dès qu'il n'y a plus rien à décider.")

if not store.list_approvals("pending"):
    st.success("Aucune action en attente.", icon=":material/task_alt:")
approvals_block(store, cfg, key="page")

history = [a for a in store.list_approvals(None) if a["status"] != "pending"][:100]
if history:
    st.markdown("**Historique**")
    labels = {"approved": "approuvée", "denied": "refusée", "used": "exécutée"}
    st.dataframe([{"#": a["id"], "Mission": a["mission_id"], "Outil": a["tool"], "Motif": a["reason"],
                   "Décision": labels.get(a["status"], a["status"]), "Demandée": fmt_time(a["created_at"]),
                   "Décidée": fmt_time(a["decided_at"]),
                   "Arguments": json.dumps(json.loads(a["args"]), ensure_ascii=False)[:200]}
                  for a in history], hide_index=True)
