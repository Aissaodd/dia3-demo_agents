"""Mémoire : ce que Jarvis sait, par type, avec recherche, ajout de préférences et oubli."""

import streamlit as st

from jarvis.memory import MEMORY_KINDS
from jarvis.ui import fmt_time, get_store

cfg, store = get_store()

KIND_LABELS = {"verified_result": "Résultats vérifiés", "lesson": "Leçons", "preference": "Préférences",
               "knowledge": "Connaissances", "hypothesis": "Hypothèses", "error_solution": "Erreurs résolues",
               "error": "Échecs", "mission": "Missions"}

with st.container(horizontal=True, vertical_alignment="bottom"):
    query = st.text_input("Rechercher", placeholder="mots-clés…", icon=":material/search:")
    kinds = st.pills("Types", [k for k in MEMORY_KINDS if k != "mission"], selection_mode="multi",
                     format_func=lambda k: KIND_LABELS.get(k, k))

if query:
    hits = store.search(query, limit=50, kinds=tuple(kinds) if kinds else None)
    if not hits:
        st.info("Aucun souvenir ne correspond.", icon=":material/info:")
    for h in hits:
        with st.container(border=True):
            if h["type"] == "mission":
                st.markdown(f"**Mission #{h['mission_id']}** — {h['objective']}")
                st.caption(h["result"][:400])
            else:
                st.caption(f"{KIND_LABELS.get(h['kind'], h['kind'])} · confiance {h['confidence']:.1f}"
                           + (f" · source : {h['source']}" if h.get("source") else ""))
                st.markdown(h["text"])
else:
    rows = [r for r in store.list_memories(limit=500) if not kinds or r["kind"] in kinds]
    if not rows:
        st.info("La mémoire est vide pour l'instant : elle se remplit à chaque mission.", icon=":material/info:")
    else:
        st.dataframe([{"#": r["id"], "Type": KIND_LABELS.get(r["kind"], r["kind"]), "Texte": r["text"],
                       "Source": r["source"] or "", "Confiance": r["confidence"], "Réutilisé": r["uses"],
                       "Créé": fmt_time(r["created_at"])} for r in rows], hide_index=True,
                     column_config={"Texte": st.column_config.TextColumn(width="large")})

left, right = st.columns(2)
with left, st.form("pref", clear_on_submit=True):
    text = st.text_input("Ajouter une préférence", placeholder="Ex. : réponds toujours en points courts")
    if st.form_submit_button("Mémoriser", icon=":material/add:") and text.strip():
        st.toast("Préférence enregistrée" if store.remember("preference", text, source="utilisateur")
                 else "Déjà connue")
with right, st.form("forget", clear_on_submit=True):
    mem_id = st.number_input("Oublier le souvenir n°", min_value=1, step=1)
    if st.form_submit_button("Oublier", icon=":material/delete:"):
        store.forget(int(mem_id))
        st.toast(f"Souvenir #{int(mem_id)} oublié")
st.caption("Les secrets, e-mails et numéros de téléphone sont masqués avant toute mémorisation ; "
           "les doublons ne sont pas recréés.")
