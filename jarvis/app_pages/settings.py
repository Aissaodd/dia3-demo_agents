"""Configuration : fournisseurs et modèles par rôle, budgets, agents, outils, usage réel."""

import os
import time

import streamlit as st

from jarvis.agents import build_default_agents
from jarvis.providers import PROVIDER_KEYS, build_llm, parse_spec
from jarvis.tools import build_default_registry
from jarvis.ui import get_store

cfg, store = get_store()

st.markdown("**Modèles**")
st.caption("Choisis dans `.env` (JARVIS_LLM, JARVIS_MODEL, JARVIS_MODEL_<RÔLE>). Aucun basculement automatique "
           "d'un fournisseur vers un autre. Les clés ne sont jamais affichées.")
specs = {"défaut": f"{cfg.llm_provider}:{cfg.llm_model}"}
specs.update(cfg.role_models)
rows = []
for role, spec in specs.items():
    prov, model = parse_spec(spec, cfg.llm_provider)
    keys = PROVIDER_KEYS.get(prov, ())
    rows.append({"Rôle": role, "Fournisseur": prov, "Modèle": model or "(défaut du fournisseur)",
                 "Clé": "non requise" if not keys else ("présente" if any(os.environ.get(k) for k in keys)
                                                        else f"MANQUANTE ({keys[0]})")})
st.dataframe(rows, hide_index=True)

with st.expander("Tester un fournisseur (appel réel)", icon=":material/bolt:"):
    st.caption("Un seul appel de quelques jetons. Facturé pour deepseek, groq et anthropic.")
    with st.form("test_provider", border=False):
        spec = st.text_input("Fournisseur:modèle", value=specs["défaut"].rstrip(":"))
        if st.form_submit_button("Tester", icon=":material/bolt:"):
            prov, model = parse_spec(spec, cfg.llm_provider)
            try:
                text, usage = build_llm(prov, model, timeout=60).complete_with_usage([
                    {"role": "user", "content": 'Réponds exactement {"ok": true}'}])
                cost = "inconnu" if usage.cost_usd is None else f"{usage.cost_usd:.6f} $"
                st.success(f"{usage.provider}:{usage.model or '(défaut)'} a répondu en {usage.latency_ms} ms "
                           f"({usage.input_tokens}/{usage.output_tokens} jetons, coût {cost}) : {text[:120]}",
                           icon=":material/task_alt:")
            except Exception as exc:
                st.error(f"Échec : {exc}", icon=":material/error:")

st.markdown("**Garde-fous par mission**")
with st.container(horizontal=True):
    st.metric("Appels max", cfg.max_llm_calls, border=True)
    st.metric("Jetons max", f"{cfg.max_tokens:,}".replace(",", " "), border=True)
    st.metric("Coût max", f"{cfg.max_cost_usd} $", border=True)
    st.metric("Pages web max", cfg.max_web_calls, border=True)
    st.metric("Étapes en parallèle", cfg.max_parallel, border=True)
st.caption(f"Délais : {cfg.step_timeout:.0f} s par tentative, {cfg.mission_timeout:.0f} s par mission · "
           f"{cfg.max_step_attempts} tentatives par étape · approbations : {cfg.approval_mode}")

st.markdown("**Usage réel des modèles**")
period = st.segmented_control("Période", ["24 h", "7 jours", "Tout"], default="7 jours")
since = {"24 h": time.time() - 86400, "7 jours": time.time() - 7 * 86400}.get(period)
usage = store.usage_summary(since=since)
if not usage:
    st.caption("Aucun appel enregistré sur la période.")
else:
    total_cost = sum(u["cost_usd"] or 0 for u in usage)
    unknown = sum(u["cost_unknown"] or 0 for u in usage)
    with st.container(horizontal=True):
        st.metric("Appels", sum(u["calls"] for u in usage), border=True)
        st.metric("Jetons", f"{sum((u['input_tokens'] or 0) + (u['output_tokens'] or 0) for u in usage):,}"
                  .replace(",", " "), border=True)
        st.metric("Coût connu", f"{total_cost:.4f} $", border=True,
                  help=f"{unknown} appel(s) au prix inconnu ne sont pas comptés" if unknown else None)
        st.metric("Erreurs", sum(u["errors"] or 0 for u in usage), border=True)
    st.dataframe([{"Fournisseur": u["provider"], "Modèle": u["model"] or "-", "Rôle": u["role"],
                   "Appels": u["calls"], "Erreurs": u["errors"],
                   "Jetons": (u["input_tokens"] or 0) + (u["output_tokens"] or 0),
                   "Coût ($)": u["cost_usd"], "Latence moy. (ms)": round(u["avg_latency_ms"] or 0)}
                  for u in usage], hide_index=True,
                 column_config={"Coût ($)": st.column_config.NumberColumn(format="%.4f")})

st.markdown("**Agents** — interchangeables : l'orchestrateur ne connaît que leur nom.")
st.dataframe([{"Agent": a.name, "Type": "exécution" if a.kind == "worker" else "système",
               "Rôle": a.description, "Compétences": ", ".join(a.skills), "Outils": ", ".join(a.tools)}
              for a in build_default_agents().all()], hide_index=True)
st.markdown("**Registre des outils** — tout outil absent de ce registre est refusé.")
st.dataframe([{"Outil": t["name"], "Catégorie": t["category"], "Risque": t["level"],
               "État": "actif" if t["enabled"] else "désactivé", "Limites": t["limits"],
               "Description": t["description"]} for t in build_default_registry(cfg, store).describe()],
             hide_index=True)
st.caption(f"Workspace : `{cfg.workspace}` · aucun outil n'exécute de commande système.")
