<<<<<<< HEAD
# JARVIS NEXUS

Un assistant personnel multi-agents. Vous lui parlez dans un chat ; quand votre
demande exige d'agir, il en fait une **mission** : il reformule l'objectif, le
découpe en étapes (exécutées en parallèle quand elles sont indépendantes), confie
chacune à l'agent compétent, fait **vérifier** chaque résultat par un vérificateur
indépendant, réessaie ou change d'agent si besoin, demande votre **accord** avant
toute action destructrice, et garde tout en **mémoire** (SQLite).

Le chat d'origine (`user_interface.py`, `llm_agent.py`, persona Barbe-Quasar) est
toujours là et fonctionne comme avant.

## Installation (Windows, PowerShell)

```powershell
cd dia3-demo_agents-main
python -m venv .venv
.venv\Scripts\Activate.ps1          # si bloqué : Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
Copy-Item .env.example .env         # puis renseignez DEEPSEEK_KEY (ou un autre fournisseur)
python -m jarvis doctor             # vérifie la configuration, sans appel payant
python -m jarvis dashboard          # interface sur http://localhost:8501
```

Sans clé : `JARVIS_LLM=echo` (ou `simulation`) fait tourner tout le moteur hors ligne,
gratuitement, avec des réponses factices. Avec un abonnement Claude et Claude Code
installé : `JARVIS_LLM=claude`.

## Fournisseurs de modèles

| `JARVIS_LLM` | Modèle par défaut | Clé (.env) | Facturé |
|---|---|---|---|
| `deepseek` | `deepseek-flash` (ou `deepseek-v4-pro`) | `DEEPSEEK_KEY` | oui |
| `groq` | `openai/gpt-oss-120b` | `GROQ_KEY` | oui |
| `anthropic` | `claude-sonnet-5-5` | `ANTHROPIC_API_KEY` | oui |
| `claude` | celui de Claude Code | aucune (`claude login`) | abonnement |
| `echo` | — | aucune | non |

- **DeepSeek a retiré `deepseek-chat` et `deepseek-reasoner` le 24/07/2026.** Si votre
  `.env` les utilise encore, `doctor` le signale : passez à `deepseek-flash`.
- Un modèle par rôle, explicitement : `JARVIS_MODEL_PLANNER=deepseek:deepseek-v4-pro`,
  `JARVIS_MODEL_VERIFIER=groq:openai/gpt-oss-120b`, `JARVIS_MODEL_AGENT_CODEUR=…`.
- **Clés de secours** : si le fournisseur est surchargé (503), limité (429), refuse la
  clé ou n'a plus de solde, Jarvis essaie ses autres clés (`GEMINI_API_KEY_2`,
  `GEMINI_API_KEY_3`… jusqu'à `_9`), puis les fournisseurs de `JARVIS_FALLBACK`
  (ex. `JARVIS_FALLBACK=deepseek,groq`), dans l'ordre. Sans ces réglages, aucun
  basculement. Chaque appel enregistre le fournisseur et le modèle réellement utilisés
  (page Configuration, `python -m jarvis usage`).
- `python -m jarvis test-provider deepseek` fait **un** appel réel (quelques jetons).
- Coûts : calculés seulement quand le prix du modèle est connu (prix officiels relevés
  le 27/10/2026, tarif le plus haut retenu). Sinon le coût est affiché « inconnu », jamais
  inventé. Surcharge : `JARVIS_PRICES='{"modele": [entree, sortie]}'` ($/million de jetons).

## Commandes

| Commande | Effet |
|---|---|
| `python -m jarvis dashboard` | interface : Chat, Missions, Approbations, Mémoire, Configuration |
| `python -m jarvis mission "objectif"` | crée et exécute une mission dans le terminal |
| `python -m jarvis submit "objectif"` | la lance en arrière-plan |
| `python -m jarvis list` / `show <id>` | suivre les missions |
| `python -m jarvis resume <id>` | reprendre une mission échouée, interrompue ou en attente |
| `python -m jarvis cancel <id>` | annuler une mission |
| `python -m jarvis approvals` / `approve <id>` / `deny <id>` | décider des actions sensibles |
| `python -m jarvis stop` / `stop --off` | **arrêt d'urgence** de toutes les missions / le lever |
| `python -m jarvis memory [recherche]` / `prefer "texte"` | consulter la mémoire / ajouter une préférence |
| `python -m jarvis usage` | appels, jetons, coût, latence par fournisseur et rôle |
| `python -m jarvis doctor` | diagnostic de configuration |
| `python -m jarvis agents` / `tools` | l'équipe et le registre d'outils |

Dans le chat : `/mission …` force une mission, `/suite …` prolonge la précédente
(son résultat est transmis), `/chat …` force une simple discussion. Sans préfixe,
un aiguilleur décide ; en cas de doute, Jarvis discute plutôt que d'agir.

## Architecture

```
chat ─► aiguilleur ─┬─► conversation (rôle chat)
                    └─► mission ─► Orchestrateur
                                     ├─ Planificateur : objectif reformulé, étapes, dépendances
                                     ├─ Ordonnanceur : étapes prêtes en parallèle (max_parallel)
                                     │    agent ─► outils (registre, permissions, approbations)
                                     │         ─► contrôles déterministes ─► Vérificateur
                                     │    échec ─► Récupération : réessai / autre agent / arrêt
                                     └─ Coordinateur : synthèse ─► mémoire (résultat vérifié, leçon)
```

| Fichier | Rôle |
|---|---|
| `jarvis/orchestrator.py` | missions : plan, ordonnancement parallèle, budgets, délais, arrêts, reprise |
| `jarvis/agents.py` | 9 agents : `generaliste`, `chercheur`, `analyste`, `codeur`, `redacteur`, `memoire` (exécution) ; `planificateur`, `verificateur`, `coordinateur`, `recuperation` (système) |
| `jarvis/verification.py` | contrôles déterministes + vérificateur ; un verdict illisible est un refus |
| `jarvis/recovery.py` | erreurs passagères/définitives, changement d'agent sans élévation de privilèges |
| `jarvis/tools.py` | registre d'outils : schémas, risques, limites, approbations |
| `jarvis/memory.py` | SQLite : missions, étapes, journal, appels de modèles, approbations, chat, mémoire typée |
| `jarvis/providers/` | DeepSeek, Groq, Anthropic, CLI Claude, echo ; routage par rôle |
| `jarvis/chat.py` | chat relié au moteur, idempotent |
| `jarvis/dashboard.py`, `jarvis/app_pages/`, `jarvis/ui.py` | interface Streamlit |
| `jarvis/config.py` | réglages lus dans `.env` |

**Reprise.** Chaque transition est écrite en base. Une erreur passagère (réseau, 429,
5xx, délai) est réessayée ; une erreur définitive (clé refusée, solde, modèle inconnu,
budget épuisé) arrête l'étape tout de suite et demande une intervention. Un résultat
refusé est refait avec le motif ; à la dernière tentative un autre agent prend le relais,
seulement s'il n'a pas plus de droits risqués. `resume` repart des étapes non finies.

**Mémoire.** Types distincts : résultat vérifié, leçon, préférence, connaissance (avec
source), hypothèse, erreur résolue, échec. Doublons ignorés ; secrets, e-mails et
téléphones masqués. Un agent ne peut mémoriser que des connaissances sourcées ou des
hypothèses. Une mission identique déjà réussie est signalée au planificateur.

## Sécurité

- Outil inconnu, désactivé ou absent de la liste de l'agent : **refusé**. Aucun outil
  n'exécute de commande système ; `analyser_code` lit le code sans l'exécuter.
- Fichiers limités à `workspace/` ; `.env`, clés, `.git`, bases de données protégés ;
  écriture atomique.
- **Approbation humaine** (`JARVIS_APPROVAL`) : `sensible` (défaut) pour écraser un
  fichier existant, `ecriture` pour toute écriture ou accès web. Une approbation vaut
  pour un appel précis et ne sert qu'une fois : rejouer l'étape ne refait pas l'action.
- Web : adresses privées et locales refusées, 1 Mo par page, `JARVIS_MAX_WEB_CALLS` par mission.
- Budget par mission (appels, jetons, coût), délais par tentative et par mission,
  annulation et arrêt d'urgence vérifiés avant chaque appel de modèle.
- Clés jamais affichées ni journalisées : masquage automatique dans le journal, les
  erreurs et la mémoire. Pages web, fichiers et réponses de modèles sont traités comme
  des données, jamais comme des ordres.
- Le CLI Claude tourne sans aucun outil (`--tools ""`), sans MCP, dans un dossier temporaire.

## Tests

```powershell
python -m pytest tests -q        # 60 tests, hors ligne, aucun appel payant
```
=======
# dia3-demo_agents
>>>>>>> 8827120f738a8dd46ee6195dab250ae160c8833d
