# dia3 → Jarvis

Au départ : un chat DeepSeek + Streamlit (`user_interface.py`, `llm_agent.py`), toujours là.
Maintenant : **Jarvis**, un moteur de missions. Tu donnes un objectif, Jarvis le
découpe en étapes, confie chaque étape à un agent spécialisé, vérifie le résultat,
réessaie si besoin et garde tout en mémoire.

## Démarrer

```bash
pip install -r requirements.txt
cp .env.example .env          # mets ta DEEPSEEK_KEY
python -m jarvis dashboard    # tableau de bord dans le navigateur
```

Sans clé, `JARVIS_LLM=echo` fait tourner tout le moteur avec de fausses réponses.

## Commandes

| Commande | Effet |
|---|---|
| `python -m jarvis mission "objectif"` | crée et exécute une mission ici |
| `python -m jarvis submit "objectif"` | la lance en arrière-plan |
| `python -m jarvis list` / `show <id>` | suivre les missions |
| `python -m jarvis resume <id>` | reprendre une mission échouée ou interrompue |
| `python -m jarvis cancel <id>` | arrêter une mission |
| `python -m jarvis agents` / `tools` | l'équipe et les outils |

## Comment c'est construit

```
objectif ─► Orchestrateur ─► Planificateur ─► étapes (en base)
                 │
                 ├─► pour chaque étape : Agent ─► Outils (registre) ─► Vérificateur
                 │        refus → réessai avec le motif ; dernier essai → agent de secours
                 │
                 └─► Synthèse ─► résultat + leçon (mémoire)
```

| Fichier | Rôle |
|---|---|
| `jarvis/orchestrator.py` | orchestrateur central : plan, exécution, vérification, reprise, annulation |
| `jarvis/agents.py` | agents (`generaliste`, `chercheur`, `analyste`, `redacteur`), planificateur, vérificateur |
| `jarvis/tools.py` | registre des outils autorisés |
| `jarvis/memory.py` | mémoire SQLite : missions, étapes, journal, leçons |
| `jarvis/llm.py` | modèles interchangeables : DeepSeek, Claude Code, echo |
| `jarvis/dashboard.py` | tableau de bord Streamlit |

**Reprise après erreur.** Chaque étape est enregistrée dès qu'elle change d'état.
Une erreur du modèle est réessayée (avec délai croissant) ; un résultat refusé par
le vérificateur est refait avec le motif du refus ; à la dernière tentative l'agent
`generaliste` prend le relais. Si une étape échoue quand même, la mission passe en
« Échec » et `resume` repart de cette étape sans refaire les précédentes. Si le
processus meurt, son verrou expire (3 min) et le tableau de bord affiche
« Interrompue » avec un bouton Reprendre.

**Sécurité des outils.** Un outil n'est utilisable que s'il est enregistré, activé
et dans la liste de l'agent. Les fichiers sont limités au dossier `workspace/`, la
lecture web refuse les adresses locales ou privées, la calculatrice n'exécute pas
de code, et il n'y a aucun outil qui lance des commandes système. Le contenu des
pages et des fichiers est présenté aux agents comme des données, pas des ordres.

## Étendre Jarvis

Ajouter (ou remplacer) un agent :

```python
from jarvis.agents import Agent, build_default_agents
agents = build_default_agents()
agents.register(Agent("traducteur", "Traduit des textes.",
                      "Tu es un traducteur professionnel.", ("lire_fichier", "ecrire_fichier")))
Orchestrator(cfg, agents=agents)
```

Ajouter un outil :

```python
from jarvis.tools import Tool
tools.register(Tool("meteo", "Météo d'une ville.", {"ville": "nom"}, ma_fonction, risk="réseau",
                    required=("ville",)))
```

Le planificateur découvre tout seul les nouveaux agents (il lit leur description).

## Tests

```bash
python -m pytest tests -q
```
