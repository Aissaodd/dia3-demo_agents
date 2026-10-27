"""JARVIS NEXUS — moteur de missions multi-agents.

Un orchestrateur central découpe un objectif en étapes, confie chaque étape
à un agent spécialisé, vérifie le résultat, réessaie en cas d'échec et garde
tout en mémoire (SQLite) pour pouvoir reprendre après une panne.
"""

__version__ = "0.2.0"
