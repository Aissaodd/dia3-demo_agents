"""Ligne de commande : python -m jarvis <commande>.

  mission "objectif"   crée une mission et l'exécute ici
  submit "objectif"    crée une mission et la lance en arrière-plan
  run <id>             exécute / reprend une mission (premier plan)
  resume <id>          relance une mission en arrière-plan
  list                 liste les missions
  show <id>            détail d'une mission
  cancel <id>          demande l'arrêt d'une mission
  agents | tools       ce que Jarvis sait faire
  dashboard            ouvre le tableau de bord
"""

import argparse
import os
import subprocess
import sys
import time

from jarvis.config import PROJECT_DIR, Config
from jarvis.memory import LeaseHeld, MissionStore

ICONS = {"pending": "·", "planning": "…", "running": "▶", "done": "✓", "failed": "✗",
         "cancelled": "■", "skipped": "-"}


def launch_worker(cfg: Config, mission_id: int) -> int:
    """Exécute la mission dans un processus séparé (survit au tableau de bord)."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log = open(cfg.logs_dir / f"mission-{mission_id}.log", "a", encoding="utf-8")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [sys.executable, "-m", "jarvis", "run", str(mission_id)],
        cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **kwargs,
    )
    return proc.pid


def _print_mission(store: MissionStore, mid: int):
    m = store.get_mission(mid)
    if not m:
        print(f"Mission {mid} introuvable.")
        return
    print(f"#{m['id']} [{m['status']}] {m['objective']}")
    for s in store.get_steps(mid):
        print(f"  {ICONS.get(s['status'], '?')} {s['position'] + 1}. {s['title']} "
              f"({s['agent_used'] or s['agent']}, {s['attempts']} tentative(s))")
        if s["status"] == "failed" and s["error"]:
            print(f"      motif : {s['error'][:200]}")
    if m["error"]:
        print(f"\nErreur : {m['error']}")
    if m["result"]:
        print(f"\n{m['result']}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="jarvis", description="Moteur de missions Jarvis")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("mission", "submit"):
        sp = sub.add_parser(name)
        sp.add_argument("objective", nargs="+")
    for name in ("run", "resume", "show", "cancel"):
        sp = sub.add_parser(name)
        sp.add_argument("id", type=int)
        if name == "run":
            sp.add_argument("--force", action="store_true", help="ignore le verrou d'un worker mort")
    sub.add_parser("list")
    sub.add_parser("agents")
    sub.add_parser("tools")
    sub.add_parser("dashboard")
    args = p.parse_args(argv)

    cfg = Config.from_env()
    store = MissionStore(cfg.db_path)

    if args.cmd in ("mission", "run"):
        from jarvis.orchestrator import Orchestrator

        mid = store.create_mission(" ".join(args.objective)) if args.cmd == "mission" else args.id
        try:
            orch = Orchestrator(cfg, store)
        except Exception as exc:
            # Sans ça, une clé manquante laissait la mission « en attente » pour toujours.
            store.update_mission(mid, status="failed", error=f"Démarrage impossible : {exc}")
            store.log(mid, "error", f"Démarrage impossible : {exc}")
            print(f"Démarrage impossible : {exc}")
            return 1
        print(f"Mission #{mid} en cours ({cfg.llm_provider})…", flush=True)
        try:
            orch.run(mid, force=getattr(args, "force", False))
        except LeaseHeld as exc:
            print(f"{exc}. Attends, ou relance avec --force si ce processus est mort.")
            return 1
        _print_mission(store, mid)
        return 0 if store.get_mission(mid)["status"] == "done" else 1

    if args.cmd == "submit":
        mid = store.create_mission(" ".join(args.objective))
        print(f"Mission #{mid} lancée en arrière-plan (pid {launch_worker(cfg, mid)}).")
    elif args.cmd == "resume":
        print(f"Reprise de la mission #{args.id} (pid {launch_worker(cfg, args.id)}).")
    elif args.cmd == "show":
        _print_mission(store, args.id)
    elif args.cmd == "cancel":
        store.request_cancel(args.id)
        print("Annulation demandée ; elle prend effet à la fin de l'appel en cours.")
    elif args.cmd == "list":
        for m in store.list_missions():
            when = time.strftime("%d/%m %H:%M", time.localtime(m["created_at"]))
            print(f"#{m['id']:<4} {ICONS.get(m['status'], '?')} {m['status']:<9} {when}  "
                  f"{m['n_done']}/{m['n_steps']}  {m['objective'][:70]}")
    elif args.cmd in ("agents", "tools"):
        from jarvis.agents import build_default_agents
        from jarvis.tools import build_default_registry

        if args.cmd == "agents":
            for a in build_default_agents().all():
                print(f"{a.name:<12} {a.description}\n{'':<12} outils : {', '.join(a.tools)}")
        else:
            reg = build_default_registry(cfg, store)
            for t in reg.all():
                state = "désactivé" if t.name in reg.disabled else "actif"
                print(f"{t.name:<17} [{t.risk}, {state}] {t.description}")
    elif args.cmd == "dashboard":
        return subprocess.call([sys.executable, "-m", "streamlit", "run",
                                str(PROJECT_DIR / "jarvis" / "dashboard.py")], cwd=PROJECT_DIR)
    return 0
