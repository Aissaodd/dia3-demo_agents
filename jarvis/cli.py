"""Ligne de commande : python -m jarvis <commande>.

  mission "objectif"      crée une mission et l'exécute ici
  submit "objectif"       crée une mission et la lance en arrière-plan
  run <id> [--force]      exécute / reprend une mission (premier plan)
  resume <id>             relance une mission en arrière-plan
  list | show <id>        suivre les missions
  cancel <id>             arrête une mission
  approvals               actions en attente d'approbation
  approve <id> | deny <id>  décide, puis relance la mission
  stop [--off]            ARRÊT D'URGENCE de toutes les missions (--off pour lever)
  memory [recherche]      ce que Jarvis a mémorisé
  prefer "texte"          enregistre une préférence
  usage                   appels de modèles, jetons, coût, latence
  doctor                  vérifie la configuration (sans appel payant)
  test-provider <spec>    UN appel réel au fournisseur (ex. deepseek, groq:openai/gpt-oss-20b)
  agents | tools          ce que Jarvis sait faire
  dashboard               ouvre l'interface
"""

import argparse
import os
import subprocess
import sys
import time

from jarvis.config import PROJECT_DIR, Config
from jarvis.memory import LeaseHeld, MissionStore

ICONS = {"pending": "·", "planning": "…", "running": "▶", "waiting": "⏸", "done": "✓", "failed": "✗",
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
    cost = f", {m['cost_usd']:.4f} $" if m.get("cost_usd") else ""
    print(f"#{m['id']} [{m['status']}] {m['objective']}")
    print(f"   {m['llm_calls']} appels, {m['input_tokens'] + m['output_tokens']} jetons{cost}")
    for s in store.get_steps(mid):
        deps = f" ← {', '.join(str(d + 1) for d in s['deps'])}" if s["deps"] else ""
        print(f"  {ICONS.get(s['status'], '?')} {s['position'] + 1}. {s['title']}{deps} "
              f"({s['agent_used'] or s['agent']}, {s['attempts']} tentative(s))")
        if s["status"] in ("failed", "waiting") and s["error"]:
            print(f"      motif : {s['error'][:200]}")
    for a in store.list_approvals("pending", mid):
        print(f"  ⏸ approbation #{a['id']} : {a['tool']} — {a['reason']}  (python -m jarvis approve {a['id']})")
    if m["error"]:
        print(f"\nErreur : {m['error']}")
    if m["result"]:
        print(f"\n{m['result']}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="jarvis", description="JARVIS NEXUS — moteur de missions")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("mission", "submit"):
        sp = sub.add_parser(name)
        sp.add_argument("objective", nargs="+")
    for name in ("run", "resume", "show", "cancel", "approve", "deny"):
        sp = sub.add_parser(name)
        sp.add_argument("id", type=int)
        if name == "run":
            sp.add_argument("--force", action="store_true", help="ignore le verrou d'un worker mort")
    sp = sub.add_parser("stop")
    sp.add_argument("--off", action="store_true", help="lève l'arrêt d'urgence")
    sp = sub.add_parser("memory")
    sp.add_argument("query", nargs="*")
    sp = sub.add_parser("prefer")
    sp.add_argument("text", nargs="+")
    sp = sub.add_parser("test-provider")
    sp.add_argument("spec")
    for name in ("list", "agents", "tools", "dashboard", "approvals", "usage", "doctor"):
        sub.add_parser(name)
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
        print(f"Mission #{mid} en cours ({', '.join(f'{k}={v}' for k, v in orch.llm.describe().items())})…",
              flush=True)
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
        print(f"Annulation demandée (statut : {store.get_mission(args.id)['status']}).")
    elif args.cmd == "approvals":
        rows = store.list_approvals("pending")
        if not rows:
            print("Aucune action en attente.")
        for a in rows:
            print(f"#{a['id']}  mission #{a['mission_id']}  {a['tool']} — {a['reason']}\n      {a['args'][:300]}")
    elif args.cmd in ("approve", "deny"):
        try:
            a = store.decide_approval(args.id, approve=args.cmd == "approve")
        except (KeyError, ValueError) as exc:
            print(exc)
            return 1
        print(f"Action #{a['id']} {'approuvée' if a['status'] == 'approved' else 'refusée'}.")
        if not store.list_approvals("pending", a["mission_id"]):
            print(f"Reprise de la mission #{a['mission_id']} (pid {launch_worker(cfg, a['mission_id'])}).")
    elif args.cmd == "stop":
        store.set_emergency_stop(not args.off)
        print("Arrêt d'urgence LEVÉ." if args.off else
              "ARRÊT D'URGENCE ACTIVÉ : les missions s'arrêtent avant leur prochain appel de modèle. "
              "Lever : python -m jarvis stop --off")
    elif args.cmd == "memory":
        q = " ".join(args.query)
        rows = store.search(q, limit=20) if q else store.list_memories(limit=50)
        for r in rows:
            kind = r.get("kind") or r.get("type")
            print(f"[{kind}] {(r.get('text') or r.get('objective') or '')[:200]}")
        if not rows:
            print("Rien en mémoire." if not q else "Aucun souvenir ne correspond.")
    elif args.cmd == "prefer":
        new = store.remember("preference", " ".join(args.text), source="utilisateur")
        print("Préférence enregistrée." if new else "Déjà connue.")
    elif args.cmd == "usage":
        rows = store.usage_summary()
        if not rows:
            print("Aucun appel de modèle enregistré.")
        for r in rows:
            cost = "?" if r["cost_unknown"] else f"{r['cost_usd'] or 0:.4f} $"
            print(f"{r['provider']}:{r['model'] or '-'} [{r['role']}]  {r['calls']} appels ({r['errors']} erreurs), "
                  f"{(r['input_tokens'] or 0) + (r['output_tokens'] or 0)} jetons, {cost}, "
                  f"{(r['avg_latency_ms'] or 0):.0f} ms en moyenne")
    elif args.cmd == "doctor":
        from jarvis.providers import parse_spec

        print(f"Fournisseur par défaut : {cfg.llm_provider}:{cfg.llm_model or '(modèle par défaut)'}")
        for role, spec in cfg.role_models.items():
            print(f"  rôle {role} → {':'.join(parse_spec(spec, cfg.llm_provider))}")
        print(f"Budget/mission : {cfg.max_llm_calls} appels, {cfg.max_tokens} jetons, {cfg.max_cost_usd} $ ; "
              f"{cfg.max_parallel} étapes en parallèle ; approbations : {cfg.approval_mode}")
        print(f"Arrêt d'urgence : {'ACTIF' if store.emergency_stop() else 'non'}")
        warns = cfg.warnings()
        for w in warns:
            print(f"⚠ {w}")
        print("Configuration OK." if not warns else f"{len(warns)} point(s) à corriger.")
        return 1 if warns else 0
    elif args.cmd == "test-provider":
        from jarvis.providers import build_llm, parse_spec

        prov, model = parse_spec(args.spec, cfg.llm_provider)
        try:
            llm = build_llm(prov, model, timeout=60)
            if llm.paid:
                print(f"Appel réel et facturé à {llm.label} (quelques jetons)…", flush=True)
            text, usage = llm.complete_with_usage([
                {"role": "system", "content": "Réponds uniquement en JSON."},
                {"role": "user", "content": 'Renvoie exactement {"ok": true, "jarvis": "nexus"}'}])
        except Exception as exc:
            print(f"ÉCHEC {prov}:{model or '(défaut)'} — {exc}")
            return 1
        from jarvis.providers import extract_json

        data = extract_json(text)
        valid = isinstance(data, dict) and data.get("ok") is True
        cost = "inconnu" if usage.cost_usd is None else f"{usage.cost_usd:.6f} $"
        print(f"{'OK' if valid else 'RÉPONSE INVALIDE'} {usage.provider}:{usage.model or '(défaut)'} — "
              f"{usage.latency_ms} ms, jetons {usage.input_tokens}/{usage.output_tokens}, coût {cost}")
        print(f"Réponse : {text[:200]}")
        return 0 if valid else 1
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
                print(f"{a.name:<14} [{a.kind}] {a.description}")
                if a.tools:
                    print(f"{'':<14} outils : {', '.join(a.tools)}")
        else:
            reg = build_default_registry(cfg, store)
            for t in reg.describe():
                state = "actif" if t["enabled"] else "désactivé"
                print(f"{t['name']:<17} [{t['category']}, risque {t['level']}, {state}] {t['description']}")
                if t["limits"]:
                    print(f"{'':<17} limites : {t['limits']}")
    elif args.cmd == "dashboard":
        return subprocess.call([sys.executable, "-m", "streamlit", "run",
                                str(PROJECT_DIR / "jarvis" / "dashboard.py")], cwd=PROJECT_DIR)
    return 0
