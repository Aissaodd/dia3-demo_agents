"""Orchestrateur central.

Cycle d'une mission :
  1. verrou (un seul processus à la fois sur une mission) ;
  2. reprise : les étapes coupées en plein vol repartent ;
  3. plan (si pas encore fait), nourri par les missions passées similaires ;
  4. pour chaque étape : agent → vérificateur → réessai avec le motif du
     refus ; à la dernière tentative on passe la main à l'agent de secours ;
  5. synthèse, leçon enregistrée, mission terminée.
Chaque transition est écrite en base avant de passer à la suite : si le
processus meurt, ``run(mission_id)`` reprend à la première étape non finie.
"""

import time

from jarvis.agents import AgentRegistry, build_default_agents, plan, synthesize, verify
from jarvis.config import Config
from jarvis.llm import LLM, build_llm
from jarvis.memory import LeaseHeld, MissionStore
from jarvis.tools import ToolRegistry, build_default_registry


class MissionCancelled(Exception):
    pass


class Orchestrator:
    def __init__(self, cfg: Config, store: MissionStore | None = None, llm: LLM | None = None,
                 tools: ToolRegistry | None = None, agents: AgentRegistry | None = None):
        self.cfg = cfg
        self.store = store or MissionStore(cfg.db_path)
        self.llm = llm or build_llm(cfg)
        self.tools = tools or build_default_registry(cfg, self.store)
        self.agents = agents or build_default_agents()
        self.sleep = time.sleep  # remplaçable dans les tests

    # -- API publique ---------------------------------------------------
    def submit(self, objective: str) -> int:
        return self.store.create_mission(objective)

    def run(self, mission_id: int, retry_failed: bool = True, force: bool = False) -> dict:
        store = self.store
        mission = store.get_mission(mission_id)
        if mission is None:
            raise KeyError(f"mission {mission_id} introuvable")
        if mission["status"] == "done":
            return mission
        store.claim(mission_id, self.cfg.lease_seconds, force=force)
        calls_before = self.llm.calls
        try:
            reset = store.reset_interrupted(mission_id, retry_failed=retry_failed)
            if reset:
                store.log(mission_id, "resume", f"Reprise : {reset} étape(s) relancée(s)")
            store.update_mission(mission_id, error=None)

            if not store.get_steps(mission_id):
                self._plan(mission_id, mission["objective"])

            store.update_mission(mission_id, status="running")
            for step in store.get_steps(mission_id):
                if step["status"] in ("done", "skipped"):
                    continue
                self._check_cancel(mission_id)
                if not self._run_step(mission_id, step):
                    store.update_mission(mission_id, status="failed",
                                         error=f"Étape {step['position'] + 1} « {step['title']} » en échec")
                    store.log(mission_id, "mission_failed", "Mission en échec (reprise possible)")
                    return store.get_mission(mission_id)

            self._finish(mission_id, mission["objective"])
        except MissionCancelled:
            store.update_mission(mission_id, status="cancelled")
            store.log(mission_id, "cancelled", "Mission annulée")
        except Exception as exc:
            store.update_mission(mission_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            store.log(mission_id, "error", f"Erreur inattendue : {exc}")
        finally:
            store.add_llm_calls(mission_id, self.llm.calls - calls_before)
            store.release(mission_id)
        return store.get_mission(mission_id)

    # -- étapes internes --------------------------------------------------
    def _check_cancel(self, mission_id: int):
        if self.store.cancel_requested(mission_id):
            raise MissionCancelled()

    def _plan(self, mission_id: int, objective: str):
        self.store.update_mission(mission_id, status="planning")
        self.store.log(mission_id, "planning", "Planification…")
        memory = self._memory_context(mission_id, objective)
        steps = plan(self.llm, self.agents, objective, memory, self.cfg.max_steps)
        self.store.save_plan(mission_id, steps)
        self.store.log(mission_id, "plan", f"Plan en {len(steps)} étape(s)", steps)

    def _memory_context(self, mission_id: int, objective: str) -> str:
        hits = self.store.search(objective, limit=3, exclude_mission=mission_id)
        lines = []
        for h in hits:
            if h["type"] == "mission":
                lines.append(f"- Mission #{h['mission_id']} « {h['objective']} » → {h['result'][:300]}")
            else:
                lines.append(f"- Leçon (mission #{h['mission_id']}) : {h['text']}")
        return "\n".join(lines)

    def _step_context(self, mission_id: int, step: dict) -> str:
        done = [s for s in self.store.get_steps(mission_id)
                if s["position"] < step["position"] and s["status"] == "done"]
        mission = self.store.get_mission(mission_id)
        parts = [f"Objectif global de la mission : {mission['objective']}"]
        for s in done:
            parts.append(f"### Étape {s['position'] + 1} — {s['title']}\n{(s['output'] or '')[:3000]}")
        return "\n\n".join(parts)

    def _run_step(self, mission_id: int, step: dict) -> bool:
        store, cfg = self.store, self.cfg
        sid = step["id"]
        label = f"Étape {step['position'] + 1} « {step['title']} »"
        store.update_step(sid, status="running", started_at=time.time())
        store.log(mission_id, "step_start", label, step_id=sid)
        feedback = ""

        for attempt in range(step["attempts"], cfg.max_step_attempts):
            self._check_cancel(mission_id)
            store.renew(mission_id, cfg.lease_seconds)
            last = attempt == cfg.max_step_attempts - 1
            # Dernière chance : si le spécialiste a échoué, on essaie l'agent de secours.
            agent_name = step["agent"]
            if last and attempt > 0 and agent_name != self.agents.fallback:
                agent_name = self.agents.fallback
                store.log(mission_id, "fallback", f"{label} : passage à l'agent {agent_name}", step_id=sid)
            agent = self.agents.get(agent_name)
            store.update_step(sid, attempts=attempt + 1, agent_used=agent.name)

            task = step["instructions"]
            if step.get("success_criteria"):
                task += f"\n\nCritères de réussite : {step['success_criteria']}"
            if feedback:
                task += f"\n\nTentative précédente refusée. Motif : {feedback}\nCorrige ce point."

            try:
                result = agent.run(
                    self.llm, self.tools, task, self._step_context(mission_id, step),
                    max_tool_calls=cfg.max_tool_calls,
                    on_event=lambda kind, msg, data=None: store.log(mission_id, kind, msg, data, step_id=sid),
                )
                verdict = verify(self.llm, step, result.output)
            except Exception as exc:
                feedback = f"erreur pendant l'exécution : {exc}"
                store.update_step(sid, error=feedback)
                store.log(mission_id, "step_error", f"{label}, tentative {attempt + 1} : {exc}", step_id=sid)
            else:
                store.update_step(sid, output=result.output, verdict=verdict.reason)
                if verdict.ok:
                    store.update_step(sid, status="done", error=None, finished_at=time.time())
                    store.log(mission_id, "step_done", f"{label} validée ({agent.name})",
                              {"verdict": verdict.reason}, step_id=sid)
                    return True
                feedback = verdict.reason or "critères non remplis"
                store.log(mission_id, "step_rejected",
                          f"{label}, tentative {attempt + 1} refusée : {feedback}", step_id=sid)

            if not last:
                self.sleep(cfg.retry_delay * (2 ** attempt))

        store.update_step(sid, status="failed", error=feedback, finished_at=time.time())
        store.log(mission_id, "step_failed", f"{label} en échec après {cfg.max_step_attempts} tentatives",
                  step_id=sid)
        return False

    def _finish(self, mission_id: int, objective: str):
        steps = self.store.get_steps(mission_id)
        try:
            result, lesson = synthesize(self.llm, objective, steps)
        except Exception as exc:
            # La synthèse est un bonus : en cas d'échec on livre le dernier résultat.
            self.store.log(mission_id, "synth_error", f"Synthèse impossible : {exc}")
            result, lesson = steps[-1]["output"] or "", ""
        self.store.update_mission(mission_id, status="done", result=result)
        if lesson:
            self.store.add_lesson(mission_id, lesson)
        self.store.log(mission_id, "mission_done", "Mission terminée", {"lesson": lesson} if lesson else None)


def make_orchestrator(cfg: Config | None = None) -> Orchestrator:
    return Orchestrator(cfg or Config.from_env())


__all__ = ["Orchestrator", "make_orchestrator", "LeaseHeld", "MissionCancelled"]
