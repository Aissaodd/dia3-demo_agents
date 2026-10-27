"""Orchestrateur central.

Cycle d'une mission :
  1. verrou (un seul processus à la fois sur une mission) ;
  2. reprise : les étapes coupées en plein vol ou en attente d'approbation repartent ;
  3. plan (si pas encore fait) : objectif reformulé, étapes, dépendances, nourri
     par la mémoire (missions et résultats vérifiés similaires) ;
  4. ordonnancement : chaque étape dont les dépendances sont terminées part,
     jusqu'à ``max_parallel`` en même temps ; chacune passe agent → contrôles →
     vérificateur → réessai selon la politique de reprise ;
  5. synthèse (coordinateur), mémorisation, mission terminée.
Chaque transition est écrite en base avant de passer à la suite : si le
processus meurt, ``run(mission_id)`` reprend aux étapes non finies.

Garde-fous : budget par mission (appels, jetons, coût), délai par étape et par
mission, annulation et arrêt d'urgence vérifiés avant CHAQUE appel de modèle,
approbation humaine des actions sensibles. Aucun changement silencieux de
fournisseur : chaque appel enregistre le fournisseur réellement utilisé.
"""

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from jarvis import recovery
from jarvis.agents import (PLANNER_ROLE, SYNTH_ROLE, AgentRegistry, build_default_agents,
                           plan_with_understanding, synthesize)
from jarvis.config import Config
from jarvis.memory import LeaseHeld, MissionStore
from jarvis.providers import BudgetExceeded, LLMError, as_router, build_router
from jarvis.recovery import StepTimeout
from jarvis.tools import ApprovalRequired, ToolContext, ToolRegistry, Workspace, build_default_registry
from jarvis.verification import VERIFIER_ROLE, verify


class MissionStop(Exception):
    """Arrêt de la mission demandé (annulation, arrêt d'urgence, délai)."""


class MissionCancelled(MissionStop):
    pass


class EmergencyStop(MissionStop):
    pass


class MissionTimeout(MissionStop):
    pass


class MeteredLLM:
    """Le modèle, vu depuis une mission : vérifie arrêt, délai et budget avant chaque
    appel, et enregistre jetons, coût, latence et fournisseur après."""

    def __init__(self, router, store: MissionStore, cfg: Config, mission_id: int, step_id: int | None = None,
                 deadline: float | None = None, stop=None):
        self.router, self.store, self.cfg = router, store, cfg
        self.mission_id, self.step_id, self.deadline, self.stop = mission_id, step_id, deadline, stop

    def bind(self, step_id: int | None, deadline: float | None = None) -> "MeteredLLM":
        return MeteredLLM(self.router, self.store, self.cfg, self.mission_id, step_id,
                          deadline or self.deadline, self.stop)

    def _check_budget(self):
        m = self.store.get_mission(self.mission_id)
        cfg = self.cfg
        if m["llm_calls"] >= cfg.max_llm_calls:
            raise BudgetExceeded(f"budget épuisé : {m['llm_calls']} appels de modèle (max {cfg.max_llm_calls}, "
                                 "JARVIS_MAX_LLM_CALLS)")
        tokens = (m.get("input_tokens") or 0) + (m.get("output_tokens") or 0)
        if tokens >= cfg.max_tokens:
            raise BudgetExceeded(f"budget épuisé : {tokens} jetons (max {cfg.max_tokens}, JARVIS_MAX_TOKENS)")
        if (m.get("cost_usd") or 0) >= cfg.max_cost_usd:
            raise BudgetExceeded(f"budget épuisé : {m['cost_usd']:.4f} $ (max {cfg.max_cost_usd} $, "
                                 "JARVIS_MAX_COST_USD)")

    def complete(self, messages, role=None) -> str:
        if self.stop:
            self.stop()
        if self.deadline and time.time() > self.deadline:
            raise StepTimeout("délai de l'étape dépassé")
        self._check_budget()
        provider = self.router.for_role(role)
        try:
            text, usage = provider.complete_with_usage(messages, role=role)
        except LLMError as exc:
            self.store.record_llm_call(self.mission_id, self.step_id, role, None, ok=False, error=str(exc),
                                       provider=provider.name, model=provider.model)
            raise
        self.store.record_llm_call(self.mission_id, self.step_id, role, usage)
        self.store.renew(self.mission_id, self.cfg.lease_seconds)
        return text


class Orchestrator:
    def __init__(self, cfg: Config, store: MissionStore | None = None, llm=None,
                 tools: ToolRegistry | None = None, agents: AgentRegistry | None = None):
        self.cfg = cfg
        self.store = store or MissionStore(cfg.db_path)
        self.llm = as_router(llm) if llm is not None else build_router(cfg)
        self.tools = tools or build_default_registry(cfg, self.store)
        self.agents = agents or build_default_agents()
        self.workspace = Workspace(cfg.workspace)
        self.sleep = time.sleep  # remplaçable dans les tests

    # -- API publique ---------------------------------------------------
    def submit(self, objective: str, **kw) -> int:
        return self.store.create_mission(objective, **kw)

    def run(self, mission_id: int, retry_failed: bool = True, force: bool = False) -> dict:
        store = self.store
        mission = store.get_mission(mission_id)
        if mission is None:
            raise KeyError(f"mission {mission_id} introuvable")
        if mission["status"] == "done":
            return mission
        if store.emergency_stop():
            store.log(mission_id, "emergency", "Arrêt d'urgence actif : mission non lancée")
            store.update_mission(mission_id, error="Arrêt d'urgence actif (python -m jarvis stop --off pour lever)")
            return store.get_mission(mission_id)
        store.claim(mission_id, self.cfg.lease_seconds, force=force)
        deadline = time.time() + self.cfg.mission_timeout
        stop = lambda: self._check_stop(mission_id, deadline)  # noqa: E731
        metered = MeteredLLM(self.llm, store, self.cfg, mission_id, stop=stop)
        try:
            reset = store.reset_interrupted(mission_id, retry_failed=retry_failed)
            if reset:
                store.log(mission_id, "resume", f"Reprise : {reset} étape(s) relancée(s)")
            store.update_mission(mission_id, error=None)

            if not store.get_steps(mission_id):
                self._plan(mission_id, mission["objective"], metered)

            store.update_mission(mission_id, status="running")
            outcome = self._execute(mission_id, metered, deadline)
            if outcome == "done":
                self._finish(mission_id, mission["objective"], metered)
            elif outcome == "waiting":
                n = len(store.list_approvals("pending", mission_id))
                store.update_mission(mission_id, status="waiting")
                store.log(mission_id, "waiting", f"Mission en attente : {n} action(s) à approuver")
            else:
                self._fail(mission_id, outcome)
        except MissionCancelled:
            store.update_mission(mission_id, status="cancelled")
            store.log(mission_id, "cancelled", "Mission annulée")
        except EmergencyStop:
            store.update_mission(mission_id, status="cancelled", error="Arrêt d'urgence")
            store.log(mission_id, "emergency", "Mission arrêtée par l'arrêt d'urgence")
        except MissionTimeout:
            store.update_mission(mission_id, status="failed",
                                 error=f"Délai de mission dépassé ({self.cfg.mission_timeout:.0f} s, reprise possible)")
            store.log(mission_id, "timeout", "Délai de mission dépassé")
        except BudgetExceeded as exc:
            store.update_mission(mission_id, status="failed", error=str(exc))
            store.log(mission_id, "budget", str(exc))
        except Exception as exc:
            store.update_mission(mission_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            store.log(mission_id, "error", f"Erreur inattendue : {exc}")
        finally:
            store.release(mission_id)
        return store.get_mission(mission_id)

    # -- arrêt, plan ------------------------------------------------------
    def _check_stop(self, mission_id: int, deadline: float | None = None):
        if self.store.emergency_stop():
            raise EmergencyStop()
        if self.store.cancel_requested(mission_id):
            raise MissionCancelled()
        if deadline and time.time() > deadline:
            raise MissionTimeout()

    # compatibilité avec l'ancienne API
    def _check_cancel(self, mission_id: int):
        self._check_stop(mission_id)

    def _plan(self, mission_id: int, objective: str, llm=None):
        llm = llm or MeteredLLM(self.llm, self.store, self.cfg, mission_id)
        self.store.update_mission(mission_id, status="planning")
        self.store.log(mission_id, "planning", "Planification…")
        memory = self._memory_context(mission_id, objective)
        understanding, steps = plan_with_understanding(llm, self.agents, objective, memory, self.cfg.max_steps)
        self.store.save_plan(mission_id, steps)
        for i, s in enumerate(steps):
            if s.get("planned_agent") and s["planned_agent"] != s["agent"]:
                self.store.log(mission_id, "coordination", f"Étape {i + 1} : agent « {s['planned_agent']} » "
                               f"inconnu, confiée à {s['agent']} (compétences)")
        parallel = sum(1 for s in steps if not s["depends_on"])
        self.store.log(mission_id, "plan", f"Plan en {len(steps)} étape(s)"
                       + (f", {parallel} démarrent en parallèle" if parallel > 1 else ""),
                       {"understanding": understanding, "steps": steps})

    def _memory_context(self, mission_id: int, objective: str) -> str:
        lines = []
        mission = self.store.get_mission(mission_id)
        if mission and mission.get("parent_id"):
            parent = self.store.get_mission(mission["parent_id"])
            if parent:
                lines.append(f"- SUITE de la mission #{parent['id']} « {parent['objective']} » → "
                             f"{(parent['result'] or parent['error'] or '')[:1500]}")
        same = self.store.find_identical_done(objective)
        if same and same["id"] != mission_id:
            lines.append(f"- DÉJÀ RÉALISÉE (mission #{same['id']}, résultat vérifié) → {(same['result'] or '')[:1500]}")
        for h in self.store.search(objective, limit=4, exclude_mission=mission_id):
            if h["type"] == "mission":
                if same and h["mission_id"] == same["id"]:
                    continue
                lines.append(f"- Mission #{h['mission_id']} « {h['objective']} » → {h['result'][:300]}")
            else:
                lines.append(f"- {h['type']} (mission #{h['mission_id']}) : {h['text'][:400]}")
        return "\n".join(lines)

    def _step_context(self, mission_id: int, step: dict, steps: list[dict] | None = None) -> str:
        steps = steps or self.store.get_steps(mission_id)
        by_pos = {s["position"]: s for s in steps}
        # Toutes les étapes dont celle-ci dépend, directement ou non.
        needed, todo = set(), list(step["deps"])
        while todo:
            p = todo.pop()
            if p in by_pos and p not in needed:
                needed.add(p)
                todo.extend(by_pos[p]["deps"])
        mission = self.store.get_mission(mission_id)
        parts = [f"Objectif global de la mission : {mission['objective']}"]
        if mission.get("parent_id"):
            parent = self.store.get_mission(mission["parent_id"])
            if parent and parent.get("result"):
                parts.append(f"### Résultat de la mission précédente (#{parent['id']})\n{parent['result'][:3000]}")
        for p in sorted(needed):
            s = by_pos[p]
            if s["status"] == "done":
                parts.append(f"### Étape {p + 1} — {s['title']}\n{(s['output'] or '')[:3000]}")
        return "\n\n".join(parts)

    # -- ordonnancement ---------------------------------------------------
    def _execute(self, mission_id: int, llm: MeteredLLM, deadline: float) -> str:
        store, cfg = self.store, self.cfg
        running: dict = {}  # future -> step id
        stop_error: BaseException | None = None
        with ThreadPoolExecutor(max_workers=cfg.max_parallel, thread_name_prefix="jarvis-step") as pool:
            while True:
                if stop_error is None:
                    try:
                        self._check_stop(mission_id, deadline)
                    except MissionStop as exc:
                        stop_error = exc
                store.renew(mission_id, cfg.lease_seconds)
                steps = store.get_steps(mission_id)
                done = {s["position"] for s in steps if s["status"] in ("done", "skipped")}
                failed = [s for s in steps if s["status"] == "failed"]
                waiting = [s for s in steps if s["status"] == "waiting"]
                busy = set(running.values())
                if stop_error is None and not failed:
                    ready = [s for s in steps if s["status"] == "pending" and s["id"] not in busy
                             and all(d in done for d in s["deps"])]
                    for s in ready[: cfg.max_parallel - len(running)]:
                        store.update_step(s["id"], status="running", started_at=time.time())
                        running[pool.submit(self._run_step, mission_id, s, steps, llm, deadline)] = s["id"]
                if not running:
                    if stop_error is not None:
                        raise stop_error
                    if len(done) == len(steps):
                        return "done"
                    if failed:
                        return "failed"
                    if waiting:
                        return "waiting"
                    return "blocked"
                finished, _ = wait(list(running), timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    running.pop(fut)
                    exc = fut.exception()
                    if isinstance(exc, MissionStop):
                        stop_error = stop_error or exc
                    elif exc is not None:
                        stop_error = stop_error or exc

    def _run_step(self, mission_id: int, step: dict, steps: list[dict], llm: MeteredLLM, deadline: float) -> str:
        store, cfg = self.store, self.cfg
        sid = step["id"]
        label = f"Étape {step['position'] + 1} « {step['title']} »"
        store.log(mission_id, "step_start", f"{label} → {step['agent']}", step_id=sid)
        feedback = ""
        agent_name = step["agent"]
        tried: list[str] = []
        context = self._step_context(mission_id, step, steps)

        for attempt in range(step["attempts"], cfg.max_step_attempts):
            self._check_stop(mission_id, deadline)
            last = attempt == cfg.max_step_attempts - 1
            # Dernière chance : un autre agent, sans élévation de privilèges.
            if last and attempt > 0:
                alt = recovery.safe_alternative(step, agent_name, tried, self.agents, self.tools)
                if alt:
                    store.log(mission_id, "fallback", f"{label} : passage à l'agent {alt}", step_id=sid)
                    agent_name = alt
            agent = self.agents.get(agent_name)
            tried.append(agent.name)
            store.update_step(sid, attempts=attempt + 1, agent_used=agent.name)

            task = step["instructions"]
            task += f"\n\nCritères d'acceptation : {step.get('success_criteria') or 'réponse complète et exacte'}"
            if feedback:
                task += f"\n\nTentative précédente refusée. Motif : {feedback}\nCorrige ce point."
            step_deadline = min(deadline, time.time() + cfg.step_timeout)
            mllm = llm.bind(sid, step_deadline)
            ctx = ToolContext(mission_id, sid, store, cfg.approval_mode, cfg.max_web_calls, agent.name)

            try:
                result = agent.run(
                    mllm, self.tools, task, context, max_tool_calls=cfg.max_tool_calls,
                    on_event=lambda kind, msg, data=None: store.log(mission_id, kind, msg, data, step_id=sid),
                    ctx=ctx, should_stop=lambda: self._check_stop(mission_id, deadline),
                )
                verdict = verify(mllm, step, result.output, self.workspace,
                                 self.agents.system("verificateur", VERIFIER_ROLE))
            except ApprovalRequired as req:
                store.update_step(sid, status="waiting", error=str(req), error_kind="approval")
                store.log(mission_id, "approval_needed", f"{label} : approbation requise — {req.reason}",
                          {"approval_id": req.approval_id, "tool": req.tool}, step_id=sid)
                return "waiting"
            except MissionStop:
                store.update_step(sid, status="pending")
                raise
            except Exception as exc:
                kind = recovery.classify(exc)
                decision = recovery.decide(kind, attempt, cfg.max_step_attempts)
                feedback = f"erreur pendant l'exécution : {exc}"
                store.update_step(sid, error=feedback, error_kind=kind)
                store.log(mission_id, "step_error", f"{label}, tentative {attempt + 1} : "
                          f"{recovery.KIND_LABELS.get(kind, kind)} — {exc}", {"decision": decision.reason},
                          step_id=sid)
                if not decision.retry:
                    return self._step_failed(mission_id, sid, label, feedback, kind, decision)
            else:
                store.update_step(sid, output=result.output, verdict=f"[{verdict.source}] {verdict.reason}")
                if verdict.ok:
                    store.update_step(sid, status="done", error=None, error_kind=None, finished_at=time.time())
                    store.log(mission_id, "step_done", f"{label} validée ({agent.name})",
                              {"verdict": verdict.reason, "checks": verdict.checks}, step_id=sid)
                    if feedback:
                        store.remember("error_solution", f"Problème : {feedback[:500]}\nSolution : l'agent "
                                       f"{agent.name} a réussi l'étape « {step['title']} » en corrigeant ce point.",
                                       source=f"mission #{mission_id}", mission_id=mission_id)
                    return "done"
                feedback = verdict.reason or "critères non remplis"
                store.update_step(sid, error_kind="rejected")
                store.log(mission_id, "step_rejected",
                          f"{label}, tentative {attempt + 1} refusée : {feedback}", step_id=sid)

            if not last:
                self.sleep(cfg.retry_delay * (2 ** attempt))

        return self._step_failed(mission_id, sid, label, feedback, "rejected", None)

    def _step_failed(self, mission_id, sid, label, feedback, kind, decision) -> str:
        self.store.update_step(sid, status="failed", error=feedback, error_kind=kind, finished_at=time.time())
        why = decision.reason if decision else f"après {self.cfg.max_step_attempts} tentatives"
        self.store.log(mission_id, "step_failed", f"{label} en échec ({why})", step_id=sid)
        return "failed"

    # -- fin de mission ---------------------------------------------------
    def _fail(self, mission_id: int, outcome: str):
        steps = self.store.get_steps(mission_id)
        failed = [s for s in steps if s["status"] == "failed"]
        if failed:
            s = failed[0]
            kinds = {f["error_kind"] for f in failed}
            hint = " — intervention requise (clé, budget ou configuration)" if kinds & {"budget", "permanent"} \
                else " (reprise possible)"
            error = f"Étape {s['position'] + 1} « {s['title']} » en échec : {(s['error'] or '')[:300]}{hint}"
        else:
            error = "Plan bloqué : des étapes attendent des dépendances qui ne peuvent pas se terminer"
        self.store.update_mission(mission_id, status="failed", error=error)
        self.store.log(mission_id, "mission_failed", error)
        mission = self.store.get_mission(mission_id)
        self.store.remember("error", f"Échec de « {mission['objective'][:300]} » : {error}",
                            source=f"mission #{mission_id}", mission_id=mission_id, confidence=0.8)

    def _finish(self, mission_id: int, objective: str, llm=None):
        llm = llm or MeteredLLM(self.llm, self.store, self.cfg, mission_id)
        steps = self.store.get_steps(mission_id)
        try:
            result, lesson = synthesize(llm, objective, steps, self.agents.system("coordinateur", SYNTH_ROLE))
        except MissionStop:
            raise
        except Exception as exc:
            # La synthèse est un bonus : en cas d'échec on livre les résultats vérifiés tels quels.
            self.store.log(mission_id, "synth_error", f"Synthèse impossible : {exc}")
            result = "\n\n".join(f"### {s['title']}\n{s['output'] or ''}" for s in steps) if len(steps) > 1 \
                else (steps[-1]["output"] or "")
            lesson = ""
        self.store.update_mission(mission_id, status="done", result=result)
        self.store.remember("verified_result", f"Objectif : {objective[:500]}\nRésultat vérifié : {result[:1500]}",
                            source=f"mission #{mission_id}", mission_id=mission_id)
        if lesson:
            self.store.add_lesson(mission_id, lesson)
        self.store.log(mission_id, "mission_done", "Mission terminée", {"lesson": lesson} if lesson else None)


def make_orchestrator(cfg: Config | None = None) -> Orchestrator:
    return Orchestrator(cfg or Config.from_env())


__all__ = ["Orchestrator", "MeteredLLM", "make_orchestrator", "LeaseHeld", "MissionCancelled", "EmergencyStop",
           "MissionTimeout", "MissionStop", "PLANNER_ROLE"]
