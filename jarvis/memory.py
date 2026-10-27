"""Mémoire des missions (SQLite).

Tout ce que fait Jarvis est écrit ici : missions, étapes, événements et
leçons. L'orchestrateur sauvegarde après chaque étape, donc une mission
interrompue reprend là où elle s'était arrêtée. Une connexion par
opération : le tableau de bord et le moteur peuvent lire en même temps.
"""

import json
import os
import re
import socket
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS missions (
    id INTEGER PRIMARY KEY,
    objective TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    worker TEXT,
    lease_until REAL NOT NULL DEFAULT 0,
    llm_calls INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY,
    mission_id INTEGER NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    instructions TEXT NOT NULL,
    agent TEXT NOT NULL,
    success_criteria TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    agent_used TEXT,
    output TEXT,
    verdict TEXT,
    error TEXT,
    started_at REAL,
    finished_at REAL,
    UNIQUE (mission_id, position)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    mission_id INTEGER NOT NULL,
    step_id INTEGER,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY,
    mission_id INTEGER,
    ts REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_mission ON steps(mission_id, position);
CREATE INDEX IF NOT EXISTS idx_events_mission ON events(mission_id, id);
"""

MISSION_STATUSES = ("pending", "planning", "running", "done", "failed", "cancelled")
STEP_STATUSES = ("pending", "running", "done", "failed", "skipped")


class LeaseHeld(Exception):
    """Un autre processus exécute déjà cette mission."""


class MissionStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- missions -----------------------------------------------------
    def create_mission(self, objective: str) -> int:
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO missions (objective, created_at, updated_at) VALUES (?, ?, ?)",
                (objective.strip(), now, now),
            )
            mid = cur.lastrowid
        self.log(mid, "mission", "Mission créée", {"objective": objective})
        return mid

    def get_mission(self, mission_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM missions WHERE id = ?", (mission_id,)).fetchone()
        return dict(row) if row else None

    def list_missions(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT m.*,
                     (SELECT COUNT(*) FROM steps s WHERE s.mission_id = m.id) AS n_steps,
                     (SELECT COUNT(*) FROM steps s WHERE s.mission_id = m.id AND s.status = 'done') AS n_done
                   FROM missions m ORDER BY m.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_mission(self, mission_id: int, **fields):
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE missions SET {cols} WHERE id = ?", (*fields.values(), mission_id))

    def add_llm_calls(self, mission_id: int, n: int):
        if n:
            with self._conn() as c:
                c.execute("UPDATE missions SET llm_calls = llm_calls + ? WHERE id = ?", (n, mission_id))

    def request_cancel(self, mission_id: int):
        self.update_mission(mission_id, cancel_requested=1)
        self.log(mission_id, "cancel", "Annulation demandée")

    def cancel_requested(self, mission_id: int) -> bool:
        m = self.get_mission(mission_id)
        return bool(m and m["cancel_requested"])

    # --- verrou d'exécution (un seul worker par mission) --------------
    @staticmethod
    def worker_id() -> str:
        return f"{socket.gethostname()}:{os.getpid()}"

    def claim(self, mission_id: int, seconds: int, force: bool = False) -> str:
        me = self.worker_id()
        now = time.time()
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT worker, lease_until FROM missions WHERE id = ?", (mission_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"mission {mission_id} introuvable")
            if not force and row["lease_until"] > now and row["worker"] != me:
                raise LeaseHeld(f"mission {mission_id} déjà en cours ({row['worker']})")
            c.execute(
                "UPDATE missions SET worker = ?, lease_until = ?, cancel_requested = 0 WHERE id = ?",
                (me, now + seconds, mission_id),
            )
        return me

    def renew(self, mission_id: int, seconds: int):
        with self._conn() as c:
            c.execute(
                "UPDATE missions SET lease_until = ? WHERE id = ? AND worker = ?",
                (time.time() + seconds, mission_id, self.worker_id()),
            )

    def release(self, mission_id: int):
        with self._conn() as c:
            c.execute(
                "UPDATE missions SET lease_until = 0 WHERE id = ? AND worker = ?",
                (mission_id, self.worker_id()),
            )

    def is_live(self, mission: dict) -> bool:
        return mission["lease_until"] > time.time()

    # --- étapes -------------------------------------------------------
    def save_plan(self, mission_id: int, steps: list[dict]):
        with self._conn() as c:
            c.execute("DELETE FROM steps WHERE mission_id = ?", (mission_id,))
            for i, s in enumerate(steps):
                c.execute(
                    """INSERT INTO steps (mission_id, position, title, instructions, agent, success_criteria)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (mission_id, i, s["title"], s["instructions"], s["agent"], s.get("success_criteria", "")),
                )

    def get_steps(self, mission_id: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM steps WHERE mission_id = ? ORDER BY position", (mission_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_step(self, step_id: int, **fields):
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE steps SET {cols} WHERE id = ?", (*fields.values(), step_id))

    def reset_interrupted(self, mission_id: int, retry_failed: bool) -> int:
        """Remet en attente les étapes coupées en plein vol (et, sur demande, les échouées)."""
        statuses = ("running", "failed") if retry_failed else ("running",)
        marks = ",".join("?" * len(statuses))
        with self._conn() as c:
            cur = c.execute(
                f"""UPDATE steps SET status = 'pending', attempts = 0, error = NULL
                    WHERE mission_id = ? AND status IN ({marks})""",
                (mission_id, *statuses),
            )
            return cur.rowcount

    # --- journal ------------------------------------------------------
    def log(self, mission_id: int, kind: str, message: str, data=None, step_id: int | None = None):
        with self._conn() as c:
            c.execute(
                "INSERT INTO events (mission_id, step_id, ts, kind, message, data) VALUES (?, ?, ?, ?, ?, ?)",
                (mission_id, step_id, time.time(), kind, message[:2000],
                 json.dumps(data, ensure_ascii=False, default=str)[:20000] if data is not None else None),
            )

    def get_events(self, mission_id: int, limit: int = 500) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM events WHERE mission_id = ? ORDER BY id DESC LIMIT ?",
                (mission_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- leçons et recherche dans le passé ----------------------------
    def add_lesson(self, mission_id: int | None, text: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO lessons (mission_id, ts, text) VALUES (?, ?, ?)",
                (mission_id, time.time(), text.strip()[:2000]),
            )

    def search(self, query: str, limit: int = 5, exclude_mission: int | None = None) -> list[dict]:
        """Missions terminées et leçons qui partagent le plus de mots avec la requête."""
        words = {w for w in re.findall(r"\w{4,}", query.lower())}
        if not words:
            return []
        with self._conn() as c:
            missions = c.execute(
                "SELECT id, objective, result FROM missions WHERE status = 'done' AND id != ?",
                (exclude_mission or -1,),
            ).fetchall()
            lessons = c.execute("SELECT mission_id, text FROM lessons").fetchall()
        scored = []
        for m in missions:
            text = f"{m['objective']} {m['result'] or ''}".lower()
            score = sum(1 for w in words if w in text)
            if score:
                scored.append((score, {"type": "mission", "mission_id": m["id"],
                                       "objective": m["objective"], "result": (m["result"] or "")[:600]}))
        for l in lessons:
            if l["mission_id"] == exclude_mission:
                continue
            score = sum(1 for w in words if w in l["text"].lower())
            if score:
                scored.append((score, {"type": "leçon", "mission_id": l["mission_id"], "text": l["text"]}))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:limit]]

    def stats(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) AS n FROM missions GROUP BY status").fetchall()
        out = {s: 0 for s in MISSION_STATUSES}
        out.update({r["status"]: r["n"] for r in rows})
        return out
