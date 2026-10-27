"""Mémoire de Jarvis (SQLite).

Tout ce que fait Jarvis est écrit ici : missions, étapes, événements, appels
de modèles (jetons, coût, latence), approbations, conversations du chat et
mémoire à long terme. L'orchestrateur sauvegarde après chaque transition,
donc une mission interrompue reprend là où elle s'était arrêtée. Une
connexion par opération : tableau de bord, chat et workers lisent en même temps.

Mémoire à long terme (table ``memories``), par type :
  mission          compte rendu d'une mission terminée
  verified_result  résultat validé par le vérificateur
  preference       préférence de l'utilisateur
  knowledge        connaissance récupérée d'une source externe (avec sa source)
  hypothesis       information incertaine, à confirmer
  error_solution   erreur rencontrée et ce qui l'a résolue
  error            échec non résolu
  lesson           leçon tirée d'une mission
Les doublons (même type, même texte normalisé) ne sont pas recréés ; les
secrets et coordonnées personnelles sont masqués avant d'être mémorisés.
"""

import hashlib
import json
import os
import re
import socket
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

from jarvis.providers.base import redact

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
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT,
    mission_id INTEGER,
    confidence REAL NOT NULL DEFAULT 1.0,
    fingerprint TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_used REAL,
    UNIQUE (kind, fingerprint)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    mission_id INTEGER,
    step_id INTEGER,
    ts REAL NOT NULL,
    role TEXT,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    ok INTEGER NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY,
    mission_id INTEGER NOT NULL,
    step_id INTEGER,
    tool TEXT NOT NULL,
    args TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    decided_at REAL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    mission_id INTEGER,
    client_key TEXT UNIQUE,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_mission ON steps(mission_id, position);
CREATE INDEX IF NOT EXISTS idx_events_mission ON events(mission_id, id);
CREATE INDEX IF NOT EXISTS idx_calls_mission ON llm_calls(mission_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, mission_id);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""

# Colonnes ajoutées après la première version : migrées sans perte de données.
MIGRATIONS = {
    "missions": {
        "input_tokens": "INTEGER NOT NULL DEFAULT 0",
        "output_tokens": "INTEGER NOT NULL DEFAULT 0",
        "cost_usd": "REAL NOT NULL DEFAULT 0",
        "cost_unknown_calls": "INTEGER NOT NULL DEFAULT 0",
        "parent_id": "INTEGER",
        "conversation_id": "INTEGER",
        "source_key": "TEXT",
    },
    "steps": {
        "depends_on": "TEXT",        # JSON : positions des étapes dont celle-ci dépend
        "error_kind": "TEXT",        # transient | permanent | budget | approval | timeout
    },
}

MISSION_STATUSES = ("pending", "planning", "running", "waiting", "done", "failed", "cancelled")
STEP_STATUSES = ("pending", "running", "waiting", "done", "failed", "skipped")
MEMORY_KINDS = ("mission", "verified_result", "preference", "knowledge", "hypothesis",
                "error_solution", "error", "lesson")
# Poids de chaque type dans la recherche : le validé passe avant l'incertain.
KIND_WEIGHT = {"verified_result": 1.5, "error_solution": 1.3, "lesson": 1.2, "preference": 1.2,
               "mission": 1.0, "knowledge": 1.0, "error": 0.8, "hypothesis": 0.6}

_MISSION_COLS = {"objective", "status", "result", "error", "cancel_requested", "worker", "lease_until",
                 "llm_calls", "updated_at", "parent_id", "conversation_id"}
_STEP_COLS = {"status", "attempts", "agent_used", "output", "verdict", "error", "started_at", "finished_at",
              "error_kind", "agent"}


class LeaseHeld(Exception):
    """Un autre processus exécute déjà cette mission."""


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"\w{3,}", normalize(text)) if not w.isdigit()} - _STOPWORDS


_STOPWORDS = {"les", "des", "une", "dans", "pour", "avec", "sur", "est", "que", "qui", "pas", "plus",
              "par", "the", "and", "for", "with", "this", "that", "aux", "ces", "son", "ses", "mes",
              "tes", "nos", "vos", "leur", "leurs", "elle", "ils", "elles", "nous", "vous", "fait",
              "faire", "etre", "avoir", "puis", "donc", "mais"}

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d .\-]{8,}\d(?!\w)")


def sanitize_memory(text: str) -> str:
    """Masque secrets, e-mails et numéros de téléphone avant mémorisation."""
    text = redact(text)
    text = _EMAIL.sub("[e-mail masqué]", text)
    return _PHONE.sub("[numéro masqué]", text)


def args_hash(tool: str, args: dict) -> str:
    raw = tool + "\x00" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class MissionStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():
                have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
                for col, decl in cols.items():
                    if col not in have:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_missions_source ON missions(source_key)")
            # Les leçons de l'ancienne table rejoignent la mémoire typée (une seule fois).
            if not c.execute("SELECT value FROM settings WHERE key = 'lessons_migrated'").fetchone():
                for r in c.execute("SELECT mission_id, ts, text FROM lessons").fetchall():
                    self._insert_memory(c, "lesson", r["text"], None, r["mission_id"], 1.0, r["ts"])
                c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('lessons_migrated', '1')")

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
    def create_mission(self, objective: str, source_key: str | None = None, parent_id: int | None = None,
                       conversation_id: int | None = None) -> int:
        """Crée une mission. Avec ``source_key``, un second appel identique renvoie la même
        mission au lieu d'en créer une autre (protection contre les doubles exécutions)."""
        now = time.time()
        with self._conn() as c:
            if source_key:
                row = c.execute("SELECT id FROM missions WHERE source_key = ?", (source_key,)).fetchone()
                if row:
                    return row["id"]
            cur = c.execute(
                """INSERT INTO missions (objective, created_at, updated_at, source_key, parent_id, conversation_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (objective.strip(), now, now, source_key, parent_id, conversation_id),
            )
            mid = cur.lastrowid
        self.log(mid, "mission", "Mission créée", {"objective": objective, "parent": parent_id})
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
        bad = set(fields) - _MISSION_COLS
        if bad:
            raise ValueError(f"colonnes inconnues : {bad}")
        if "error" in fields and fields["error"]:
            fields["error"] = redact(fields["error"])[:2000]
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE missions SET {cols} WHERE id = ?", (*fields.values(), mission_id))

    def add_llm_calls(self, mission_id: int, n: int):
        if n:
            with self._conn() as c:
                c.execute("UPDATE missions SET llm_calls = llm_calls + ? WHERE id = ?", (n, mission_id))

    def request_cancel(self, mission_id: int):
        m = self.get_mission(mission_id)
        if m is None:
            raise KeyError(f"mission {mission_id} introuvable")
        self.update_mission(mission_id, cancel_requested=1)
        self.log(mission_id, "cancel", "Annulation demandée")
        # Aucun worker vivant : personne ne lira la demande, on annule tout de suite.
        if m["status"] not in ("done", "failed", "cancelled") and not self.is_live(m):
            self.update_mission(mission_id, status="cancelled")
            self.log(mission_id, "cancelled", "Mission annulée (aucun worker actif)")
            self._deny_pending_approvals(mission_id, "mission annulée")

    def cancel_requested(self, mission_id: int) -> bool:
        m = self.get_mission(mission_id)
        return bool(m and m["cancel_requested"])

    # --- arrêt d'urgence (toutes missions) ----------------------------
    def set_emergency_stop(self, on: bool):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('emergency_stop', ?)",
                      ("1" if on else "0",))

    def emergency_stop(self) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key = 'emergency_stop'").fetchone()
        return bool(row and row["value"] == "1")

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
                deps = s.get("depends_on")
                c.execute(
                    """INSERT INTO steps (mission_id, position, title, instructions, agent, success_criteria,
                                          depends_on)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (mission_id, i, s["title"], s["instructions"], s["agent"], s.get("success_criteria", ""),
                     json.dumps(deps) if deps is not None else None),
                )

    def get_steps(self, mission_id: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM steps WHERE mission_id = ? ORDER BY position", (mission_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                deps = json.loads(d["depends_on"]) if d.get("depends_on") else None
            except ValueError:
                deps = None
            # Ancien plan sans dépendances déclarées : chaque étape attend la précédente.
            d["deps"] = deps if isinstance(deps, list) else ([d["position"] - 1] if d["position"] > 0 else [])
            out.append(d)
        return out

    def update_step(self, step_id: int, **fields):
        bad = set(fields) - _STEP_COLS
        if bad:
            raise ValueError(f"colonnes inconnues : {bad}")
        if fields.get("error"):
            fields["error"] = redact(fields["error"])[:2000]
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE steps SET {cols} WHERE id = ?", (*fields.values(), step_id))

    def reset_interrupted(self, mission_id: int, retry_failed: bool) -> int:
        """Remet en attente les étapes coupées en plein vol ou en attente d'approbation
        (et, sur demande, les échouées)."""
        statuses = ("running", "waiting", "failed") if retry_failed else ("running", "waiting")
        marks = ",".join("?" * len(statuses))
        with self._conn() as c:
            cur = c.execute(
                f"""UPDATE steps SET status = 'pending', attempts = 0, error = NULL, error_kind = NULL
                    WHERE mission_id = ? AND status IN ({marks})""",
                (mission_id, *statuses),
            )
            return cur.rowcount

    # --- journal ------------------------------------------------------
    def log(self, mission_id: int, kind: str, message: str, data=None, step_id: int | None = None):
        payload = None
        if data is not None:
            payload = redact(json.dumps(data, ensure_ascii=False, default=str))[:20000]
        with self._conn() as c:
            c.execute(
                "INSERT INTO events (mission_id, step_id, ts, kind, message, data) VALUES (?, ?, ?, ?, ?, ?)",
                (mission_id, step_id, time.time(), kind, redact(message)[:2000], payload),
            )

    def get_events(self, mission_id: int, limit: int = 500, after_id: int = 0) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM events WHERE mission_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
                (mission_id, after_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_tool_uses(self, mission_id: int, tool: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM events WHERE mission_id = ? AND kind = 'tool' "
                "AND json_extract(data, '$.tool') = ?", (mission_id, tool)).fetchone()
        return row["n"]

    # --- appels de modèles : jetons, coût, latence --------------------
    def record_llm_call(self, mission_id: int | None, step_id: int | None, role: str | None, usage=None,
                        ok: bool = True, error: str | None = None, provider: str = "", model: str = ""):
        tin = getattr(usage, "input_tokens", None)
        tout = getattr(usage, "output_tokens", None)
        cost = getattr(usage, "cost_usd", None)
        with self._conn() as c:
            c.execute(
                """INSERT INTO llm_calls (mission_id, step_id, ts, role, provider, model, input_tokens,
                                          output_tokens, cost_usd, latency_ms, ok, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mission_id, step_id, time.time(), role, getattr(usage, "provider", provider),
                 getattr(usage, "model", model), tin, tout, cost, getattr(usage, "latency_ms", None),
                 int(ok), redact(error)[:500] if error else None),
            )
            if mission_id is not None:
                c.execute(
                    """UPDATE missions SET llm_calls = llm_calls + 1, input_tokens = input_tokens + ?,
                         output_tokens = output_tokens + ?, cost_usd = cost_usd + ?,
                         cost_unknown_calls = cost_unknown_calls + ? WHERE id = ?""",
                    (tin or 0, tout or 0, cost or 0, int(ok and cost is None), mission_id),
                )

    def usage_summary(self, mission_id: int | None = None, since: float | None = None) -> list[dict]:
        where, params = [], []
        if mission_id is not None:
            where.append("mission_id = ?")
            params.append(mission_id)
        if since is not None:
            where.append("ts >= ?")
            params.append(since)
        sql = """SELECT provider, model, role, COUNT(*) AS calls, SUM(ok = 0) AS errors,
                        SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                        SUM(cost_usd) AS cost_usd, SUM(cost_usd IS NULL AND ok = 1) AS cost_unknown,
                        AVG(latency_ms) AS avg_latency_ms
                 FROM llm_calls""" + (" WHERE " + " AND ".join(where) if where else "") + \
              " GROUP BY provider, model, role ORDER BY calls DESC"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    # --- approbations humaines ----------------------------------------
    def request_approval(self, mission_id: int, step_id: int | None, tool: str, args: dict, reason: str) -> int:
        h = args_hash(tool, args)
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM approvals WHERE mission_id = ? AND args_hash = ? AND status = 'pending'",
                (mission_id, h),
            ).fetchone()
            if row:
                return row["id"]
            cur = c.execute(
                """INSERT INTO approvals (mission_id, step_id, tool, args, args_hash, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (mission_id, step_id, tool, redact(json.dumps(args, ensure_ascii=False, default=str))[:20000],
                 h, reason, time.time()),
            )
            return cur.lastrowid

    def find_approval(self, mission_id: int, tool: str, args: dict) -> dict | None:
        """La décision la plus récente pour exactement cet appel (outil + arguments)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM approvals WHERE mission_id = ? AND args_hash = ? ORDER BY id DESC LIMIT 1",
                (mission_id, args_hash(tool, args)),
            ).fetchone()
        return dict(row) if row else None

    def decide_approval(self, approval_id: int, approve: bool) -> dict:
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            if row is None:
                raise KeyError(f"approbation {approval_id} introuvable")
            if row["status"] != "pending":
                raise ValueError(f"approbation {approval_id} déjà traitée ({row['status']})")
            status = "approved" if approve else "denied"
            c.execute("UPDATE approvals SET status = ?, decided_at = ? WHERE id = ?",
                      (status, time.time(), approval_id))
        self.log(row["mission_id"], "approval", f"Action {row['tool']} {'approuvée' if approve else 'refusée'}"
                 f" par l'utilisateur (#{approval_id})", step_id=row["step_id"])
        return {**dict(row), "status": status}

    def consume_approval(self, approval_id: int) -> bool:
        """Marque une approbation comme utilisée. Atomique : une action approuvée ne
        s'exécute qu'une fois, même si l'étape est rejouée."""
        with self._conn() as c:
            cur = c.execute("UPDATE approvals SET status = 'used' WHERE id = ? AND status = 'approved'",
                            (approval_id,))
            return cur.rowcount == 1

    def list_approvals(self, status: str | None = "pending", mission_id: int | None = None) -> list[dict]:
        sql, params = "SELECT * FROM approvals WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if mission_id is not None:
            sql += " AND mission_id = ?"
            params.append(mission_id)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql + " ORDER BY id DESC", params).fetchall()]

    def _deny_pending_approvals(self, mission_id: int, why: str):
        with self._conn() as c:
            c.execute("UPDATE approvals SET status = 'denied', decided_at = ? WHERE mission_id = ? "
                      "AND status = 'pending'", (time.time(), mission_id))

    # --- mémoire à long terme -----------------------------------------
    @staticmethod
    def _insert_memory(c, kind, text, source, mission_id, confidence, ts=None) -> int | None:
        fp = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
        row = c.execute("SELECT id FROM memories WHERE kind = ? AND fingerprint = ?", (kind, fp)).fetchone()
        if row:
            c.execute("UPDATE memories SET uses = uses + 1, last_used = ? WHERE id = ?", (time.time(), row["id"]))
            return None
        cur = c.execute(
            """INSERT INTO memories (kind, text, source, mission_id, confidence, fingerprint, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (kind, text, source, mission_id, confidence, fp, ts or time.time()),
        )
        return cur.lastrowid

    def remember(self, kind: str, text: str, source: str | None = None, mission_id: int | None = None,
                 confidence: float = 1.0) -> int | None:
        """Mémorise une information. Renvoie son id, ou None si c'était un doublon ou vide."""
        if kind not in MEMORY_KINDS:
            raise ValueError(f"type de mémoire inconnu : {kind} ({', '.join(MEMORY_KINDS)})")
        text = sanitize_memory((text or "").strip())[:4000]
        if len(text) < 3:
            return None
        with self._conn() as c:
            return self._insert_memory(c, kind, text, sanitize_memory(source) if source else None, mission_id,
                                       max(0.0, min(1.0, confidence)))

    def list_memories(self, kind: str | None = None, limit: int = 200) -> list[dict]:
        sql, params = "SELECT * FROM memories", []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql + " ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()]

    def forget(self, memory_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def add_lesson(self, mission_id: int | None, text: str):
        self.remember("lesson", text, mission_id=mission_id)

    def search(self, query: str, limit: int = 5, exclude_mission: int | None = None,
               kinds: tuple[str, ...] | None = None) -> list[dict]:
        """Missions terminées et souvenirs les plus proches de la requête (mots communs,
        pondérés par le type : un résultat vérifié pèse plus qu'une hypothèse)."""
        words = keywords(query)
        if not words:
            return []
        scored = []
        with self._conn() as c:
            if kinds is None or "mission" in kinds:
                for m in c.execute(
                    "SELECT id, objective, result, updated_at FROM missions WHERE status = 'done' AND id != ?",
                    (exclude_mission or -1,),
                ).fetchall():
                    obj_words = keywords(m["objective"])
                    text_words = obj_words | keywords(m["result"] or "")
                    hits = len(words & text_words) + 0.5 * len(words & obj_words)
                    if hits:
                        scored.append((hits, m["updated_at"], {
                            "type": "mission", "mission_id": m["id"], "objective": m["objective"],
                            "result": (m["result"] or "")[:600]}))
            rows = c.execute("SELECT * FROM memories").fetchall()
        for r in rows:
            if kinds is not None and r["kind"] not in kinds:
                continue
            if exclude_mission is not None and r["mission_id"] == exclude_mission:
                continue
            hits = len(words & keywords(r["text"]))
            if hits:
                label = "leçon" if r["kind"] == "lesson" else r["kind"]
                scored.append((hits * KIND_WEIGHT.get(r["kind"], 1.0) * (0.5 + r["confidence"] / 2),
                               r["created_at"], {"type": label, "kind": r["kind"], "id": r["id"],
                                                 "mission_id": r["mission_id"], "text": r["text"],
                                                 "source": r["source"], "confidence": r["confidence"]}))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [s[2] for s in scored[:limit]]

    def find_identical_done(self, objective: str) -> dict | None:
        """Une mission déjà réussie avec exactement le même objectif (pour la réutiliser)."""
        target = normalize(objective)
        with self._conn() as c:
            rows = c.execute("SELECT * FROM missions WHERE status = 'done' ORDER BY id DESC LIMIT 500").fetchall()
        for r in rows:
            if normalize(r["objective"]) == target:
                return dict(r)
        return None

    def stats(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) AS n FROM missions GROUP BY status").fetchall()
        out = {s: 0 for s in MISSION_STATUSES}
        out.update({r["status"]: r["n"] for r in rows})
        return out

    # --- conversations du chat ----------------------------------------
    def create_conversation(self, title: str = "Nouvelle conversation") -> int:
        with self._conn() as c:
            return c.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)",
                             (title[:120], time.time())).lastrowid

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def rename_conversation(self, conv_id: int, title: str):
        with self._conn() as c:
            c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title[:120], conv_id))

    def add_message(self, conv_id: int, role: str, content: str, mission_id: int | None = None,
                    client_key: str | None = None) -> tuple[int, bool]:
        """Ajoute un message. Renvoie (id, créé). Avec ``client_key``, un message déjà
        reçu (rafraîchissement de la page, double clic) n'est pas ajouté une seconde fois."""
        with self._conn() as c:
            if client_key:
                row = c.execute("SELECT id FROM messages WHERE client_key = ?", (client_key,)).fetchone()
                if row:
                    return row["id"], False
            cur = c.execute(
                """INSERT INTO messages (conversation_id, role, content, mission_id, client_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conv_id, role, content, mission_id, client_key, time.time()),
            )
            return cur.lastrowid, True

    def set_message_mission(self, message_id: int, mission_id: int):
        with self._conn() as c:
            c.execute("UPDATE messages SET mission_id = ? WHERE id = ?", (mission_id, message_id))

    def get_messages(self, conv_id: int, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?) "
                "ORDER BY id", (conv_id, limit)).fetchall()
        return [dict(r) for r in rows]
