"""SQLite durable state for Hermes Agent Runtime."""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

from .models import JobClaim, RuntimeEvent, RuntimeJob, RuntimeRun
from .roles import get_role

VALID_RUN_STATUSES = {"planning", "running", "attention", "needs_approval", "done", "failed", "cancelled"}
VALID_JOB_STATUSES = {"planned", "ready", "leased", "running", "succeeded", "failed", "cancelled", "waiting_approval", "attention"}
VALID_WORKSPACE_KINDS = {"scratch", "repo", "worktree", "dir"}

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runtime_runs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    objective TEXT NOT NULL DEFAULT '',
    owner_source TEXT NOT NULL DEFAULT '',
    public_ref TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    risk_level TEXT NOT NULL DEFAULT 'medium',
    orchestrator_session_id TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    closed_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runtime_runs_status ON runtime_runs(status);
CREATE INDEX IF NOT EXISTS idx_runtime_runs_public_ref ON runtime_runs(public_ref);

CREATE TABLE IF NOT EXISTS runtime_jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runtime_runs(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    priority INTEGER NOT NULL DEFAULT 0,
    workspace_kind TEXT NOT NULL DEFAULT 'scratch',
    workspace_path TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL DEFAULT '',
    max_attempts INTEGER NOT NULL DEFAULT 2,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at INTEGER,
    heartbeat_at INTEGER,
    result_summary TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runtime_jobs_run_status ON runtime_jobs(run_id, status);
CREATE INDEX IF NOT EXISTS idx_runtime_jobs_ready ON runtime_jobs(status, priority, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runtime_jobs_idempotency
    ON runtime_jobs(run_id, idempotency_key)
    WHERE idempotency_key != '';

CREATE TABLE IF NOT EXISTS runtime_job_dependencies (
    parent_job_id TEXT NOT NULL REFERENCES runtime_jobs(id) ON DELETE CASCADE,
    child_job_id TEXT NOT NULL REFERENCES runtime_jobs(id) ON DELETE CASCADE,
    condition TEXT NOT NULL DEFAULT 'succeeded',
    PRIMARY KEY(parent_job_id, child_job_id)
);
CREATE INDEX IF NOT EXISTS idx_runtime_deps_child ON runtime_job_dependencies(child_job_id);

CREATE TABLE IF NOT EXISTS runtime_attempts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES runtime_jobs(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    pid INTEGER,
    status TEXT NOT NULL DEFAULT 'starting',
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    stdout_log TEXT NOT NULL DEFAULT '',
    stderr_log TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_runtime_attempts_job ON runtime_attempts(job_id, started_at);

CREATE TABLE IF NOT EXISTS runtime_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runtime_runs(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES runtime_jobs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runtime_events_run ON runtime_events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_runtime_events_job ON runtime_events(job_id, id);

CREATE TABLE IF NOT EXISTS runtime_artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runtime_runs(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES runtime_jobs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_findings (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runtime_runs(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES runtime_jobs(id) ON DELETE SET NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence_ref TEXT NOT NULL DEFAULT '',
    recommendation TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    created_at INTEGER NOT NULL,
    resolved_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runtime_findings_status ON runtime_findings(run_id, status, severity);

CREATE TABLE IF NOT EXISTS runtime_approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runtime_runs(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES runtime_jobs(id) ON DELETE SET NULL,
    target TEXT NOT NULL,
    commands_json TEXT NOT NULL DEFAULT '[]',
    command_hashes_json TEXT NOT NULL DEFAULT '[]',
    reason TEXT NOT NULL DEFAULT '',
    blast_radius TEXT NOT NULL DEFAULT '',
    rollback TEXT NOT NULL DEFAULT '',
    verification_json TEXT NOT NULL DEFAULT '[]',
    approved_by TEXT NOT NULL DEFAULT '',
    approval_source TEXT NOT NULL DEFAULT '',
    approved_at INTEGER NOT NULL,
    expires_at INTEGER,
    scope_hash TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS runtime_decisions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runtime_runs(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES runtime_jobs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    linked_findings_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
);
"""


def _now(now: Optional[int] = None) -> int:
    return int(time.time() if now is None else now)


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, ensure_ascii=False, sort_keys=True)


def _loads(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def runtime_home() -> Path:
    return get_hermes_home() / "agent-runtime"


def runtime_db_path() -> Path:
    return runtime_home() / "runtime.db"


def init_db(path: Path | str | None = None) -> Path:
    db_path = Path(path) if path else runtime_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return db_path


@contextlib.contextmanager
def connect(path: Path | str | None = None):
    db_path = Path(path) if path else runtime_db_path()
    if not db_path.exists():
        init_db(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _run_from_row(row: sqlite3.Row | None) -> RuntimeRun | None:
    if row is None:
        return None
    return RuntimeRun(
        id=row["id"], title=row["title"], objective=row["objective"],
        owner_source=row["owner_source"], public_ref=row["public_ref"],
        status=row["status"], risk_level=row["risk_level"],
        orchestrator_session_id=row["orchestrator_session_id"], summary=row["summary"],
        created_at=row["created_at"], updated_at=row["updated_at"], closed_at=row["closed_at"],
    )


def _job_from_row(row: sqlite3.Row | None) -> RuntimeJob | None:
    if row is None:
        return None
    return RuntimeJob(
        id=row["id"], run_id=row["run_id"], role=row["role"], title=row["title"],
        body=row["body"], status=row["status"], priority=row["priority"],
        workspace_kind=row["workspace_kind"], workspace_path=row["workspace_path"],
        idempotency_key=row["idempotency_key"], max_attempts=row["max_attempts"],
        attempt_count=row["attempt_count"], lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"], heartbeat_at=row["heartbeat_at"],
        result_summary=row["result_summary"], created_at=row["created_at"],
        started_at=row["started_at"], completed_at=row["completed_at"],
    )


def _event_from_row(row: sqlite3.Row) -> RuntimeEvent:
    return RuntimeEvent(
        id=row["id"], run_id=row["run_id"], job_id=row["job_id"], kind=row["kind"],
        payload=_loads(row["payload_json"], {}), created_at=row["created_at"],
    )


def add_event(conn: sqlite3.Connection, *, run_id: str, kind: str, job_id: str | None = None, payload: Any = None, now: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO runtime_events (run_id, job_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, job_id, kind, _json(payload or {}), _now(now)),
    )
    return int(cur.lastrowid)


def create_run(
    conn: sqlite3.Connection,
    *,
    title: str,
    objective: str = "",
    owner_source: str = "",
    public_ref: str = "",
    risk_level: str = "medium",
    orchestrator_session_id: str = "",
    now: int | None = None,
) -> str:
    ts = _now(now)
    run_id = _id("run")
    conn.execute(
        """
        INSERT INTO runtime_runs
        (id, title, objective, owner_source, public_ref, status, risk_level,
         orchestrator_session_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)
        """,
        (run_id, title, objective, owner_source, public_ref, risk_level, orchestrator_session_id, ts, ts),
    )
    add_event(conn, run_id=run_id, kind="run_created", payload={"title": title, "public_ref": public_ref}, now=ts)
    return run_id


def get_run(conn: sqlite3.Connection, run_id: str) -> RuntimeRun | None:
    return _run_from_row(conn.execute("SELECT * FROM runtime_runs WHERE id=?", (run_id,)).fetchone())


def list_runs(conn: sqlite3.Connection, *, limit: int = 50) -> list[RuntimeRun]:
    rows = conn.execute(
        "SELECT * FROM runtime_runs ORDER BY created_at DESC, id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    return [_run_from_row(r) for r in rows if r is not None]


def _run_exists(conn: sqlite3.Connection, run_id: str) -> bool:
    return conn.execute("SELECT 1 FROM runtime_runs WHERE id=?", (run_id,)).fetchone() is not None


def _run_status(conn: sqlite3.Connection, run_id: str) -> str | None:
    row = conn.execute("SELECT status FROM runtime_runs WHERE id=?", (run_id,)).fetchone()
    return str(row["status"]) if row is not None else None


def _job_exists(conn: sqlite3.Connection, job_id: str) -> bool:
    return conn.execute("SELECT 1 FROM runtime_jobs WHERE id=?", (job_id,)).fetchone() is not None


def _job_run_id(conn: sqlite3.Connection, job_id: str) -> str | None:
    row = conn.execute("SELECT run_id FROM runtime_jobs WHERE id=?", (job_id,)).fetchone()
    return str(row["run_id"]) if row is not None else None


def _validate_optional_job_ref(conn: sqlite3.Connection, *, run_id: str, job_id: str | None) -> None:
    if not job_id:
        return
    job_run_id = _job_run_id(conn, job_id)
    if job_run_id is None:
        raise ValueError(f"unknown job_id: {job_id}")
    if job_run_id != run_id:
        raise ValueError("job_id must belong to the same run")


def _approval_writer_authorized(approval_writer: Any | None = None) -> bool:
    # Approval writes are deliberately not authorized through an importable DB
    # helper.  The reviewed operator CLI validates packets and performs the
    # insert in its own command path; model-callable tools cannot self-mint by
    # importing this module or forging an in-process writer object.
    return False


def _validate_linked_findings(conn: sqlite3.Connection, *, run_id: str, linked_findings: Iterable[str] | None) -> None:
    for finding_id in linked_findings or []:
        row = conn.execute("SELECT run_id FROM runtime_findings WHERE id=?", (finding_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown finding_id: {finding_id}")
        if str(row["run_id"]) != run_id:
            raise ValueError("linked findings must belong to the same run")


def _validate_approval_packet(packet: dict[str, Any]) -> None:
    from .policy import approval_scope_hash, command_hash

    target = str(packet.get("target") or "").strip()
    commands = [str(c) for c in (packet.get("commands") or [])]
    command_hashes = [str(h) for h in (packet.get("command_hashes") or [])]
    if not target:
        raise ValueError("approval packet target is required")
    if not commands or any(not c for c in commands):
        raise ValueError("approval packet commands are required")
    if command_hashes != [command_hash(c) for c in commands]:
        raise ValueError("approval packet command hashes do not match commands")
    if not str(packet.get("approved_by") or "").strip():
        raise ValueError("approval packet approved_by is required")
    for field in ("reason", "blast_radius", "rollback"):
        if not str(packet.get(field) or "").strip():
            raise ValueError(f"approval packet {field} is required")
    verification = packet.get("verification") or []
    if not isinstance(verification, list) or not verification or any(not str(item).strip() for item in verification):
        raise ValueError("approval packet verification is required")
    from . import approval_channel

    approval_channel.validate_approval_source(str(packet.get("approval_source") or ""))
    expires_at = packet.get("expires_at")
    if expires_at is not None and not isinstance(expires_at, int):
        raise ValueError("approval packet expires_at must be an integer timestamp")
    if packet.get("scope_hash") != approval_scope_hash(target, commands):
        raise ValueError("approval packet scope hash does not match target and commands")


def validate_approval_packet(packet: dict[str, Any]) -> None:
    _validate_approval_packet(packet)


def create_job(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    role: str,
    title: str,
    body: str = "",
    depends_on: Iterable[str] | None = None,
    priority: int = 0,
    workspace_kind: str = "scratch",
    workspace_path: str = "",
    idempotency_key: str = "",
    max_attempts: int = 2,
    now: int | None = None,
) -> str:
    run_status = _run_status(conn, run_id)
    if run_status is None:
        raise ValueError(f"unknown run_id: {run_id}")
    if run_status in {"done", "failed", "cancelled"}:
        raise ValueError("cannot create runtime jobs under a terminal run")
    runtime_role = get_role(role)  # validates canonical role name
    if runtime_role.mode == "main_session":
        raise ValueError("runtime jobs must use a bounded worker role, not the main-session orchestrator")
    max_attempts = int(max_attempts)
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if workspace_kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(f"invalid workspace_kind: {workspace_kind}")
    parents = list(depends_on or [])
    for parent in parents:
        parent_run_id = _job_run_id(conn, parent)
        if parent_run_id is None:
            raise ValueError(f"unknown parent job: {parent}")
        if parent_run_id != run_id:
            raise ValueError("parent job dependencies must belong to the same run")
    status = "planned" if parents else "ready"
    ts = _now(now)
    job_id = _id("job")
    conn.execute(
        """
        INSERT INTO runtime_jobs
        (id, run_id, role, title, body, status, priority, workspace_kind,
         workspace_path, idempotency_key, max_attempts, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, run_id, runtime_role.name, title, body, status, int(priority), workspace_kind, workspace_path, idempotency_key, max_attempts, ts),
    )
    for parent in parents:
        conn.execute(
            "INSERT OR IGNORE INTO runtime_job_dependencies (parent_job_id, child_job_id, condition) VALUES (?, ?, 'succeeded')",
            (parent, job_id),
        )
    add_event(conn, run_id=run_id, job_id=job_id, kind="job_created", payload={"role": runtime_role.name, "status": status, "parents": parents}, now=ts)
    return job_id


def get_job(conn: sqlite3.Connection, job_id: str) -> RuntimeJob | None:
    return _job_from_row(conn.execute("SELECT * FROM runtime_jobs WHERE id=?", (job_id,)).fetchone())


def list_jobs(conn: sqlite3.Connection, run_id: str) -> list[RuntimeJob]:
    rows = conn.execute(
        "SELECT * FROM runtime_jobs WHERE run_id=? ORDER BY created_at, id",
        (run_id,),
    ).fetchall()
    return [_job_from_row(r) for r in rows if r is not None]


def complete_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    summary: str = "",
    now: int | None = None,
    lease_owner: str | None = None,
    attempt_id: str | None = None,
) -> None:
    job = get_job(conn, job_id)
    if job is None:
        raise ValueError(f"unknown job_id: {job_id}")
    run_status = _run_status(conn, job.run_id)
    if run_status not in {"planning", "running", "attention"}:
        raise ValueError("cannot complete jobs for a non-active run")
    ts = _now(now)
    if job.status in {"leased", "running"}:
        if not lease_owner or not attempt_id:
            raise ValueError("leased job completion requires lease_owner and attempt_id")
        updated = conn.execute(
            """
            UPDATE runtime_jobs
            SET status='succeeded', completed_at=?, lease_owner=NULL,
                lease_expires_at=NULL, heartbeat_at=NULL, result_summary=?
            WHERE id=?
              AND status IN ('leased', 'running')
              AND lease_owner=?
              AND (lease_expires_at IS NULL OR lease_expires_at > ?)
              AND EXISTS (
                  SELECT 1 FROM runtime_runs r
                  WHERE r.id = runtime_jobs.run_id
                    AND r.status IN ('planning', 'running', 'attention')
              )
              AND EXISTS (
                  SELECT 1 FROM runtime_attempts a
                  WHERE a.id=? AND a.job_id=runtime_jobs.id
                    AND a.status IN ('starting', 'running')
              )
            """,
            (ts, summary, job_id, lease_owner, ts, attempt_id),
        ).rowcount
        if updated != 1:
            raise ValueError("leased job completion lost the lease or active attempt")
        conn.execute(
            "UPDATE runtime_attempts SET status='succeeded', ended_at=?, summary=? WHERE id=? AND job_id=? AND status IN ('starting', 'running')",
            (ts, summary, attempt_id, job_id),
        )
    else:
        raise ValueError("job completion requires an active lease and attempt")
    add_event(conn, run_id=job.run_id, job_id=job_id, kind="job_succeeded", payload={"summary": summary}, now=ts)


def fail_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    error: str,
    now: int | None = None,
    lease_owner: str | None = None,
    attempt_id: str | None = None,
    allow_expired_lease: bool = False,
) -> None:
    job = get_job(conn, job_id)
    if job is None:
        raise ValueError(f"unknown job_id: {job_id}")
    run_status = _run_status(conn, job.run_id)
    if run_status not in {"planning", "running", "attention"}:
        raise ValueError("cannot fail jobs for a non-active run")
    if job.status not in {"leased", "running"}:
        raise ValueError("job failure requires a leased or running job")
    if not lease_owner or not attempt_id:
        raise ValueError("leased job failure requires lease_owner and attempt_id")
    ts = _now(now)
    # A stale worker-reported success/failure must not mutate state after lease
    # expiry.  The only caller that may set allow_expired_lease=True is the
    # trusted parent reaper after it has killed or observed a killed worker
    # subprocess: the lease owner + attempt predicates below must still match,
    # and recovered/re-leased jobs no longer satisfy them.
    next_status = "failed" if job.attempt_count >= job.max_attempts else "ready"
    updated = conn.execute(
        """
        UPDATE runtime_jobs
        SET status=?, lease_owner=NULL, lease_expires_at=NULL,
            heartbeat_at=NULL, result_summary=?,
            completed_at=CASE WHEN ?='failed' THEN ? ELSE completed_at END
        WHERE id=?
          AND status IN ('leased', 'running')
          AND lease_owner=?
          AND (lease_expires_at IS NULL OR lease_expires_at > ? OR ?)
          AND EXISTS (
              SELECT 1 FROM runtime_runs r
              WHERE r.id = runtime_jobs.run_id
                AND r.status IN ('planning', 'running', 'attention')
          )
          AND EXISTS (
              SELECT 1 FROM runtime_attempts a
              WHERE a.id=? AND a.job_id=runtime_jobs.id
                AND a.status IN ('starting', 'running')
          )
        """,
        (next_status, error, next_status, ts, job_id, lease_owner, ts, 1 if allow_expired_lease else 0, attempt_id),
    ).rowcount
    if updated != 1:
        raise ValueError("leased job failure lost the lease or active attempt")
    conn.execute(
        "UPDATE runtime_attempts SET status='failed', ended_at=?, error=? WHERE id=? AND job_id=? AND status IN ('starting', 'running')",
        (ts, error, attempt_id, job_id),
    )
    add_event(conn, run_id=job.run_id, job_id=job_id, kind="job_attempt_failed", payload={"error": error, "next_status": next_status}, now=ts)
    if next_status == "failed":
        add_event(conn, run_id=job.run_id, job_id=job_id, kind="job_failed", payload={"error": error}, now=ts)


def promote_ready_jobs(conn: sqlite3.Connection, *, run_id: str, now: int | None = None) -> list[str]:
    ts = _now(now)
    planned = conn.execute(
        "SELECT id FROM runtime_jobs WHERE run_id=? AND status='planned' ORDER BY priority DESC, created_at, id",
        (run_id,),
    ).fetchall()
    promoted: list[str] = []
    for row in planned:
        job_id = row["id"]
        blockers = conn.execute(
            """
            SELECT d.parent_job_id
            FROM runtime_job_dependencies d
            JOIN runtime_jobs p ON p.id = d.parent_job_id
            WHERE d.child_job_id=? AND p.status != 'succeeded'
            """,
            (job_id,),
        ).fetchall()
        if blockers:
            continue
        conn.execute("UPDATE runtime_jobs SET status='ready' WHERE id=?", (job_id,))
        add_event(conn, run_id=run_id, job_id=job_id, kind="job_ready", payload={}, now=ts)
        promoted.append(job_id)
    return promoted


def _create_attempt(conn: sqlite3.Connection, job: RuntimeJob, *, now: int, status: str = "starting") -> str:
    attempt_id = _id("att")
    role = get_role(job.role)
    conn.execute(
        """
        INSERT INTO runtime_attempts
        (id, job_id, role, model, reasoning, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (attempt_id, job.id, job.role, role.model, role.reasoning_effort, status, now),
    )
    return attempt_id


def claim_next_job(
    conn: sqlite3.Connection,
    *,
    lease_owner: str,
    lease_ttl_seconds: int = 900,
    now: int | None = None,
    role: str | None = None,
) -> JobClaim | None:
    ts = _now(now)
    params: list[Any] = []
    role_filter = ""
    if role:
        role_filter = " AND j.role=?"
        params.append(role)
    row = conn.execute(
        """
        SELECT j.*
        FROM runtime_jobs j
        JOIN runtime_runs r ON r.id = j.run_id
        WHERE j.status='ready'
          AND j.attempt_count < j.max_attempts
          AND r.status IN ('planning', 'running', 'attention')
        """ + role_filter + " ORDER BY j.priority DESC, j.created_at, j.id LIMIT 1",
        tuple(params),
    ).fetchone()
    job = _job_from_row(row)
    if job is None:
        return None
    expires = ts + max(1, int(lease_ttl_seconds))
    attempt_id = _create_attempt(conn, job, now=ts, status="running")
    updated = conn.execute(
        """
        UPDATE runtime_jobs
        SET status='leased', lease_owner=?, lease_expires_at=?, heartbeat_at=?,
            started_at=COALESCE(started_at, ?), attempt_count=attempt_count + 1
        WHERE id=? AND status='ready'
          AND attempt_count < max_attempts
          AND EXISTS (
              SELECT 1 FROM runtime_runs r
              WHERE r.id = runtime_jobs.run_id
                AND r.status IN ('planning', 'running', 'attention')
          )
        """,
        (lease_owner, expires, ts, ts, job.id),
    ).rowcount
    if updated != 1:
        conn.execute("DELETE FROM runtime_attempts WHERE id=?", (attempt_id,))
        return None
    add_event(conn, run_id=job.run_id, job_id=job.id, kind="job_leased", payload={"lease_owner": lease_owner, "attempt_id": attempt_id}, now=ts)
    return JobClaim(job_id=job.id, attempt_id=attempt_id, lease_owner=lease_owner, lease_expires_at=expires)


def heartbeat(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    lease_owner: str,
    attempt_id: str,
    lease_ttl_seconds: int = 900,
    now: int | None = None,
) -> None:
    ts = _now(now)
    expires = ts + max(1, int(lease_ttl_seconds))
    updated = conn.execute(
        """
            UPDATE runtime_jobs
            SET heartbeat_at=?, lease_expires_at=?
            WHERE id=?
              AND lease_owner=?
              AND status IN ('leased', 'running')
              AND (lease_expires_at IS NULL OR lease_expires_at > ?)
              AND EXISTS (
                  SELECT 1 FROM runtime_runs r
                  WHERE r.id = runtime_jobs.run_id
                    AND r.status IN ('planning', 'running', 'attention')
              )
              AND EXISTS (
                  SELECT 1 FROM runtime_attempts a
                  WHERE a.id=? AND a.job_id=runtime_jobs.id
                AND a.status IN ('starting', 'running')
          )
        """,
        (ts, expires, job_id, lease_owner, ts, attempt_id),
    ).rowcount
    if updated != 1:
        raise ValueError("job lease is not active for this owner/attempt or active run")


def recover_expired_leases(conn: sqlite3.Connection, *, now: int | None = None) -> list[str]:
    ts = _now(now)
    rows = conn.execute(
        """
        SELECT j.*
        FROM runtime_jobs j
        JOIN runtime_runs r ON r.id = j.run_id
        WHERE j.status IN ('leased', 'running')
          AND j.lease_expires_at IS NOT NULL
          AND j.lease_expires_at <= ?
          AND r.status IN ('planning', 'running', 'attention')
        ORDER BY j.lease_expires_at, j.id
        """,
        (ts,),
    ).fetchall()
    recovered: list[str] = []
    for row in rows:
        job = _job_from_row(row)
        if job is None:
            continue
        next_status = "failed" if job.attempt_count >= job.max_attempts else "ready"
        updated = conn.execute(
            """
            UPDATE runtime_jobs
            SET status=?, lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                completed_at=CASE WHEN ?='failed' THEN ? ELSE completed_at END
            WHERE id=?
              AND status IN ('leased', 'running')
              AND lease_owner IS ?
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= ?
              AND EXISTS (
                  SELECT 1 FROM runtime_runs r
                  WHERE r.id = runtime_jobs.run_id
                    AND r.status IN ('planning', 'running', 'attention')
              )
            """,
            (next_status, next_status, ts, job.id, job.lease_owner, ts),
        ).rowcount
        if updated != 1:
            continue
        conn.execute(
            "UPDATE runtime_attempts SET status='timed_out', ended_at=?, error=? WHERE job_id=? AND status IN ('starting', 'running')",
            (ts, "lease expired", job.id),
        )
        add_event(conn, run_id=job.run_id, job_id=job.id, kind="lease_expired", payload={"next_status": next_status}, now=ts)
        if next_status == "failed":
            add_event(conn, run_id=job.run_id, job_id=job.id, kind="job_failed", payload={"reason": "max attempts exhausted"}, now=ts)
        recovered.append(job.id)
    return recovered


def list_events(conn: sqlite3.Connection, *, run_id: str | None = None, job_id: str | None = None, limit: int = 100) -> list[RuntimeEvent]:
    if job_id:
        rows = conn.execute("SELECT * FROM runtime_events WHERE job_id=? ORDER BY id LIMIT ?", (job_id, int(limit))).fetchall()
    elif run_id:
        rows = conn.execute("SELECT * FROM runtime_events WHERE run_id=? ORDER BY id LIMIT ?", (run_id, int(limit))).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runtime_events ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [_event_from_row(r) for r in rows]


def add_finding(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    severity: str,
    category: str,
    summary: str,
    job_id: str | None = None,
    evidence_ref: str = "",
    recommendation: str = "",
    now: int | None = None,
) -> str:
    _validate_optional_job_ref(conn, run_id=run_id, job_id=job_id)
    ts = _now(now)
    finding_id = _id("find")
    conn.execute(
        """
        INSERT INTO runtime_findings
        (id, run_id, job_id, severity, category, summary, evidence_ref, recommendation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (finding_id, run_id, job_id, severity, category, summary, evidence_ref, recommendation, ts),
    )
    add_event(conn, run_id=run_id, job_id=job_id, kind="finding_added", payload={"finding_id": finding_id, "severity": severity, "category": category}, now=ts)
    return finding_id


def record_decision(conn: sqlite3.Connection, *, run_id: str, kind: str, rationale: str = "", job_id: str | None = None, linked_findings: list[str] | None = None, now: int | None = None) -> str:
    _validate_optional_job_ref(conn, run_id=run_id, job_id=job_id)
    _validate_linked_findings(conn, run_id=run_id, linked_findings=linked_findings)
    ts = _now(now)
    decision_id = _id("dec")
    conn.execute(
        "INSERT INTO runtime_decisions (id, run_id, job_id, kind, rationale, linked_findings_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (decision_id, run_id, job_id, kind, rationale, _json(linked_findings or []), ts),
    )
    add_event(conn, run_id=run_id, job_id=job_id, kind="decision_recorded", payload={"decision_id": decision_id, "kind": kind}, now=ts)
    return decision_id


def record_approval(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    packet: dict[str, Any],
    job_id: str | None = None,
    approval_writer: Any | None = None,
    now: int | None = None,
) -> str:
    if not _approval_writer_authorized(approval_writer):
        raise PermissionError("runtime approval writer is not authorized")
    if not _run_exists(conn, run_id):
        raise ValueError(f"unknown run_id: {run_id}")
    _validate_optional_job_ref(conn, run_id=run_id, job_id=job_id)
    _validate_approval_packet(packet)
    ts = _now(now)
    approval_id = _id("appr")
    conn.execute(
        """
        INSERT INTO runtime_approvals
        (id, run_id, job_id, target, commands_json, command_hashes_json, reason,
         blast_radius, rollback, verification_json, approved_by, approval_source,
         approved_at, expires_at, scope_hash, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
        """,
        (
            approval_id,
            run_id,
            job_id,
            packet.get("target", ""),
            _json(packet.get("commands") or []),
            _json(packet.get("command_hashes") or []),
            packet.get("reason", ""),
            packet.get("blast_radius", ""),
            packet.get("rollback", ""),
            _json(packet.get("verification") or []),
            packet.get("approved_by", ""),
            packet.get("approval_source", ""),
            int(packet.get("approved_at") or ts),
            packet.get("expires_at"),
            packet.get("scope_hash", ""),
        ),
    )
    add_event(conn, run_id=run_id, job_id=job_id, kind="approval_recorded", payload={"approval_id": approval_id, "target": packet.get("target", "")}, now=ts)
    return approval_id


def _approval_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "job_id": row["job_id"],
        "target": row["target"],
        "commands": _loads(row["commands_json"], []),
        "command_hashes": _loads(row["command_hashes_json"], []),
        "reason": row["reason"],
        "blast_radius": row["blast_radius"],
        "rollback": row["rollback"],
        "verification": _loads(row["verification_json"], []),
        "approved_by": row["approved_by"],
        "approval_source": row["approval_source"],
        "approved_at": row["approved_at"],
        "expires_at": row["expires_at"],
        "scope_hash": row["scope_hash"],
        "status": row["status"],
    }


def list_active_approvals_for_worker(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    job_id: str,
    now: int | None = None,
) -> list[dict[str, Any]]:
    """Return approval snapshots a worker may consult without DB access.

    Job-scoped approvals are visible only to the matching job. Run-level
    approvals (`job_id IS NULL`) are visible to every worker in the run.
    """
    ts = _now(now)
    rows = conn.execute(
        """
        SELECT * FROM runtime_approvals
        WHERE run_id=? AND status='active'
          AND (job_id IS NULL OR job_id=?)
          AND (expires_at IS NULL OR expires_at >= ?)
        ORDER BY approved_at DESC, id DESC
        """,
        (run_id, job_id, ts),
    ).fetchall()
    return [_approval_row_to_dict(row) for row in rows]


def find_approval_for_command(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    target: str,
    command: str,
    job_id: str | None = None,
    scope_hash: str | None = None,
    now: int | None = None,
) -> dict[str, Any] | None:
    from .policy import approval_scope_hash, command_hash

    ts = _now(now)
    h = command_hash(command)
    scoped_job_id = str(job_id or "").strip()
    if scoped_job_id:
        job_scope_sql = "AND (job_id IS NULL OR job_id=?)"
        params: tuple[Any, ...] = (run_id, target, ts, scoped_job_id)
    else:
        # Callers without a concrete worker job must not receive job-scoped
        # approvals that were issued for a different job in the same run.
        job_scope_sql = "AND job_id IS NULL"
        params = (run_id, target, ts)
    rows = conn.execute(
        f"""
        SELECT * FROM runtime_approvals
        WHERE run_id=? AND target=? AND status='active'
          AND (expires_at IS NULL OR expires_at >= ?)
          {job_scope_sql}
        ORDER BY approved_at DESC, id DESC
        """,
        params,
    ).fetchall()
    for row in rows:
        payload = _approval_row_to_dict(row)
        commands = [str(c) for c in (payload.get("commands") or [])]
        if command not in commands:
            continue
        expected_hashes = [command_hash(c) for c in commands]
        if list(payload.get("command_hashes") or []) != expected_hashes:
            continue
        expected_scope_hash = approval_scope_hash(target, commands)
        if payload.get("scope_hash") != expected_scope_hash:
            continue
        if scope_hash is not None and scope_hash != expected_scope_hash:
            continue
        if h in set(expected_hashes):
            return payload
    return None


def close_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    status: str = "done",
    summary: str = "",
    now: int | None = None,
) -> None:
    if status not in {"done", "failed", "cancelled"}:
        raise ValueError(f"invalid terminal run status: {status}")
    if not _run_exists(conn, run_id):
        raise ValueError(f"unknown run_id: {run_id}")
    ts = _now(now)
    conn.execute(
        "UPDATE runtime_runs SET status=?, summary=?, updated_at=?, closed_at=? WHERE id=?",
        (status, summary, ts, ts, run_id),
    )
    job_status = "failed" if status == "failed" else "cancelled"
    conn.execute(
        """
        UPDATE runtime_jobs
        SET status=?, lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
            completed_at=COALESCE(completed_at, ?)
        WHERE run_id=? AND status NOT IN ('succeeded', 'failed', 'cancelled')
        """,
        (job_status, ts, run_id),
    )
    conn.execute(
        """
        UPDATE runtime_attempts
        SET status=?, ended_at=COALESCE(ended_at, ?), error=COALESCE(NULLIF(error, ''), ?)
        WHERE job_id IN (SELECT id FROM runtime_jobs WHERE run_id=?)
          AND status IN ('starting', 'running')
        """,
        (job_status, ts, f"run closed: {status}", run_id),
    )
    add_event(conn, run_id=run_id, kind="run_closed", payload={"status": status, "summary": summary}, now=ts)


def doctor_status(conn: sqlite3.Connection) -> dict[str, Any]:
    runs = int(conn.execute("SELECT COUNT(*) FROM runtime_runs").fetchone()[0])
    jobs = int(conn.execute("SELECT COUNT(*) FROM runtime_jobs").fetchone()[0])
    ready_jobs = int(conn.execute("SELECT COUNT(*) FROM runtime_jobs WHERE status='ready'").fetchone()[0])
    leased_jobs = int(conn.execute("SELECT COUNT(*) FROM runtime_jobs WHERE status IN ('leased','running')").fetchone()[0])
    open_findings = int(conn.execute("SELECT COUNT(*) FROM runtime_findings WHERE status='open'").fetchone()[0])
    active_approvals = int(conn.execute("SELECT COUNT(*) FROM runtime_approvals WHERE status='active'").fetchone()[0])
    return {
        "ok": True,
        "db_path": str(runtime_db_path()),
        "runs": runs,
        "jobs": jobs,
        "ready_jobs": ready_jobs,
        "leased_jobs": leased_jobs,
        "open_findings": open_findings,
        "active_approvals": active_approvals,
    }
