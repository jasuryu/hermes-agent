"""Small scheduler primitives for Agent Runtime.

Scheduling is deliberately conservative: default ticks only recover expired
leases and promote DAG children.  Worker subprocess spawn is available only when
the operator passes the explicit spawn gate and a reviewed isolation launch
backend is available; default CLI/daemon execution remains recovery-only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import sqlite3
import subprocess
import time
from typing import Any, Callable

from . import db, spawner, worker_broker, worker_isolation
from .models import JobClaim
from .worker_isolation import assess_worker_isolation


@dataclass(frozen=True)
class DispatchResult:
    recovered: int = 0
    promoted: int = 0
    claimed: int = 0
    spawned: int = 0
    claims: tuple[JobClaim, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["claims"] = [claim.to_dict() for claim in self.claims]
        data["errors"] = list(self.errors)
        return data


def _active_runs_promote(conn: sqlite3.Connection, *, now: int | None) -> tuple[int, int]:
    recovered = db.recover_expired_leases(conn, now=now)
    promoted_count = 0
    for run in db.list_runs(conn, limit=200):
        if run.status in {"running", "planning", "attention"}:
            promoted_count += len(db.promote_ready_jobs(conn, run_id=run.id, now=now))
    return len(recovered), promoted_count


def _fail_claim(
    conn: sqlite3.Connection,
    claim: JobClaim,
    *,
    error: str,
    now: int | None,
    allow_expired_lease: bool = False,
) -> None:
    db.fail_job(
        conn,
        claim.job_id,
        error=error,
        lease_owner=claim.lease_owner,
        attempt_id=claim.attempt_id,
        now=now,
        allow_expired_lease=allow_expired_lease,
    )


def _record_attempt_pid(conn: sqlite3.Connection, claim: JobClaim, pid: int, *, now: int | None) -> None:
    ts = int(time.time() if now is None else now)
    updated = conn.execute(
        """
        UPDATE runtime_attempts
        SET pid=?
        WHERE id=? AND job_id=?
          AND status IN ('starting', 'running')
          AND EXISTS (
              SELECT 1
              FROM runtime_jobs j
              JOIN runtime_runs r ON r.id = j.run_id
              WHERE j.id = runtime_attempts.job_id
                AND j.status IN ('leased', 'running')
                AND j.lease_owner=?
                AND (j.lease_expires_at IS NULL OR j.lease_expires_at > ?)
                AND r.status IN ('planning', 'running', 'attention')
          )
        """,
        (int(pid), claim.attempt_id, claim.job_id, claim.lease_owner, ts),
    ).rowcount
    if updated != 1:
        raise ValueError("pid update lost active lease or attempt")
    conn.commit()


def _terminate_process_safely(process: Any) -> None:
    kill = getattr(process, "kill", None)
    if callable(kill):
        try:
            kill()
        except Exception:
            pass
    communicate = getattr(process, "communicate", None)
    if callable(communicate):
        try:
            communicate(timeout=None)
        except Exception:
            pass


def _spawn_claimed_worker(
    conn: sqlite3.Connection,
    claim: JobClaim,
    *,
    isolation_backend: str,
    executable_resolver: Callable[[str], str | None] | None,
    popen_factory: Callable[..., Any],
    workspace_root: str | None,
    worker_timeout_seconds: int,
    now: int | None,
) -> tuple[bool, str]:
    job = db.get_job(conn, claim.job_id)
    if job is None:
        raise ValueError(f"claimed runtime job disappeared: {claim.job_id}")
    try:
        bundle = worker_broker.materialize_worker_context(
            conn,
            job_id=claim.job_id,
            lease_owner=claim.lease_owner,
            attempt_id=claim.attempt_id,
            workspace_root=workspace_root,
            now=now,
        )
        invocation = spawner.build_worker_invocation(
            job_id=job.id,
            run_id=job.run_id,
            role=job.role,
            attempt_id=claim.attempt_id,
            lease_owner=claim.lease_owner,
            context_path=bundle.context_path,
            sandbox=bundle.sandbox,
            enable_execution=True,
        )
        plan = worker_isolation.build_launch_plan(
            backend=isolation_backend,
            worker_argv=invocation.argv,
            worker_env=invocation.env,
            cwd=invocation.cwd,
            sandbox=bundle.sandbox,
            context_path=bundle.context_path,
            executable_resolver=executable_resolver,
        )
        if not plan.allows_spawn:
            raise RuntimeError(plan.reason or "worker launch plan does not allow spawn")
        # Durable claim/attempt + brokered context must be committed before the
        # child process is launched.  Future worker/broker processes observe the
        # same SQLite DB, not this connection's uncommitted transaction.
        conn.commit()
    except Exception as exc:
        error = f"worker spawn failed: {exc}"
        try:
            _fail_claim(conn, claim, error=error, now=now)
        except Exception as fail_exc:
            return False, f"{error}; additionally failed to release claim: {fail_exc}"
        return False, error

    try:
        process = popen_factory(
            list(plan.argv),
            cwd=str(plan.cwd),
            env=plan.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        error = f"worker spawn failed: {exc}"
        try:
            _fail_claim(conn, claim, error=error, now=now)
        except Exception as fail_exc:
            return False, f"{error}; additionally failed to release claim: {fail_exc}"
        return False, error

    try:
        pid = getattr(process, "pid", None)
        if pid is not None:
            _record_attempt_pid(conn, claim, int(pid), now=now)
    except Exception as exc:
        error = f"worker post-spawn bookkeeping failed: {exc}"
        _terminate_process_safely(process)
        try:
            _fail_claim(conn, claim, error=error, now=now)
        except Exception as fail_exc:
            return True, f"{error}; additionally failed to release claim: {fail_exc}"
        return True, error

    try:
        record = worker_broker.reap_worker_process(
            conn,
            process=process,
            job_id=claim.job_id,
            lease_owner=claim.lease_owner,
            attempt_id=claim.attempt_id,
            timeout=worker_timeout_seconds,
            now=now,
        )
    except Exception as exc:
        error = f"worker reaper failed: {exc}"
        _terminate_process_safely(process)
        try:
            _fail_claim(conn, claim, error=error, now=now)
        except Exception as fail_exc:
            return True, f"{error}; additionally failed to release claim: {fail_exc}"
        return True, error
    if not record.success:
        return True, record.error or "worker failed"
    return True, ""


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    lease_owner: str = "agent-runtime-daemon",
    spawn: bool = False,
    enable_spawn: bool = False,
    max_claims: int = 1,
    now: int | None = None,
    isolation_backend: str = "disabled",
    executable_resolver: Callable[[str], str | None] | None = None,
    popen_factory: Callable[..., Any] | None = None,
    workspace_root: str | None = None,
    lease_ttl_seconds: int = 900,
    worker_timeout_seconds: int | None = None,
) -> DispatchResult:
    recovered_count, promoted_count = _active_runs_promote(conn, now=now)

    if not spawn:
        return DispatchResult(recovered=recovered_count, promoted=promoted_count)

    if not enable_spawn:
        return DispatchResult(
            recovered=recovered_count,
            promoted=promoted_count,
            errors=("spawn mode requires explicit enable_spawn=True and a reviewed isolation backend",),
        )

    assessment = assess_worker_isolation(backend=isolation_backend, executable_resolver=executable_resolver)
    if not assessment.available:
        return DispatchResult(
            recovered=recovered_count,
            promoted=promoted_count,
            errors=(f"spawn mode requires an available reviewed isolation backend; {assessment.reason}",),
        )
    if not assessment.allows_spawn:
        return DispatchResult(
            recovered=recovered_count,
            promoted=promoted_count,
            errors=(f"spawn mode requires a reviewed isolation launch policy; {assessment.reason}",),
        )

    popen = popen_factory or subprocess.Popen
    lease_ttl = max(2, int(lease_ttl_seconds or 2))
    requested_timeout = int(worker_timeout_seconds) if worker_timeout_seconds is not None else lease_ttl - 1
    timeout = max(1, min(requested_timeout, lease_ttl - 1))
    claims: list[JobClaim] = []
    spawned = 0
    errors: list[str] = []
    for _ in range(max(1, int(max_claims or 1))):
        claim = db.claim_next_job(
            conn,
            lease_owner=lease_owner,
            lease_ttl_seconds=lease_ttl,
            now=now,
        )
        if claim is None:
            break
        claims.append(claim)
        did_spawn, error = _spawn_claimed_worker(
            conn,
            claim,
            isolation_backend=assessment.backend,
            executable_resolver=executable_resolver,
            popen_factory=popen,
            workspace_root=workspace_root,
            worker_timeout_seconds=timeout,
            now=now,
        )
        if did_spawn:
            spawned += 1
        if error:
            errors.append(error)
            break

    return DispatchResult(
        recovered=recovered_count,
        promoted=promoted_count,
        claimed=len(claims),
        spawned=spawned,
        claims=tuple(claims),
        errors=tuple(errors),
    )
