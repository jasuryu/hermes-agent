"""Trusted worker lease/context broker for Agent Runtime.

This module is part of the trusted control plane.  It validates a live runtime
lease against the writable runtime DB, creates an isolated scratch filesystem
layout, materializes a sanitized, read-only context snapshot for worker
execution, and records subprocess results through parent-side lease predicates.
Workers still never open the runtime DB directly; worker_main consumes only the
brokered context file and the parent records completion/failure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import tempfile
import time
from typing import Any

from . import db, rag_broker
from .models import RuntimeJob, RuntimeRun


@dataclass(frozen=True)
class WorkerSandbox:
    root: Path
    home: Path
    workdir: Path
    tmp: Path
    xdg_config_home: Path
    xdg_cache_home: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class WorkerContextBundle:
    context_path: Path
    sandbox: WorkerSandbox
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_path": str(self.context_path),
            "sandbox": self.sandbox.to_dict(),
            "context": self.context,
        }


@dataclass(frozen=True)
class WorkerResultRecord:
    job_id: str
    attempt_id: str
    lease_owner: str
    success: bool
    exit_code: int
    summary: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now(now: int | None = None) -> int:
    return int(time.time() if now is None else now)


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "runtime").strip(".-")
    return cleaned[:80] or "runtime"


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _ensure_private_workspace_root(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("worker sandbox workspace_root must not be a symlink")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(path, 0o700)
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("worker sandbox workspace_root must not be a symlink")
    if not stat.S_ISDIR(st.st_mode):
        raise ValueError("worker sandbox workspace_root must be a directory")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise ValueError("worker sandbox workspace_root must be owned by the current user")
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise ValueError("worker sandbox workspace_root must not be group/world accessible")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    os.close(fd)


def create_worker_sandbox(
    *,
    workspace_root: str | Path | None = None,
    job_id: str,
    attempt_id: str,
    hermes_home: str | Path | None = None,
) -> WorkerSandbox:
    """Create a private scratch layout outside HERMES_HOME for one worker.

    The worker should receive this scratch directory as HOME/TMPDIR/cwd.  The
    runtime DB remains outside this tree and is not passed to the worker.
    """
    requested_root = Path(workspace_root) if workspace_root is not None else Path("/tmp") / "hermes-agent-runtime-workers"
    hermes = Path(hermes_home) if hermes_home is not None else db.runtime_home()
    if requested_root.is_symlink():
        raise ValueError("worker sandbox workspace_root must not be a symlink")
    if requested_root.parent.is_symlink():
        raise ValueError("worker sandbox workspace_root parent must not be a symlink")
    if _is_relative_to(requested_root, hermes):
        raise ValueError("worker sandbox workspace_root must not live under HERMES_HOME")
    trusted_parent = Path("/tmp")
    if _is_relative_to(trusted_parent, hermes):
        raise ValueError("worker sandbox trusted temp parent must not live under HERMES_HOME")
    if trusted_parent.is_symlink() or not trusted_parent.is_dir():
        raise ValueError("worker sandbox trusted temp parent must be a real directory")

    # Treat workspace_root as a naming prefix only.  The actual sandbox root is
    # created in a fixed trusted temp parent with tempfile.mkdtemp's atomic
    # random directory creation, never by trusting/reusing caller-supplied or
    # ambient TMPDIR paths.
    sandbox_root = Path(tempfile.mkdtemp(
        prefix=f"{_safe_component(requested_root.name)}-{_safe_component(job_id)}-{_safe_component(attempt_id)}-",
        dir=str(trusted_parent),
    ))
    os.chmod(sandbox_root, 0o700)
    if _is_relative_to(sandbox_root, hermes):
        raise ValueError("worker sandbox root must not live under HERMES_HOME")
    home = sandbox_root / "home"
    workdir = sandbox_root / "work"
    tmp = sandbox_root / "tmp"
    xdg_config_home = sandbox_root / "xdg-config"
    xdg_cache_home = sandbox_root / "xdg-cache"
    for path in (sandbox_root, home, workdir, tmp, xdg_config_home, xdg_cache_home):
        _mkdir_private(path)
    return WorkerSandbox(
        root=sandbox_root,
        home=home,
        workdir=workdir,
        tmp=tmp,
        xdg_config_home=xdg_config_home,
        xdg_cache_home=xdg_cache_home,
    )


def _worker_approval_snapshot(approval: dict[str, Any]) -> dict[str, Any]:
    """Minimize approval data exposed to workers for command matching."""
    return {
        "id": approval.get("id", ""),
        "run_id": approval.get("run_id", ""),
        "job_id": approval.get("job_id"),
        "target": approval.get("target", ""),
        "commands": list(approval.get("commands") or []),
        "command_hashes": list(approval.get("command_hashes") or []),
        "expires_at": approval.get("expires_at"),
        "scope_hash": approval.get("scope_hash", ""),
        "status": approval.get("status", "active"),
    }


def _active_job_for_lease(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    lease_owner: str,
    attempt_id: str,
    now: int | None = None,
) -> tuple[RuntimeRun, RuntimeJob]:
    ts = _now(now)
    row = conn.execute(
        """
        SELECT 1
        FROM runtime_jobs j
        JOIN runtime_runs r ON r.id = j.run_id
        JOIN runtime_attempts a ON a.job_id = j.id
        WHERE j.id=?
          AND j.status IN ('leased', 'running')
          AND j.lease_owner=?
          AND (j.lease_expires_at IS NULL OR j.lease_expires_at > ?)
          AND r.status IN ('planning', 'running', 'attention')
          AND a.id=?
          AND a.status IN ('starting', 'running')
        LIMIT 1
        """,
        (job_id, lease_owner, ts, attempt_id),
    ).fetchone()
    if row is None:
        raise ValueError("job does not have an active lease for this owner/attempt")
    job = db.get_job(conn, job_id)
    if job is None:
        raise ValueError(f"unknown job_id: {job_id}")
    run = db.get_run(conn, job.run_id)
    if run is None:
        raise ValueError(f"unknown run_id: {job.run_id}")
    return run, job


def build_worker_context(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    lease_owner: str,
    attempt_id: str,
    now: int | None = None,
    rag_retriever: Any | None = None,
) -> dict[str, Any]:
    """Validate an active lease and return a sanitized worker context snapshot."""
    ts = _now(now)
    run, job = _active_job_for_lease(
        conn,
        job_id=job_id,
        lease_owner=lease_owner,
        attempt_id=attempt_id,
        now=ts,
    )
    brokered_rag = rag_broker.build_brokered_rag_context(
        run=run,
        job=job,
        retriever=rag_retriever,
        now=ts,
    )
    return {
        "version": 1,
        "issued_at": ts,
        "expires_at": job.lease_expires_at,
        "lease": {
            "attempt_id": attempt_id,
            "lease_owner": lease_owner,
        },
        "run": {
            "id": run.id,
            "title": run.title,
            "objective": run.objective,
            "public_ref": run.public_ref,
            "risk_level": run.risk_level,
        },
        "job": {
            "id": job.id,
            "run_id": job.run_id,
            "role": job.role,
            "title": job.title,
            "body": job.body,
            "workspace_kind": job.workspace_kind,
            "workspace_path": job.workspace_path,
        },
        "constraints": {
            "runtime_db_access": "forbidden",
            "result_submission": "trusted_broker_required",
            "approval_creation": "trusted_operator_channel_required",
        },
        "rag": brokered_rag,
        "approvals": [
            _worker_approval_snapshot(approval)
            for approval in db.list_active_approvals_for_worker(conn, run_id=run.id, job_id=job.id, now=ts)
        ],
    }


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    tmp = Path(tmp_name)
    os.chmod(tmp, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    os.chmod(path, 0o600)


def materialize_worker_context(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    lease_owner: str,
    attempt_id: str,
    workspace_root: str | Path | None = None,
    hermes_home: str | Path | None = None,
    now: int | None = None,
    rag_retriever: Any | None = None,
) -> WorkerContextBundle:
    """Validate a lease, create sandbox dirs, and write sanitized context JSON."""
    context = build_worker_context(
        conn,
        job_id=job_id,
        lease_owner=lease_owner,
        attempt_id=attempt_id,
        now=now,
        rag_retriever=rag_retriever,
    )
    sandbox = create_worker_sandbox(
        workspace_root=workspace_root,
        job_id=job_id,
        attempt_id=attempt_id,
        hermes_home=hermes_home,
    )
    context_path = sandbox.root / "context.json"
    _write_private_json(context_path, context)
    return WorkerContextBundle(context_path=context_path, sandbox=sandbox, context=context)


def _decode_output(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in (stdout or "").splitlines()]):
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("worker result JSON must be an object")
        return payload
    raise ValueError("worker did not emit a result JSON object")


def _bounded_text(value: str, *, limit: int = 4000) -> str:
    text = value or ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def record_worker_result(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    lease_owner: str,
    attempt_id: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    now: int | None = None,
    allow_expired_lease_failure: bool = False,
) -> WorkerResultRecord:
    """Record one worker subprocess result through the trusted parent broker.

    Stdout is treated as untrusted.  A job succeeds only when the process exits 0
    and the last non-empty stdout line is a JSON object with `success: true`.
    All DB mutations are delegated to db.complete_job/db.fail_job so active
    lease+attempt predicates remain authoritative.  Expired leases are accepted
    only when the trusted parent reaper passes allow_expired_lease_failure=True
    after killing or observing a killed worker; stale success is never allowed.
    """
    code = int(exit_code)
    payload: dict[str, Any] = {}
    parse_error = ""
    try:
        payload = _parse_worker_stdout(stdout)
    except Exception as exc:
        parse_error = str(exc)

    if code == 0 and not parse_error and payload.get("success") is True:
        summary = _bounded_text(str(payload.get("summary") or "worker completed"))
        db.complete_job(conn, job_id, summary=summary, lease_owner=lease_owner, attempt_id=attempt_id, now=now)
        return WorkerResultRecord(job_id=job_id, attempt_id=attempt_id, lease_owner=lease_owner, success=True, exit_code=code, summary=summary)

    if code != 0:
        detail = _bounded_text(stderr or str(payload.get("error") or payload.get("summary") or ""))
        error = f"worker exited with exit code {code}"
        if detail:
            error = f"{error}: {detail}"
    elif parse_error:
        error = f"worker result invalid: {parse_error}"
    else:
        error = _bounded_text(str(payload.get("error") or payload.get("summary") or "worker reported failure"))
    db.fail_job(
        conn,
        job_id,
        error=error,
        lease_owner=lease_owner,
        attempt_id=attempt_id,
        now=now,
        allow_expired_lease=allow_expired_lease_failure,
    )
    return WorkerResultRecord(job_id=job_id, attempt_id=attempt_id, lease_owner=lease_owner, success=False, exit_code=code, error=error)


def reap_worker_process(
    conn: sqlite3.Connection,
    *,
    process: Any,
    job_id: str,
    lease_owner: str,
    attempt_id: str,
    timeout: int | None = None,
    now: int | None = None,
) -> WorkerResultRecord:
    """Wait for a worker process and record success/failure via the broker."""
    allow_expired_failure = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        exit_code = getattr(process, "returncode", 0)
        allow_expired_failure = exit_code is not None and int(exit_code) < 0
    except subprocess.TimeoutExpired:
        allow_expired_failure = True
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
        try:
            stdout, stderr = process.communicate(timeout=None)
        except Exception:
            stdout, stderr = "", ""
        exit_code = getattr(process, "returncode", -9)
        stderr = f"worker timed out after {timeout}s\n{_decode_output(stderr)}"
    return record_worker_result(
        conn,
        job_id=job_id,
        lease_owner=lease_owner,
        attempt_id=attempt_id,
        exit_code=int(exit_code if exit_code is not None else -1),
        stdout=_decode_output(stdout),
        stderr=_decode_output(stderr),
        now=now,
        allow_expired_lease_failure=allow_expired_failure,
    )
