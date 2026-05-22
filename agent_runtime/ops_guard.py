"""Mandatory ops command guard for Agent Runtime workers.

This module is deliberately usable inside the isolated worker process without
opening the writable Runtime DB.  The trusted parent broker materializes a
read-only context snapshot that contains only currently-active approval packet
metadata for the worker's run/job.  The guard combines deterministic command
classification with exact approval matching before any ops command can execute.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Callable, Mapping

from . import policy


_SECRET_ENV_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "KUBECONFIG")
_ALLOWED_EXEC_ENV = {
    "HOME",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "TERM",
    "TZ",
    "USER",
}


@dataclass(frozen=True)
class OpsCommandGuardResult:
    allowed: bool
    requires_approval: bool
    category: str
    reason: str
    approval_id: str = ""
    command: str = ""
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpsCommandExecution:
    exit_code: int
    output: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dict_value(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _approval_snapshots(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    approvals = context.get("approvals")
    if not isinstance(approvals, list):
        return []
    return [item for item in approvals if isinstance(item, dict)]


def _effective_now(now: int | None = None) -> int:
    return int(time.time() if now is None else now)


def _context_expiry_error(context: Mapping[str, Any], *, now: int | None = None) -> str:
    expires_at = context.get("expires_at")
    if not isinstance(expires_at, int) or isinstance(expires_at, bool):
        return "worker context is missing integer expires_at lease expiry"
    if _effective_now(now) >= expires_at:
        return "worker context expired with its runtime lease"
    return ""


def guard_ops_command(
    context: Mapping[str, Any],
    *,
    command: str,
    target: str,
    now: int | None = None,
) -> OpsCommandGuardResult:
    job = _dict_value(context, "job")
    constraints = _dict_value(context, "constraints")
    job_id = str(job.get("id") or "")
    if constraints.get("runtime_db_access") != "forbidden":
        return OpsCommandGuardResult(False, True, "invalid_context", "worker context does not forbid runtime DB access", command=command, target=target)
    if not job_id:
        return OpsCommandGuardResult(False, True, "invalid_context", "worker context is missing job identity", command=command, target=target)
    expiry_error = _context_expiry_error(context, now=now)
    if expiry_error:
        return OpsCommandGuardResult(False, True, "invalid_context", expiry_error, command=command, target=target)

    verdict = policy.classify_command(command)
    if verdict.allowed_without_approval:
        return OpsCommandGuardResult(True, False, verdict.category, verdict.reason, command=command, target=target)

    for approval in _approval_snapshots(context):
        scoped_job_id = str(approval.get("job_id") or "")
        if scoped_job_id and scoped_job_id != job_id:
            continue
        if str(approval.get("status") or "") != "active":
            continue
        if policy.approval_allows_command(approval, command, target=target, now=now):
            return OpsCommandGuardResult(
                True,
                True,
                verdict.category,
                "command allowed by exact runtime approval packet",
                approval_id=str(approval.get("id") or ""),
                command=command,
                target=target,
            )
    return OpsCommandGuardResult(
        False,
        verdict.requires_approval,
        verdict.category,
        "command requires an exact active runtime approval packet for this target and job scope",
        command=command,
        target=target,
    )


def _safe_exec_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = environ if environ is not None else os.environ
    safe: dict[str, str] = {}
    for key, value in env.items():
        upper = str(key).upper()
        if any(fragment in upper for fragment in _SECRET_ENV_FRAGMENTS):
            continue
        if key in {"HERMES_HOME", "PYTHONPATH", "VIRTUAL_ENV", "PATH"}:
            continue
        if key in _ALLOWED_EXEC_ENV:
            safe[str(key)] = str(value)
    return safe


def _default_runner(argv: list[str], *, cwd: str, timeout: int, env: dict[str, str]) -> OpsCommandExecution:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            timeout=timeout,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return OpsCommandExecution(exit_code=-1, output="", error=str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return OpsCommandExecution(exit_code=-1, output=stdout, error=stderr or f"command timed out after {timeout}s")
    return OpsCommandExecution(exit_code=int(completed.returncode), output=completed.stdout or "", error=completed.stderr or "")


def guarded_ops_terminal(
    context: Mapping[str, Any],
    *,
    command: str,
    target: str,
    timeout: int = 120,
    runner: Callable[..., OpsCommandExecution] | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    guard = guard_ops_command(context, command=command, target=target, now=now)
    if not guard.allowed:
        return {"status": "blocked", "guard": guard.to_dict(), "output": "", "exit_code": -1, "error": guard.reason}
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        blocked = OpsCommandGuardResult(False, True, "invalid_command", f"command could not be parsed safely: {exc}", command=command, target=target)
        return {"status": "blocked", "guard": blocked.to_dict(), "output": "", "exit_code": -1, "error": blocked.reason}
    if not argv:
        blocked = OpsCommandGuardResult(False, True, "invalid_command", "empty command is not allowed", command=command, target=target)
        return {"status": "blocked", "guard": blocked.to_dict(), "output": "", "exit_code": -1, "error": blocked.reason}
    exec_runner = runner or _default_runner
    execution = exec_runner(argv, cwd=os.getcwd(), timeout=int(timeout), env=_safe_exec_env())
    if isinstance(execution, OpsCommandExecution):
        data = execution.to_dict()
    elif isinstance(execution, Mapping):
        data = dict(execution)
    else:
        raise TypeError("ops command runner must return OpsCommandExecution or mapping")
    return {"status": "ok" if int(data.get("exit_code", -1)) == 0 else "error", "guard": guard.to_dict(), **data}


def load_context(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ops guard context must be a JSON object")
    return payload
