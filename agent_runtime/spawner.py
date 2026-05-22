"""Worker process invocation helpers for Agent Runtime."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

from .worker_broker import WorkerSandbox

_SAFE_ENV_KEYS = {
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "TERM",
    "TZ",
    "USER",
}
_SECRET_KEY_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


@dataclass(frozen=True)
class WorkerInvocation:
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    context_path: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["argv"] = list(self.argv)
        data["cwd"] = str(self.cwd)
        data["context_path"] = str(self.context_path)
        return data


def _is_secret_like_env_key(key: str) -> bool:
    upper = key.upper()
    return any(fragment in upper for fragment in _SECRET_KEY_FRAGMENTS)


def _is_allowed_extra_env_key(key: str) -> bool:
    if _is_secret_like_env_key(key):
        return False
    if key.startswith("HERMES_AGENT_RUNTIME_APPROVAL"):
        return False
    if key.startswith("HERMES_AGENT_RUNTIME_") and key != "HERMES_AGENT_RUNTIME_TRACE":
        return False
    if key in {"HOME", "PATH", "PYTHONPATH", "VIRTUAL_ENV", "HERMES_HOME"}:
        return False
    return key in _SAFE_ENV_KEYS or key in {"HERMES_AGENT_RUNTIME_TRACE"}


def _base_worker_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_ENV_KEYS and not _is_secret_like_env_key(key)
    }


def python_executable() -> str:
    return sys.executable


def _worker_code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _worker_bootstrap(job_id: str) -> str:
    code_root = str(_worker_code_root())
    return "".join(
        (
            "import runpy,sys;",
            f"sys.path.insert(0, {code_root!r});",
            f"sys.argv = ['agent_runtime.worker_main', '--job', {job_id!r}];",
            "runpy.run_module('agent_runtime.worker_main', run_name='__main__')",
        )
    )


def _is_private_dir(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    st = path.lstat()
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return False
    return stat.S_ISDIR(st.st_mode) and not (stat.S_IMODE(st.st_mode) & 0o077)


def _validate_sandbox(sandbox: WorkerSandbox, context: Path) -> None:
    root = sandbox.root.resolve()
    if not _is_private_dir(sandbox.root):
        raise ValueError("worker sandbox root must be a private real directory")
    for label, path in {
        "home": sandbox.home,
        "workdir": sandbox.workdir,
        "tmp": sandbox.tmp,
        "xdg_config_home": sandbox.xdg_config_home,
        "xdg_cache_home": sandbox.xdg_cache_home,
    }.items():
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"worker sandbox {label} must live inside sandbox root") from exc
        if not _is_private_dir(path):
            raise ValueError(f"worker sandbox {label} must be a private real directory")
    try:
        context.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("context_path must live inside the worker sandbox") from exc
    if context.is_symlink() or not context.is_file():
        raise ValueError("context_path must be a regular file inside the worker sandbox")
    if stat.S_IMODE(context.lstat().st_mode) & 0o077:
        raise ValueError("context_path must not be group/world accessible")


def build_worker_invocation(
    *,
    job_id: str,
    run_id: str,
    role: str,
    attempt_id: str = "",
    lease_owner: str = "",
    context_path: str | Path | None = None,
    sandbox: WorkerSandbox | None = None,
    extra_env: Mapping[str, str] | None = None,
    enable_execution: bool = False,
) -> WorkerInvocation:
    if not attempt_id or not lease_owner:
        raise ValueError("worker invocation requires attempt_id and lease_owner")
    if context_path is None:
        raise ValueError("worker invocation requires context_path from trusted broker")
    if sandbox is None:
        raise ValueError("worker invocation requires sandbox from trusted broker")
    context = Path(context_path)
    _validate_sandbox(sandbox, context)

    env = _base_worker_env()
    if extra_env:
        env.update({
            str(k): str(v)
            for k, v in extra_env.items()
            if _is_allowed_extra_env_key(str(k))
        })

    # Fixed launch identity and scratch filesystem pointers are trusted-parent
    # controlled and cannot be overridden by caller-provided extra_env.
    env["HERMES_AGENT_RUNTIME_ATTEMPT_ID"] = attempt_id
    env["HERMES_AGENT_RUNTIME_LEASE_OWNER"] = lease_owner
    env["HERMES_AGENT_RUNTIME_CONTEXT"] = str(context)
    env["HOME"] = str(sandbox.home)
    env["TMPDIR"] = str(sandbox.tmp)
    env["XDG_CONFIG_HOME"] = str(sandbox.xdg_config_home)
    env["XDG_CACHE_HOME"] = str(sandbox.xdg_cache_home)
    env["PYTHONNOUSERSITE"] = "1"
    if enable_execution:
        env["HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION"] = "1"

    argv = (python_executable(), "-c", _worker_bootstrap(job_id))
    return WorkerInvocation(argv=argv, env=env, cwd=sandbox.workdir, context_path=context)
