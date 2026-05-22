"""Worker process isolation preflight for Agent Runtime.

Scratch HOME/TMPDIR and sanitized env reduce accidental leakage, but they are not
a security boundary for code-capable workers running as the same Unix user.  This
module makes that explicit: real spawning requires a configured OS/process
isolation backend plus a backend-specific launch profile constructed and enforced
by the trusted parent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import shutil
from typing import Callable

from .worker_broker import WorkerSandbox


@dataclass(frozen=True)
class IsolationAssessment:
    backend: str
    available: bool
    allows_spawn: bool
    executable: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerLaunchPlan:
    backend: str
    executable: str
    argv: tuple[str, ...]
    worker_argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    allows_spawn: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["argv"] = list(self.argv)
        data["worker_argv"] = list(self.worker_argv)
        data["cwd"] = str(self.cwd)
        return data


_BACKEND_BINARIES = {
    "bubblewrap": "bwrap",
    "bwrap": "bwrap",
    "firejail": "firejail",
}
_FORBIDDEN_LAUNCH_ENV = {"HERMES_HOME", "PATH", "PYTHONPATH", "VIRTUAL_ENV"}
_SECRET_KEY_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
_ALLOWED_RUNTIME_ENV = {
    "HERMES_AGENT_RUNTIME_ATTEMPT_ID",
    "HERMES_AGENT_RUNTIME_LEASE_OWNER",
    "HERMES_AGENT_RUNTIME_CONTEXT",
    "HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION",
    "HERMES_AGENT_RUNTIME_TRACE",
}
_ALLOWED_LAUNCH_ENV = {
    "HOME",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "PYTHONNOUSERSITE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "TERM",
    "TZ",
    "USER",
    *_ALLOWED_RUNTIME_ENV,
}


def assess_worker_isolation(
    *,
    backend: str = "disabled",
    executable_resolver: Callable[[str], str | None] | None = None,
) -> IsolationAssessment:
    """Return whether real worker spawning may be enabled for this backend."""
    normalized = (backend or "disabled").strip().lower()
    resolver = executable_resolver or shutil.which
    if normalized in {"", "disabled", "none", "off"}:
        return IsolationAssessment(
            backend="disabled",
            available=False,
            allows_spawn=False,
            reason="worker process isolation backend is disabled; spawn must remain fail-closed",
        )
    if normalized in {"env_only", "scratch_env", "scratch-home"}:
        return IsolationAssessment(
            backend=normalized,
            available=False,
            allows_spawn=False,
            reason="scratch HOME/TMPDIR/env-only isolation is not a security boundary for same-user workers",
        )
    binary = _BACKEND_BINARIES.get(normalized)
    if not binary:
        return IsolationAssessment(
            backend=normalized,
            available=False,
            allows_spawn=False,
            reason=f"unknown worker isolation backend: {backend}",
        )
    executable = resolver(binary)
    if not executable:
        return IsolationAssessment(
            backend=normalized,
            available=False,
            allows_spawn=False,
            reason=f"worker isolation backend {normalized} requires executable {binary!r}",
        )
    allows_spawn = normalized in {"bubblewrap", "bwrap"}
    reason = (
        "reviewed launch policy is available for bubblewrap worker spawn"
        if allows_spawn else
        "worker isolation executable is available, but no reviewed launch policy exists for this backend"
    )
    return IsolationAssessment(
        backend="bubblewrap" if normalized == "bwrap" else normalized,
        available=True,
        allows_spawn=allows_spawn,
        executable=str(executable),
        reason=reason,
    )


def _relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_launch_paths(*, sandbox: WorkerSandbox, context_path: Path, cwd: Path) -> None:
    root = sandbox.root.resolve()
    if cwd.resolve() != sandbox.workdir.resolve():
        raise ValueError("worker launch cwd must be exactly sandbox.workdir")
    if not _relative_to(context_path, sandbox.root):
        raise ValueError("worker launch context_path must live inside sandbox root")
    if context_path.is_symlink() or not context_path.is_file():
        raise ValueError("worker launch context_path must be a regular file")
    for path in (sandbox.home, sandbox.workdir, sandbox.tmp, sandbox.xdg_config_home, sandbox.xdg_cache_home):
        if not _relative_to(path, sandbox.root):
            raise ValueError("worker sandbox paths must live inside sandbox root")
    if root == Path("/").resolve():
        raise ValueError("worker sandbox root cannot be filesystem root")


def _expect_env_path(worker_env: dict[str, str], key: str, expected: Path) -> None:
    value = worker_env.get(key)
    if value is None or Path(value).resolve() != expected.resolve():
        raise ValueError(f"worker launch env {key} must equal trusted sandbox path")


def _validate_launch_env(worker_env: dict[str, str], *, sandbox: WorkerSandbox, context_path: Path) -> None:
    for key in worker_env:
        upper = key.upper()
        if key not in _ALLOWED_LAUNCH_ENV:
            raise ValueError(f"worker launch env contains non-allowlisted key: {key}")
        if key in _FORBIDDEN_LAUNCH_ENV or any(fragment in upper for fragment in _SECRET_KEY_FRAGMENTS):
            raise ValueError(f"worker launch env contains forbidden key: {key}")
        if key.startswith("HERMES_AGENT_RUNTIME_") and key not in _ALLOWED_RUNTIME_ENV:
            raise ValueError(f"worker launch env contains forbidden runtime key: {key}")
    _expect_env_path(worker_env, "HOME", sandbox.home)
    _expect_env_path(worker_env, "TMPDIR", sandbox.tmp)
    _expect_env_path(worker_env, "XDG_CONFIG_HOME", sandbox.xdg_config_home)
    _expect_env_path(worker_env, "XDG_CACHE_HOME", sandbox.xdg_cache_home)
    _expect_env_path(worker_env, "HERMES_AGENT_RUNTIME_CONTEXT", context_path)
    for key in ("HERMES_AGENT_RUNTIME_ATTEMPT_ID", "HERMES_AGENT_RUNTIME_LEASE_OWNER"):
        if not worker_env.get(key):
            raise ValueError(f"worker launch env {key} is required")


def _setenv_args(worker_env: dict[str, str]) -> tuple[str, ...]:
    args: list[str] = []
    for key in sorted(worker_env):
        args.extend(("--setenv", str(key), str(worker_env[key])))
    return tuple(args)


def _system_ro_bind_args() -> tuple[str, ...]:
    args: list[str] = []
    for raw in ("/usr", "/bin", "/lib", "/lib64"):
        path = Path(raw)
        if path.exists():
            args.extend(("--ro-bind", raw, raw))
    return tuple(args)


def _covered_by_system_bind(path: Path) -> bool:
    for raw in ("/usr", "/bin", "/lib", "/lib64"):
        root = Path(raw)
        if not root.exists():
            continue
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _python_runtime_ro_bind_args(worker_argv: tuple[str, ...]) -> tuple[str, ...]:
    """Bind the real Python runtime for venv executables outside /usr.

    In this install the venv executable lives under /usr/local but is a symlink
    to a uv-managed interpreter under /root/.local.  Binding /usr is not enough
    for execve to resolve that symlink inside bubblewrap.
    """
    if not worker_argv:
        return ()
    executable = Path(worker_argv[0])
    if not executable.is_absolute() or not executable.exists():
        return ()
    roots: list[Path] = []
    if executable.is_symlink():
        raw_target = Path(os.readlink(executable))
        if raw_target.is_absolute() and len(raw_target.parents) >= 3:
            # Bind the Python-install parent so absolute venv symlinks and their
            # own version aliases both resolve inside bubblewrap.
            roots.append(raw_target.parents[2])
    real_executable = executable.resolve()
    if real_executable.exists():
        roots.append(real_executable.parent.parent)
    if executable.parent.name == "bin":
        roots.append(executable.parent.parent)

    args: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists() or _covered_by_system_bind(root):
            continue
        raw = str(root)
        if raw in seen:
            continue
        seen.add(raw)
        args.extend(("--ro-bind", raw, raw))
    return tuple(args)


def _scratch_bind_args(sandbox: WorkerSandbox) -> tuple[str, ...]:
    args: list[str] = ["--dir", str(sandbox.root)]
    for path in (sandbox.home, sandbox.workdir, sandbox.tmp, sandbox.xdg_config_home, sandbox.xdg_cache_home):
        raw = str(path)
        args.extend(("--bind", raw, raw))
    return tuple(args)


def _bubblewrap_argv(
    *,
    executable: str,
    worker_argv: tuple[str, ...],
    worker_env: dict[str, str],
    sandbox: WorkerSandbox,
    cwd: Path,
    context_path: Path,
) -> tuple[str, ...]:
    context = str(context_path)
    return (
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--clearenv",
        "--dev",
        "/dev",
        *_system_ro_bind_args(),
        *_python_runtime_ro_bind_args(worker_argv),
        "--tmpfs",
        "/tmp",
        *_scratch_bind_args(sandbox),
        "--ro-bind",
        context,
        context,
        *_setenv_args(worker_env),
        "--chdir",
        str(cwd),
        "--",
        *worker_argv,
    )


def build_launch_plan(
    *,
    backend: str,
    worker_argv: list[str] | tuple[str, ...],
    worker_env: dict[str, str],
    cwd: str | Path,
    sandbox: WorkerSandbox,
    context_path: str | Path,
    executable_resolver: Callable[[str], str | None] | None = None,
) -> WorkerLaunchPlan:
    """Build a reviewed backend-specific worker launch plan."""
    normalized = (backend or "disabled").strip().lower()
    if normalized not in {"bubblewrap", "bwrap"}:
        raise ValueError(f"unsupported worker isolation launch backend: {backend}")
    assessment = assess_worker_isolation(backend=normalized, executable_resolver=executable_resolver)
    if not assessment.executable:
        raise ValueError(assessment.reason)
    context = Path(context_path)
    launch_cwd = Path(cwd)
    _validate_launch_paths(sandbox=sandbox, context_path=context, cwd=launch_cwd)
    worker_tuple = tuple(str(part) for part in worker_argv)
    launch_env = {str(k): str(v) for k, v in worker_env.items()}
    _validate_launch_env(launch_env, sandbox=sandbox, context_path=context)
    argv = _bubblewrap_argv(
        executable=assessment.executable,
        worker_argv=worker_tuple,
        worker_env=launch_env,
        sandbox=sandbox,
        cwd=launch_cwd,
        context_path=context,
    )
    return WorkerLaunchPlan(
        backend="bubblewrap",
        executable=assessment.executable,
        argv=argv,
        worker_argv=worker_tuple,
        env=launch_env,
        cwd=launch_cwd,
        allows_spawn=True,
        reason="reviewed bubblewrap launch policy is constructed and spawn may proceed behind scheduler enable_spawn gate",
    )
