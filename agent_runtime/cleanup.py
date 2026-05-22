"""Cleanup policy for Agent Runtime worker sandbox directories."""

from __future__ import annotations

from pathlib import Path
import shutil
import time
from typing import Any

from hermes_constants import get_hermes_home

DEFAULT_PARENT = Path("/tmp")
DEFAULT_PREFIX = "hermes-agent-runtime-workers-"


def _now(now: int | None = None) -> int:
    return int(time.time() if now is None else now)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts:
        if part in {"", path.anchor}:
            continue
        current = current / part
        if current.is_symlink():
            return True
    return False


def _validate_parent(parent: Path) -> Path:
    raw = Path(parent).expanduser()
    try:
        resolved = raw.resolve()
    except RuntimeError as exc:
        raise ValueError("unsafe worker sandbox cleanup parent: resolution failed") from exc
    if resolved == Path("/").resolve():
        raise ValueError("unsafe worker sandbox cleanup parent: filesystem root is not allowed")
    if _has_symlink_component(raw):
        raise ValueError("unsafe worker sandbox cleanup parent: symlink components are not allowed")
    if not resolved.is_dir():
        raise ValueError("unsafe worker sandbox cleanup parent: must be a real directory")
    hermes_home = get_hermes_home().resolve()
    if _is_relative_to(resolved, hermes_home) or _is_relative_to(hermes_home, resolved):
        raise ValueError("unsafe worker sandbox cleanup parent: must not overlap HERMES_HOME")
    return resolved


def _candidate_payload(path: Path, *, parent: Path, now: int) -> dict[str, Any] | None:
    if not path.name.startswith(DEFAULT_PREFIX):
        return None
    if path.is_symlink() or not path.is_dir():
        return {"path": str(path), "skip_reason": "not a real directory"}
    try:
        resolved = path.resolve()
        resolved.relative_to(parent)
    except (RuntimeError, ValueError):
        return {"path": str(path), "skip_reason": "path escapes cleanup parent"}
    st = path.lstat()
    age_seconds = max(0, now - int(st.st_mtime))
    return {
        "path": str(path),
        "age_seconds": age_seconds,
        "modified_at": int(st.st_mtime),
    }


def cleanup_worker_sandboxes(
    *,
    parent: str | Path | None = None,
    max_age_seconds: int = 86400,
    now: int | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or execute deletion of stale worker sandbox directories.

    Dry-run is the default.  Only directories in a trusted parent whose names use
    the Runtime sandbox prefix are candidates; symlinks and escaping paths are
    skipped fail-closed.
    """
    ts = _now(now)
    cleanup_parent = _validate_parent(Path(parent) if parent is not None else DEFAULT_PARENT)
    max_age = max(0, int(max_age_seconds))
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for child in sorted(cleanup_parent.iterdir(), key=lambda p: p.name):
        payload = _candidate_payload(child, parent=cleanup_parent, now=ts)
        if payload is None:
            continue
        if payload.get("skip_reason"):
            skipped.append(payload)
            continue
        if int(payload["age_seconds"]) >= max_age:
            planned.append(payload)
    removed: list[str] = []
    if execute:
        for item in planned:
            target = Path(str(item["path"]))
            if target.is_symlink() or not target.is_dir():
                skipped.append({"path": str(target), "skip_reason": "candidate changed before removal"})
                continue
            try:
                target.resolve().relative_to(cleanup_parent)
            except (RuntimeError, ValueError):
                skipped.append({"path": str(target), "skip_reason": "candidate escaped before removal"})
                continue
            shutil.rmtree(target)
            removed.append(str(target))
    return {
        "success": True,
        "executed": bool(execute),
        "parent": str(cleanup_parent),
        "prefix": DEFAULT_PREFIX,
        "max_age_seconds": max_age,
        "generated_at": ts,
        "candidates": len(planned),
        "planned": planned,
        "removed": removed,
        "skipped": skipped,
    }
