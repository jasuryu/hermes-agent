"""Deterministic Agent Runtime → Obsidian runbook mirror helpers.

This module is intentionally not a worker scheduler and not an approval writer.
It renders Runtime SQLite state into a documentation-only Markdown note that can
be reviewed in Obsidian/RAG without turning Obsidian into an execution queue.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from hermes_constants import get_hermes_home

from . import db

RUNBOOK_RELATIVE_DIR = Path("01 Hermes") / "Agent Runtime" / "Runs"
_KEYED_SECRET_PATTERN = re.compile(
    r"(?i)\b([A-Za-z0-9_.-]*(?:api[_-]?key|secret[_-]?key|secret|access[_-]?token|auth[_-]?token|github[_-]?token|token|password|passwd))\s*[:=]\s*['\"]?[^\s'\",;`]{3,}"
)
_STANDALONE_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bxox(?:b|a|p|r|s)-[A-Za-z0-9-]{10,}\b"),
]
_SECRET_NAME_PATTERN = re.compile(
    r"(?i)\b[A-Za-z0-9][A-Za-z0-9_.-]*(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|github[_-]?token|[_-]token|[_-]secret|[_-]password|[_-]passwd)\b(?!=\[REDACTED\])"
)


def _utc(ts: int | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def _sanitize_text(value: Any, *, limit: int = 500) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\0", "")
    text = _KEYED_SECRET_PATTERN.sub(lambda match: "[REDACTED_KEY]=[REDACTED]", text)
    for pattern in _STANDALONE_SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = _SECRET_NAME_PATTERN.sub("[REDACTED_KEY]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _safe_filename_part(value: str, *, limit: int = 96) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x1f\x7f/\\]+", " ", text)
    text = re.sub(r"[<>:\"|?*]+", "-", text)
    text = text.replace("..", " ")
    text = re.sub(r"\s+", " ", text).strip(" ._-")
    if len(text) > limit:
        text = text[:limit].rstrip(" ._-")
    return text or "runtime-run"


def _yaml_string(value: Any) -> str:
    return json.dumps(_sanitize_text(value, limit=300), ensure_ascii=False)


def _code_text(value: Any, *, limit: int = 160) -> str:
    return _sanitize_text(value, limit=limit).replace("`", "ʼ")


def _load_json_list(text: str | None) -> list[Any]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _list_decisions(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runtime_decisions WHERE run_id=? ORDER BY created_at, id",
        (run_id,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "run_id": row["run_id"],
            "job_id": row["job_id"],
            "kind": row["kind"],
            "rationale": row["rationale"],
            "linked_findings": _load_json_list(row["linked_findings_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _list_findings(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runtime_findings WHERE run_id=? ORDER BY created_at, id",
        (run_id,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "run_id": row["run_id"],
            "job_id": row["job_id"],
            "severity": row["severity"],
            "category": row["category"],
            "summary": row["summary"],
            "evidence_ref": row["evidence_ref"],
            "recommendation": row["recommendation"],
            "status": row["status"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }
        for row in rows
    ]


def default_obsidian_vault_path() -> Path:
    configured = os.environ.get("OBSIDIAN_VAULT_PATH")
    if configured:
        return Path(configured).expanduser()
    return get_hermes_home() / "obsidian"


def runbook_relative_path(run: Any) -> Path:
    base = " - ".join(part for part in (run.public_ref, run.title) if part)
    safe = _safe_filename_part(_sanitize_text(base or run.id, limit=180))
    safe_id = _safe_filename_part(_sanitize_text(run.id, limit=120), limit=120)
    return RUNBOOK_RELATIVE_DIR / f"{safe} - {safe_id}.md"


def _reject_existing_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("runtime runbook path contains symlink component")


def _resolve_runbook_path(vault: Path, relative: Path) -> tuple[Path, Path, Path]:
    try:
        vault_resolved = vault.resolve()
    except RuntimeError as exc:
        raise ValueError("runtime runbook path resolution failed") from exc
    raw_path = vault_resolved / relative
    _reject_existing_symlink_components(vault_resolved, relative)
    try:
        path = raw_path.resolve(strict=False)
        runbook_root = (vault_resolved / RUNBOOK_RELATIVE_DIR).resolve(strict=False)
        path.relative_to(vault_resolved)
        path.relative_to(runbook_root)
    except RuntimeError as exc:
        raise ValueError("runtime runbook path resolution failed") from exc
    except ValueError as exc:
        raise ValueError("runtime runbook path escaped Obsidian vault") from exc
    return vault_resolved, runbook_root, path


def render_runbook_markdown(conn: sqlite3.Connection, run_id: str, *, generated_at: int | None = None) -> str:
    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"runtime run not found: {run_id}")

    jobs = db.list_jobs(conn, run_id)
    decisions = _list_decisions(conn, run_id)
    findings = _list_findings(conn, run_id)
    events = db.list_events(conn, run_id=run_id, limit=50)
    job_counts = Counter(job.status for job in jobs)
    open_findings = sum(1 for finding in findings if finding["status"] == "open")
    generated = _utc(generated_at if generated_at is not None else run.updated_at)

    lines: list[str] = [
        "---",
        "type: hermes-agent-runtime-runbook",
        f"runtime_run_id: {_yaml_string(run.id)}",
        f"title: {_yaml_string(run.title)}",
        f"public_ref: {_yaml_string(run.public_ref)}",
        f"status: {run.status}",
        f"risk_level: {run.risk_level}",
        f"generated_at: {generated}",
        "source: Hermes Agent Runtime SQLite",
        "mirror_only: true",
        "---",
        "",
        f"# {_sanitize_text(run.title or run.id, limit=160)}",
        "",
        "> Deterministic Agent Runtime documentation mirror only. This note is not an execution queue, not an approval source, and not a worker scheduler.",
        "",
        "## Current state",
        f"- Runtime run: `{_code_text(run.id)}`",
        f"- Public ref: `{_sanitize_text(run.public_ref)}`" if run.public_ref else "- Public ref: none",
        f"- Status: `{run.status}`",
        f"- Risk level: `{run.risk_level}`",
        f"- Objective: {_sanitize_text(run.objective) or 'none'}",
        f"- Owner source: {_sanitize_text(run.owner_source) or 'none'}",
        f"- Created: {_utc(run.created_at)}",
        f"- Updated: {_utc(run.updated_at)}",
        f"- Jobs: {len(jobs)} total" + (f" ({', '.join(f'{status}: {count}' for status, count in sorted(job_counts.items()))})" if jobs else ""),
        f"- Decisions: {len(decisions)}",
        f"- Findings: {len(findings)} total; {open_findings} open",
        "",
        "## Jobs",
    ]

    if not jobs:
        lines.append("- None.")
    for job in jobs:
        lines.append(f"- `{job.id}` — **{job.status}** — `{job.role}` — {_sanitize_text(job.title, limit=220)}")
        lines.append(f"  - Attempts: {job.attempt_count}/{job.max_attempts}")
        lines.append(f"  - Workspace: `{job.workspace_kind}`" + (f" `{_sanitize_text(job.workspace_path, limit=160)}`" if job.workspace_path else ""))
        if job.result_summary:
            lines.append(f"  - Result: {_sanitize_text(job.result_summary)}")
        else:
            lines.append("  - Result: none yet")
        if job.lease_owner:
            lines.append(f"  - Lease owner: `{_sanitize_text(job.lease_owner, limit=120)}` until {_utc(job.lease_expires_at)}")
        lines.append("  - Body: omitted from Obsidian mirror to avoid leaking prompts/context/secrets.")

    lines.extend(["", "## Decisions"])
    if not decisions:
        lines.append("- None.")
    for decision in decisions:
        job_ref = f" — job `{decision['job_id']}`" if decision.get("job_id") else ""
        linked = decision.get("linked_findings") or []
        lines.append(f"- `{decision['id']}` — **{_sanitize_text(decision['kind'], limit=80)}**{job_ref} — {_utc(decision['created_at'])}")
        if decision.get("rationale"):
            lines.append(f"  - Rationale: {_sanitize_text(decision['rationale'])}")
        if linked:
            lines.append("  - Linked findings: " + ", ".join(f"`{_sanitize_text(item, limit=80)}`" for item in linked))

    lines.extend(["", "## Findings"])
    if not findings:
        lines.append("- None.")
    for finding in findings:
        job_ref = f" — job `{finding['job_id']}`" if finding.get("job_id") else ""
        lines.append(
            f"- `{finding['id']}` — **{_sanitize_text(finding['severity'], limit=40)}** "
            f"`{_sanitize_text(finding['category'], limit=80)}` — {finding['status']}{job_ref} — {_utc(finding['created_at'])}"
        )
        lines.append(f"  - Summary: {_sanitize_text(finding['summary'])}")
        if finding.get("evidence_ref"):
            lines.append(f"  - Evidence: {_sanitize_text(finding['evidence_ref'])}")
        if finding.get("recommendation"):
            lines.append(f"  - Recommendation: {_sanitize_text(finding['recommendation'])}")

    lines.extend(["", "## Recent events"])
    if not events:
        lines.append("- None.")
    for event in events[-50:]:
        job_ref = f" — job `{event.job_id}`" if event.job_id else ""
        lines.append(f"- `{event.id}` — `{_sanitize_text(event.kind, limit=80)}`{job_ref} — {_utc(event.created_at)}")

    lines.extend([
        "",
        "## Safety boundary",
        "- Runtime SQLite remains the machine execution truth.",
        "- Obsidian/RAG copies are documentation mirrors only and must not be polled as work queues.",
        "- Approval packets must come from a trusted operator channel, not from this note.",
        "",
    ])
    return "\n".join(lines)


def sync_runbook_to_obsidian(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    vault_path: str | Path | None = None,
    write: bool = False,
    generated_at: int | None = None,
) -> dict[str, Any]:
    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"runtime run not found: {run_id}")
    vault = Path(vault_path).expanduser() if vault_path else default_obsidian_vault_path()
    relative = runbook_relative_path(run)
    _vault_resolved, _runbook_root, path = _resolve_runbook_path(vault, relative)
    markdown = render_runbook_markdown(conn, run_id, generated_at=generated_at)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        _vault_resolved, _runbook_root, path = _resolve_runbook_path(vault, relative)
        path.write_text(markdown, encoding="utf-8")
    return {
        "success": True,
        "run_id": run_id,
        "written": bool(write),
        "path": str(path),
        "relative_path": relative.as_posix(),
        "markdown": markdown,
    }
