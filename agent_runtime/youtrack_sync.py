"""Deterministic Agent Runtime → YouTrack public mirror helpers.

This module mirrors selected Runtime SQLite state into YouTrack comments and,
optionally, a caller-requested Stage transition. YouTrack is a human-visible
status surface only: it is not an execution queue, approval source, scheduler,
or worker command channel.
"""

from __future__ import annotations

from collections import Counter
import re
import subprocess
from typing import Any, Callable, Sequence

from . import db
from .scribe_sync import _list_decisions, _list_findings, _sanitize_text, _utc

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

_ISSUE_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _validate_issue_id(value: str | None) -> str:
    issue_id = str(value or "").strip()
    if not _ISSUE_RE.fullmatch(issue_id):
        raise ValueError("valid YouTrack issue id is required, e.g. HP-88")
    return issue_id


def _validate_stage(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    raw_stage = str(value)
    if _CONTROL_CHARS_RE.search(raw_stage):
        raise ValueError("valid YouTrack Stage value is required")
    stage = raw_stage.strip()
    if not stage or len(stage) > 80 or _CONTROL_CHARS_RE.search(stage):
        raise ValueError("valid YouTrack Stage value is required")
    if "[REDACTED" in _sanitize_text(stage, limit=120):
        raise ValueError("valid YouTrack Stage value is required")
    return stage


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, check=False, timeout=60)


def render_youtrack_comment(
    conn,
    run_id: str,
    *,
    issue_id: str | None = None,
    generated_at: int | None = None,
    max_chars: int = 9000,
) -> str:
    """Render a safe public YouTrack status comment for one Runtime run.

    The comment deliberately excludes raw job bodies, prompts, private context,
    approval packets, and command payloads. It is safe to place on a public-ish
    tracker issue, but remains a mirror only; Runtime SQLite stays the machine
    source of truth.
    """

    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"runtime run not found: {run_id}")

    jobs = db.list_jobs(conn, run_id)
    decisions = _list_decisions(conn, run_id)
    findings = _list_findings(conn, run_id)
    open_findings = [finding for finding in findings if finding["status"] == "open"]
    job_counts = Counter(job.status for job in jobs)
    generated = _utc(generated_at if generated_at is not None else run.updated_at)
    rendered_issue = issue_id or run.public_ref

    lines: list[str] = [
        "Hermes Agent Runtime public mirror",
        "",
        "Public status mirror only: this YouTrack comment is not an execution queue, not an approval source, and not a worker scheduler.",
        "Runtime SQLite remains the machine execution truth; YouTrack is the human-visible audit/status surface.",
        "",
        "Current state:",
        f"- Runtime run: `{_sanitize_text(run.id, limit=120)}`",
        f"- YouTrack issue: `{_sanitize_text(rendered_issue, limit=80)}`" if rendered_issue else "- YouTrack issue: none",
        f"- Title: {_sanitize_text(run.title, limit=180)}",
        f"- Status: `{_sanitize_text(run.status, limit=40)}`",
        f"- Risk level: `{_sanitize_text(run.risk_level, limit=60)}`",
        "- Objective: omitted from YouTrack mirror; free-form objective/result/rationale/finding text omitted for public safety.",
        f"- Updated: {_utc(run.updated_at)}",
        f"- Mirror generated: {generated}",
        "",
        "Job summary:",
        f"- Jobs: {len(jobs)} total" + (f" ({', '.join(f'{_sanitize_text(status, limit=40)}: {count}' for status, count in sorted(job_counts.items()))})" if jobs else ""),
    ]

    for job in jobs[:20]:
        lines.append(f"- `{_sanitize_text(job.id, limit=80)}` — **{_sanitize_text(job.status, limit=40)}** — `{_sanitize_text(job.role, limit=60)}`")
        lines.append("  - Title/result/body/prompt/context: omitted from YouTrack mirror.")
    if len(jobs) > 20:
        lines.append(f"- … {len(jobs) - 20} more job(s) omitted from this public mirror.")

    lines.extend(["", "Decisions:"])
    if not decisions:
        lines.append("- None.")
    for decision in decisions[:20]:
        lines.append(f"- `{_sanitize_text(decision['id'], limit=80)}` — **{_sanitize_text(decision['kind'], limit=80)}** — rationale omitted")
    if len(decisions) > 20:
        lines.append(f"- … {len(decisions) - 20} more decision(s) omitted.")

    lines.extend(["", "Findings:"])
    if not findings:
        lines.append("- None.")
    else:
        lines.append(f"- Findings: {len(findings)} total; {len(open_findings)} open")
    for finding in findings[:20]:
        lines.append(
            f"- `{_sanitize_text(finding['id'], limit=80)}` — **{_sanitize_text(finding['severity'], limit=40)}** "
            f"`{_sanitize_text(finding['category'], limit=80)}` — {_sanitize_text(finding['status'], limit=40)} — summary/recommendation omitted"
        )

    lines.extend([
        "",
        "Safety boundary:",
        "- Do not treat this YouTrack issue/comment as a Runtime command channel.",
        "- Approval packets must come from a trusted operator channel, not from YouTrack comments/status.",
        "- Runtime workers must read Runtime SQLite leases/approvals only, never YouTrack text.",
    ])

    comment = "\n".join(lines).strip() + "\n"
    if len(comment) > max_chars:
        suffix = "\n… truncated by Hermes Runtime YouTrack mirror; see Runtime DB/Obsidian runbook for full details.\n"
        comment = comment[: max(0, max_chars - len(suffix))].rstrip() + suffix
    return comment


def sync_run_to_youtrack(
    conn,
    run_id: str,
    *,
    issue_id: str | None = None,
    stage: str | None = None,
    write: bool = False,
    runner: Runner | None = None,
    ytctl: str = "ytctl",
    generated_at: int | None = None,
) -> dict[str, Any]:
    """Render or publish a Runtime status mirror to YouTrack.

    Dry-run is the default. With ``write=True`` this executes only explicit
    ``ytctl`` comment/stage commands using argv lists (never a shell string).
    It does not mutate Runtime SQLite and does not read YouTrack as input.
    """

    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"runtime run not found: {run_id}")
    issue = _validate_issue_id(issue_id or run.public_ref)
    target_stage = _validate_stage(stage)
    comment = render_youtrack_comment(conn, run_id, issue_id=issue, generated_at=generated_at)

    commands: list[list[str]] = [[str(ytctl), "comment", issue, comment]]
    operations = ["comment"]
    if target_stage:
        commands.append([str(ytctl), "update", issue, "stage", target_stage])
        operations.append("stage")

    results: list[dict[str, Any]] = []
    if write:
        call = runner or _default_runner
        for argv in commands:
            try:
                completed = call(argv)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"ytctl {argv[1]} timed out") from exc
            returncode = int(getattr(completed, "returncode", 0))
            stdout = str(getattr(completed, "stdout", "") or "")
            stderr = str(getattr(completed, "stderr", "") or "")
            results.append({"argv": list(argv[:3]) + (["<comment>"] if argv[1] == "comment" else argv[3:]), "returncode": returncode})
            if returncode != 0:
                detail = (stderr or stdout or f"ytctl {argv[1]} failed").strip()
                raise RuntimeError(detail)

    return {
        "success": True,
        "run_id": run_id,
        "issue_id": issue,
        "written": bool(write),
        "operations": operations,
        "stage": target_stage or "",
        "commands": commands,
        "comment": comment,
        "results": results,
        "mirror_only": True,
    }
