"""Read-only Agent Runtime mirror snapshots for dashboards/Kanban-style views.

Runtime SQLite remains the execution source of truth.  This module only derives
compact display DTOs and intentionally omits raw job bodies, approval command
payloads, and worker context.
"""

from __future__ import annotations

from collections import Counter
import re
import sqlite3
import time
from typing import Any

from . import db
from .models import RuntimeJob, RuntimeRun

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|private[_-]?key|session[_-]?cookie|database[_-]?url|db[_-]?(?:uri|url))[A-Za-z0-9_-]*)\b\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_PROVIDER_TOKEN_RE = re.compile(r"\b(sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9_]{12,}|hf_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b")


def _now(now: int | None = None) -> int:
    return int(time.time() if now is None else now)


def _safe_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _PROVIDER_TOKEN_RE.sub("[REDACTED]", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _job_lane(status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized in {"planned", "blocked"}:
        return "blocked"
    if normalized == "ready":
        return "ready"
    if normalized in {"leased", "running"}:
        return "active"
    if normalized == "succeeded":
        return "done"
    if normalized in {"failed", "cancelled", "timed_out"}:
        return "attention"
    return "unknown"


def _run_lane(status: str, *, has_attention: bool) -> str:
    if has_attention:
        return "attention"
    normalized = (status or "").strip().lower()
    if normalized in {"planning", "running"}:
        return "active"
    if normalized == "attention":
        return "attention"
    if normalized == "done":
        return "done"
    if normalized in {"failed", "cancelled"}:
        return "attention"
    return "unknown"


def _jobs_by_status(jobs: list[RuntimeJob]) -> dict[str, int]:
    return dict(sorted(Counter(job.status for job in jobs).items()))


def _progress(jobs: list[RuntimeJob]) -> dict[str, Any]:
    total = len(jobs)
    done = sum(1 for job in jobs if job.status == "succeeded")
    failed = sum(1 for job in jobs if job.status in {"failed", "cancelled"})
    percent = round((done / total) * 100, 1) if total else 0.0
    return {
        "source": "runtime_job_status_counts",
        "completed": done,
        "failed": failed,
        "total": total,
        "percent": percent,
    }


def _job_alerts(job: RuntimeJob, *, now: int, stale_lease_seconds: int) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if job.status in {"leased", "running"} and job.lease_expires_at is not None and job.lease_expires_at <= now:
        alerts.append({"severity": "critical", "code": "lease_expired", "message": "Runtime job lease has expired."})
    if job.status in {"leased", "running"} and job.heartbeat_at is not None and now - int(job.heartbeat_at) > stale_lease_seconds:
        alerts.append({"severity": "warning", "code": "heartbeat_stale", "message": "Runtime job heartbeat is stale."})
    if job.status == "failed":
        alerts.append({"severity": "warning", "code": "job_failed", "message": "Runtime job failed or exhausted attempts."})
    return alerts


def _run_card(run: RuntimeRun, jobs: list[RuntimeJob], *, now: int, stale_lease_seconds: int) -> dict[str, Any]:
    alerts = [alert for job in jobs for alert in _job_alerts(job, now=now, stale_lease_seconds=stale_lease_seconds)]
    has_attention = bool(alerts or any(job.status == "failed" for job in jobs) or run.status in {"attention", "failed", "cancelled"})
    return {
        "id": run.id,
        "card_type": "run",
        "lane": _run_lane(run.status, has_attention=has_attention),
        "title": _safe_text(run.title),
        "status": run.status,
        "public_ref": _safe_text(run.public_ref, limit=80),
        "risk_level": run.risk_level,
        "updated_at": run.updated_at,
        "progress": _progress(jobs),
        "alerts": alerts[:10],
        "mirror_only": True,
    }


def _job_card(job: RuntimeJob, *, run: RuntimeRun, now: int, stale_lease_seconds: int) -> dict[str, Any]:
    return {
        "id": job.id,
        "card_type": "job",
        "run_id": job.run_id,
        "run_title": _safe_text(run.title, limit=160),
        "lane": _job_lane(job.status),
        "title": _safe_text(job.title),
        "role": job.role,
        "status": job.status,
        "priority": job.priority,
        "attempts": {"used": job.attempt_count, "max": job.max_attempts},
        "lease_owner": _safe_text(job.lease_owner, limit=120) if job.lease_owner else None,
        "lease_expires_at": job.lease_expires_at,
        "heartbeat_at": job.heartbeat_at,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "alerts": _job_alerts(job, now=now, stale_lease_seconds=stale_lease_seconds),
        "mirror_only": True,
    }


def _global_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    run_rows = conn.execute("SELECT status, COUNT(*) AS count FROM runtime_runs GROUP BY status").fetchall()
    job_rows = conn.execute("SELECT status, COUNT(*) AS count FROM runtime_jobs GROUP BY status").fetchall()
    return {
        "runs_by_status": {str(row["status"]): int(row["count"]) for row in run_rows},
        "jobs_by_status": {str(row["status"]): int(row["count"]) for row in job_rows},
        "open_findings": int(conn.execute("SELECT COUNT(*) FROM runtime_findings WHERE status='open'").fetchone()[0]),
        "active_approvals": int(conn.execute("SELECT COUNT(*) FROM runtime_approvals WHERE status='active'").fetchone()[0]),
    }


def build_dashboard_snapshot(
    conn: sqlite3.Connection,
    *,
    now: int | None = None,
    limit: int = 50,
    stale_lease_seconds: int = 900,
) -> dict[str, Any]:
    """Return a compact read-only dashboard/Kanban mirror of Runtime state."""
    ts = _now(now)
    runs = db.list_runs(conn, limit=limit)
    run_items: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for run in runs:
        jobs = db.list_jobs(conn, run.id)
        run_item = {
            **run.to_dict(),
            "title": _safe_text(run.title),
            "objective": _safe_text(run.objective, limit=500),
            "owner_source": _safe_text(run.owner_source, limit=180),
            "public_ref": _safe_text(run.public_ref, limit=80),
            "jobs_total": len(jobs),
            "jobs_by_status": _jobs_by_status(jobs),
            "progress": _progress(jobs),
        }
        run_items.append(run_item)
        cards.append(_run_card(run, jobs, now=ts, stale_lease_seconds=stale_lease_seconds))
        cards.extend(_job_card(job, run=run, now=ts, stale_lease_seconds=stale_lease_seconds) for job in jobs)
    return {
        "success": True,
        "mirror_only": True,
        "source": "runtime_sqlite",
        "generated_at": ts,
        "counts": _global_counts(conn),
        "runs": run_items,
        "cards": cards,
        "raw_job_bodies_returned": False,
        "approval_command_payloads_returned": False,
    }
