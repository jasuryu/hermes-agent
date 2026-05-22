"""Dataclasses used by the Agent Runtime DB layer."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class RuntimeRun:
    id: str
    title: str
    objective: str = ""
    owner_source: str = ""
    public_ref: str = ""
    status: str = "running"
    risk_level: str = "medium"
    orchestrator_session_id: str = ""
    summary: str = ""
    created_at: int = 0
    updated_at: int = 0
    closed_at: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeJob:
    id: str
    run_id: str
    role: str
    title: str
    body: str = ""
    status: str = "ready"
    priority: int = 0
    workspace_kind: str = "scratch"
    workspace_path: str = ""
    idempotency_key: str = ""
    max_attempts: int = 2
    attempt_count: int = 0
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[int] = None
    heartbeat_at: Optional[int] = None
    result_summary: str = ""
    created_at: int = 0
    started_at: Optional[int] = None
    completed_at: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeAttempt:
    id: str
    job_id: str
    role: str
    model: str = ""
    reasoning: str = ""
    pid: Optional[int] = None
    status: str = "starting"
    started_at: int = 0
    ended_at: Optional[int] = None
    stdout_log: str = ""
    stderr_log: str = ""
    summary: str = ""
    error: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = self.metadata or {}
        return data


@dataclass(frozen=True)
class RuntimeEvent:
    id: int
    run_id: str
    kind: str
    job_id: Optional[str] = None
    payload: dict[str, Any] | None = None
    created_at: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["payload"] = self.payload or {}
        return data


@dataclass(frozen=True)
class JobClaim:
    job_id: str
    attempt_id: str
    lease_owner: str
    lease_expires_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
