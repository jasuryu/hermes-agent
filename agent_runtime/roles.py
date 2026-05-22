"""Role defaults for the final Hermes Agent Runtime."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RuntimeRole:
    name: str
    model: str
    reasoning_effort: str
    description: str
    toolsets: tuple[str, ...]
    mode: str = "worker"
    can_mutate: bool = False
    mutation_requires_approval_packet: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["toolsets"] = list(self.toolsets)
        return data


DEFAULT_ROLES: dict[str, RuntimeRole] = {
    "orchestrator": RuntimeRole(
        name="orchestrator",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        description="Main session brain: plan, spawn, reconcile, review, decide, escalate.",
        toolsets=("all",),
        mode="main_session",
        can_mutate=True,
        mutation_requires_approval_packet=False,
    ),
    "explorer": RuntimeRole(
        name="explorer",
        model="gpt-5.4-mini",
        reasoning_effort="high",
        description="Read-only research using brokered RAG/SafeWeb/integrations.",
        toolsets=("safe_web", "file_readonly", "integrations_readonly"),
        can_mutate=False,
    ),
    "code_worker": RuntimeRole(
        name="code_worker",
        model="gpt-5.3-codex",
        reasoning_effort="high",
        description="Bounded repo/worktree implementation, tests, local docker only.",
        toolsets=("terminal", "file", "code_execution", "git"),
        can_mutate=True,
    ),
    "ops_worker": RuntimeRole(
        name="ops_worker",
        model="gpt-5.4",
        reasoning_effort="xhigh",
        description="Read-only-first infra/SRE worker; mutations require approval packet.",
        toolsets=("ops_terminal", "monitoring", "logs", "file_readonly"),
        can_mutate=True,
        mutation_requires_approval_packet=True,
    ),
    "scribe": RuntimeRole(
        name="scribe",
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        description="Obsidian docs, runbooks, ADR, daily/project logs.",
        toolsets=("obsidian_write", "file"),
        can_mutate=True,
    ),
    "sentinel": RuntimeRole(
        name="sentinel",
        model="o4-mini",
        reasoning_effort="low",
        description="Permissive read-only verifier; records findings and never owns workflow.",
        toolsets=("file_readonly", "git_readonly", "artifact_readonly", "safe_web"),
        mode="permissive_verifier",
        can_mutate=False,
    ),
}


def get_role(name: str) -> RuntimeRole:
    key = (name or "").strip().lower().replace("-", "_")
    if key not in DEFAULT_ROLES:
        raise ValueError(f"unknown runtime role: {name!r}")
    return DEFAULT_ROLES[key]


def role_names() -> list[str]:
    return sorted(DEFAULT_ROLES)


def roles_snapshot() -> dict[str, dict[str, Any]]:
    return {name: role.to_dict() for name, role in sorted(DEFAULT_ROLES.items())}
