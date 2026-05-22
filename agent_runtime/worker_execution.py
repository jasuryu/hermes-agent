"""Role-specific worker execution for Agent Runtime.

This module runs inside the isolated worker process.  It consumes only the
trusted parent-brokered context JSON and sanitized environment; it never opens the
writable runtime DB and never creates approval packets.  The trusted parent is
responsible for lease validation before materializing context and for recording
worker results after subprocess completion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import stat
from typing import Any, Callable

from hermes_constants import parse_reasoning_effort

from .roles import RuntimeRole, get_role


@dataclass(frozen=True)
class WorkerExecutionResult:
    success: bool
    role: str
    model: str
    reasoning_effort: str
    toolsets: tuple[str, ...]
    summary: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["toolsets"] = list(self.toolsets)
        return data


def execution_gate_enabled(environ: dict[str, str] | None = None) -> bool:
    import os

    env = environ if environ is not None else os.environ
    return str(env.get("HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION", "")).strip() == "1"


def _load_context(path: str | Path) -> dict[str, Any]:
    context_path = Path(path)
    if context_path.is_symlink() or not context_path.is_file():
        raise ValueError("worker context must be a regular brokered file")
    if stat.S_IMODE(context_path.lstat().st_mode) & 0o077:
        raise ValueError("worker context must not be group/world accessible")
    with context_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("worker context must be a JSON object")
    return data


def _dict_value(context: dict[str, Any], key: str) -> dict[str, Any]:
    value = context.get(key)
    return value if isinstance(value, dict) else {}


def _validate_context_identity(
    context: dict[str, Any],
    *,
    job_id: str,
    attempt_id: str,
    lease_owner: str,
) -> None:
    lease = _dict_value(context, "lease")
    job = _dict_value(context, "job")
    constraints = _dict_value(context, "constraints")
    if str(job.get("id") or "") != job_id:
        raise ValueError("context identity mismatch for job")
    if str(lease.get("attempt_id") or "") != attempt_id:
        raise ValueError("context identity mismatch for attempt")
    if str(lease.get("lease_owner") or "") != lease_owner:
        raise ValueError("context identity mismatch for lease owner")
    if constraints.get("runtime_db_access") != "forbidden":
        raise ValueError("worker context must forbid runtime DB access")


def _role_guardrail(role: RuntimeRole) -> str:
    lines = [
        "You are a bounded Hermes Agent Runtime worker, not the Orchestrator.",
        "Use only the toolsets assigned to your runtime role.",
        "Do not read or write the Agent Runtime DB; result submission is handled by the trusted parent broker.",
        "Do not create, mint, or approve approval packets.",
        "Keep output focused on the job result summary and any artifacts/findings to report.",
    ]
    if role.mutation_requires_approval_packet:
        lines.append(
            "This role is read-only-first: any infrastructure or production mutation requires an exact runtime approval packet and must fail closed without one."
        )
    elif not role.can_mutate:
        lines.append("This role is read-only; do not mutate files, infrastructure, or external systems.")
    else:
        lines.append("Mutate only within the bounded workspace described by the job and avoid external side effects.")
    return "\n".join(lines)


def build_worker_prompt(context: dict[str, Any], role: RuntimeRole) -> str:
    run = _dict_value(context, "run")
    job = _dict_value(context, "job")
    constraints = _dict_value(context, "constraints")
    rag = _dict_value(context, "rag")
    rag_block = str(rag.get("context_block") or "").strip() if rag.get("allowed") else ""
    parts = [
        "Agent Runtime worker assignment",
        "",
        f"Role: {role.name}",
        f"Role description: {role.description}",
        f"Model: {role.model}",
        f"Reasoning effort: {role.reasoning_effort}",
        f"Toolsets: {', '.join(role.toolsets)}",
        "",
        f"Run id: {run.get('id', '')}",
        f"Run title: {run.get('title', '')}",
        f"Run objective: {run.get('objective', '')}",
        f"Run risk level: {run.get('risk_level', '')}",
        "",
        f"Job id: {job.get('id', '')}",
        f"Job title: {job.get('title', '')}",
        f"Job body:\n{job.get('body', '')}",
        f"Workspace kind: {job.get('workspace_kind', '')}",
        f"Workspace path: {job.get('workspace_path', '')}",
        "",
        "Runtime constraints:",
        json.dumps(constraints, ensure_ascii=False, sort_keys=True),
    ]
    if rag_block:
        parts.extend(["", "Brokered RAG context:", rag_block])
    elif rag.get("allowed"):
        parts.extend(
            [
                "",
                "Brokered RAG context:",
                json.dumps(
                    {
                        "mode": rag.get("mode"),
                        "status": rag.get("status"),
                        "reason": rag.get("reason"),
                        "warning": rag.get("warning"),
                        "evidence_only": rag.get("evidence_only"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
        )
    parts.extend(
        [
            "",
            "Return a concise final response summarizing completed work, tests run, artifacts, findings, and blockers.",
        ]
    )
    return "\n".join(parts)


def _default_agent_factory(**kwargs: Any):
    from run_agent import AIAgent

    return AIAgent(**kwargs)


def _summary_from_agent_result(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("final_response", "response", "summary"):
            value = result.get(key)
            if value:
                return str(value)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    return str(result or "")


def run_role_worker(
    *,
    job_id: str,
    attempt_id: str,
    lease_owner: str,
    context_path: str | Path,
    agent_factory: Callable[..., Any] | None = None,
) -> WorkerExecutionResult:
    context = _load_context(context_path)
    _validate_context_identity(
        context,
        job_id=job_id,
        attempt_id=attempt_id,
        lease_owner=lease_owner,
    )
    job = _dict_value(context, "job")
    role = get_role(str(job.get("role") or ""))
    if role.mode == "main_session":
        raise ValueError("worker execution cannot use a main-session role")
    factory = agent_factory or _default_agent_factory
    prompt = build_worker_prompt(context, role)
    agent = factory(
        model=role.model,
        enabled_toolsets=list(role.toolsets),
        quiet_mode=True,
        ephemeral_system_prompt=_role_guardrail(role),
        reasoning_config=parse_reasoning_effort(role.reasoning_effort),
        platform="agent_runtime",
        user_id="agent-runtime-worker",
        user_name=f"Agent Runtime {role.name}",
        chat_id=job_id,
        chat_name="Agent Runtime",
        chat_type="worker",
        gateway_session_key=f"agent-runtime:{job_id}:{attempt_id}",
        session_id=f"agent_runtime_{job_id}_{attempt_id}",
        skip_context_files=True,
        skip_memory=True,
        load_soul_identity=False,
        max_iterations=20,
    )
    raw = agent.run_conversation(prompt)
    summary = _summary_from_agent_result(raw)
    return WorkerExecutionResult(
        success=True,
        role=role.name,
        model=role.model,
        reasoning_effort=role.reasoning_effort,
        toolsets=role.toolsets,
        summary=summary,
    )
