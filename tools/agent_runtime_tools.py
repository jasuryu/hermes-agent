"""Agent Runtime tools for the Orchestrator.

These tools expose the new runtime machine truth without surfacing legacy Kanban
internals.  They are intentionally compact: create/read runs and jobs, record
Orchestrator decisions, and record Sentinel findings.
"""

from __future__ import annotations

import json
from typing import Any

from agent_runtime import db, policy


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _err(message: str) -> str:
    return _json({"success": False, "error": message})


def runtime_create_run(
    title: str,
    objective: str = "",
    owner_source: str = "",
    public_ref: str = "",
    risk_level: str = "medium",
) -> str:
    try:
        db.init_db()
        with db.connect() as conn:
            run_id = db.create_run(
                conn,
                title=title or "Untitled runtime run",
                objective=objective or "",
                owner_source=owner_source or "",
                public_ref=public_ref or "",
                risk_level=risk_level or "medium",
            )
            run = db.get_run(conn, run_id)
            return _json({"success": True, **run.to_dict()})
    except Exception as exc:
        return _err(str(exc))


def runtime_create_job(
    run_id: str,
    role: str,
    title: str,
    body: str = "",
    depends_on: list[str] | None = None,
) -> str:
    try:
        db.init_db()
        with db.connect() as conn:
            job_id = db.create_job(
                conn,
                run_id=run_id,
                role=role,
                title=title or "Untitled runtime job",
                body=body or "",
                depends_on=depends_on or [],
            )
            job = db.get_job(conn, job_id)
            return _json({"success": True, **job.to_dict()})
    except Exception as exc:
        return _err(str(exc))


def runtime_get_status(run_id: str = "") -> str:
    try:
        db.init_db()
        with db.connect() as conn:
            if run_id:
                run = db.get_run(conn, run_id)
                if run is None:
                    return _err(f"runtime run not found: {run_id}")
                payload = {
                    "success": True,
                    "run": run.to_dict(),
                    "jobs": [j.to_dict() for j in db.list_jobs(conn, run_id)],
                    "events": [e.to_dict() for e in db.list_events(conn, run_id=run_id, limit=200)],
                }
            else:
                payload = {"success": True, **db.doctor_status(conn)}
            return _json(payload)
    except Exception as exc:
        return _err(str(exc))


def runtime_record_decision(
    run_id: str,
    kind: str,
    rationale: str = "",
    job_id: str = "",
    linked_findings: list[str] | None = None,
) -> str:
    try:
        db.init_db()
        with db.connect() as conn:
            decision_id = db.record_decision(
                conn,
                run_id=run_id,
                kind=kind,
                rationale=rationale or "",
                job_id=job_id or None,
                linked_findings=linked_findings or [],
            )
            return _json({"success": True, "id": decision_id, "run_id": run_id, "kind": kind})
    except Exception as exc:
        return _err(str(exc))


def runtime_add_finding(
    run_id: str,
    severity: str,
    category: str,
    summary: str,
    job_id: str = "",
    evidence_ref: str = "",
    recommendation: str = "",
) -> str:
    try:
        db.init_db()
        with db.connect() as conn:
            finding_id = db.add_finding(
                conn,
                run_id=run_id,
                job_id=job_id or None,
                severity=severity,
                category=category,
                summary=summary,
                evidence_ref=evidence_ref,
                recommendation=recommendation,
            )
            return _json({"success": True, "id": finding_id, "run_id": run_id, "severity": severity})
    except Exception as exc:
        return _err(str(exc))


def runtime_check_command(command: str) -> str:
    verdict = policy.classify_command(command)
    return _json({"success": True, **verdict.to_dict()})


def runtime_record_approval(
    run_id: str,
    target: str,
    commands: list[str],
    reason: str,
    blast_radius: str,
    rollback: str,
    verification: list[str],
    approved_by: str,
    approval_source: str = "operator",
    job_id: str = "",
    approval_nonce: str = "",
) -> str:
    return _err(
        "runtime approval writer is disabled for model-callable tools; "
        "use trusted operator CLI: hermes runtime approve-command --write "
        "--operator-confirm APPROVE_RUNTIME_APPROVAL"
    )



def check_agent_runtime_requirements() -> bool:
    return True


def check_agent_runtime_approval_writer_requirements(approval_nonce: str = "") -> bool:
    return False


RUNTIME_CREATE_RUN_SCHEMA = {
    "name": "runtime_create_run",
    "description": "Create a durable Agent Runtime run. Runtime is machine execution truth; YouTrack remains human/project truth.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "objective": {"type": "string"},
            "owner_source": {"type": "string", "description": "Owner/chat/session source reference."},
            "public_ref": {"type": "string", "description": "Human-visible ref such as HP-88."},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high", "prod_sensitive"], "default": "medium"},
        },
        "required": ["title"],
    },
}

RUNTIME_CREATE_JOB_SCHEMA = {
    "name": "runtime_create_job",
    "description": "Create one bounded Agent Runtime job for explorer, code_worker, ops_worker, scribe, or sentinel.",
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "role": {"type": "string", "enum": ["explorer", "code_worker", "ops_worker", "scribe", "sentinel"]},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["run_id", "role", "title"],
    },
}

RUNTIME_GET_STATUS_SCHEMA = {
    "name": "runtime_get_status",
    "description": "Read Agent Runtime status. With run_id returns run/jobs/events; without run_id returns doctor-style counts.",
    "parameters": {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": [],
    },
}

RUNTIME_RECORD_DECISION_SCHEMA = {
    "name": "runtime_record_decision",
    "description": "Record an Orchestrator decision such as proceed, fix_forward, accept_risk, safer_alternative, no_go, or close.",
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "kind": {"type": "string"},
            "rationale": {"type": "string"},
            "job_id": {"type": "string"},
            "linked_findings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["run_id", "kind"],
    },
}

RUNTIME_ADD_FINDING_SCHEMA = {
    "name": "runtime_add_finding",
    "description": "Record a permissive Sentinel finding. Findings do not block by themselves; Orchestrator reconciles them before closing a run.",
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "severity": {"type": "string", "enum": ["info", "low", "medium", "high", "critical"]},
            "category": {"type": "string"},
            "summary": {"type": "string"},
            "job_id": {"type": "string"},
            "evidence_ref": {"type": "string"},
            "recommendation": {"type": "string"},
        },
        "required": ["run_id", "severity", "category", "summary"],
    },
}

RUNTIME_CHECK_COMMAND_SCHEMA = {
    "name": "runtime_check_command",
    "description": "Classify a shell/ops command against Runtime policy. Destructive/prod mutations require an approval packet.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}

RUNTIME_RECORD_APPROVAL_SCHEMA = {
    "name": "runtime_record_approval",
    "description": "Record an exact-scope approval packet for one or more commands. Used before Ops Worker mutations.",
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "target": {"type": "string"},
            "commands": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
            "blast_radius": {"type": "string"},
            "rollback": {"type": "string"},
            "verification": {"type": "array", "items": {"type": "string"}},
            "approved_by": {"type": "string"},
            "approval_source": {"type": "string"},
            "job_id": {"type": "string"},
        },
        "required": ["run_id", "target", "commands", "reason", "blast_radius", "rollback", "verification", "approved_by"],
    },
}

from tools.registry import registry

registry.register(
    name="runtime_create_run",
    toolset="agent_runtime",
    schema=RUNTIME_CREATE_RUN_SCHEMA,
    handler=lambda args, **kw: runtime_create_run(
        title=args.get("title", ""),
        objective=args.get("objective", ""),
        owner_source=args.get("owner_source", ""),
        public_ref=args.get("public_ref", ""),
        risk_level=args.get("risk_level", "medium"),
    ),
    check_fn=check_agent_runtime_requirements,
    emoji="🧠",
)
registry.register(
    name="runtime_create_job",
    toolset="agent_runtime",
    schema=RUNTIME_CREATE_JOB_SCHEMA,
    handler=lambda args, **kw: runtime_create_job(
        run_id=args.get("run_id", ""),
        role=args.get("role", ""),
        title=args.get("title", ""),
        body=args.get("body", ""),
        depends_on=args.get("depends_on") or [],
    ),
    check_fn=check_agent_runtime_requirements,
    emoji="🧠",
)
registry.register(
    name="runtime_get_status",
    toolset="agent_runtime",
    schema=RUNTIME_GET_STATUS_SCHEMA,
    handler=lambda args, **kw: runtime_get_status(run_id=args.get("run_id", "")),
    check_fn=check_agent_runtime_requirements,
    emoji="🧠",
)
registry.register(
    name="runtime_record_decision",
    toolset="agent_runtime",
    schema=RUNTIME_RECORD_DECISION_SCHEMA,
    handler=lambda args, **kw: runtime_record_decision(
        run_id=args.get("run_id", ""),
        kind=args.get("kind", ""),
        rationale=args.get("rationale", ""),
        job_id=args.get("job_id", ""),
        linked_findings=args.get("linked_findings") or [],
    ),
    check_fn=check_agent_runtime_requirements,
    emoji="🧠",
)
registry.register(
    name="runtime_add_finding",
    toolset="agent_runtime",
    schema=RUNTIME_ADD_FINDING_SCHEMA,
    handler=lambda args, **kw: runtime_add_finding(
        run_id=args.get("run_id", ""),
        severity=args.get("severity", ""),
        category=args.get("category", ""),
        summary=args.get("summary", ""),
        job_id=args.get("job_id", ""),
        evidence_ref=args.get("evidence_ref", ""),
        recommendation=args.get("recommendation", ""),
    ),
    check_fn=check_agent_runtime_requirements,
    emoji="🧠",
)
registry.register(
    name="runtime_check_command",
    toolset="agent_runtime",
    schema=RUNTIME_CHECK_COMMAND_SCHEMA,
    handler=lambda args, **kw: runtime_check_command(command=args.get("command", "")),
    check_fn=check_agent_runtime_requirements,
    emoji="🛡️",
)
