"""Guarded ops terminal tool for Agent Runtime Ops Workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent_runtime import ops_guard


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _err(message: str) -> str:
    return _json({"success": False, "status": "blocked", "error": message})


def check_runtime_ops_terminal_requirements() -> bool:
    if os.getenv("HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION") != "1":
        return False
    context_path = os.getenv("HERMES_AGENT_RUNTIME_CONTEXT", "")
    return bool(context_path and Path(context_path).is_file())


def runtime_ops_terminal(command: str, target: str, timeout: int = 120) -> str:
    """Execute one guarded ops command using brokered context approval snapshots.

    This tool never opens the writable Runtime DB.  It only reads the context
    file already mounted read-only into the worker by the trusted parent.
    """
    try:
        context_path = os.getenv("HERMES_AGENT_RUNTIME_CONTEXT", "")
        if not context_path:
            return _err("runtime ops terminal requires brokered worker context")
        context = ops_guard.load_context(context_path)
        return _json(ops_guard.guarded_ops_terminal(
            context,
            command=command or "",
            target=target or "",
            timeout=int(timeout or 120),
        ))
    except Exception as exc:
        return _err(str(exc))


RUNTIME_OPS_TERMINAL_SCHEMA = {
    "name": "runtime_ops_terminal",
    "description": (
        "Run one infrastructure/ops command through the Agent Runtime mandatory command guard. "
        "Read-only discovery may run without approval; secret-bearing reads and mutations require "
        "an exact active approval packet in the brokered worker context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Single command string. Compound shell syntax is blocked unless exactly approved."},
            "target": {"type": "string", "description": "Exact approval target/scope, e.g. cluster/name namespace/name resource/name."},
            "timeout": {"type": "integer", "description": "Foreground timeout in seconds", "default": 120},
        },
        "required": ["command", "target"],
    },
}


from tools.registry import registry

registry.register(
    name="runtime_ops_terminal",
    toolset="ops_terminal",
    schema=RUNTIME_OPS_TERMINAL_SCHEMA,
    handler=lambda args, **kw: runtime_ops_terminal(
        command=args.get("command", ""),
        target=args.get("target", ""),
        timeout=args.get("timeout", 120),
    ),
    check_fn=check_runtime_ops_terminal_requirements,
    emoji="🛡️",
    max_result_size_chars=100_000,
)
