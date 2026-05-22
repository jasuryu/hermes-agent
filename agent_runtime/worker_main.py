"""Agent Runtime worker entrypoint.

The worker stays fail-closed unless the trusted parent explicitly enables role
execution and provides a brokered context file.  It never opens the writable
runtime DB; the parent broker validates leases before context materialization and
records results after subprocess completion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

from . import worker_execution


def _print_json(payload: dict[str, Any], *, stderr: bool = False) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr if stderr else sys.stdout)


def main(argv: list[str] | None = None, *, agent_factory: Callable[..., Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Hermes Agent Runtime job")
    parser.add_argument("--job", required=True, help="Runtime job id")
    args = parser.parse_args(argv)

    attempt_id = os.getenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", "")
    lease_owner = os.getenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "")
    if not attempt_id or not lease_owner:
        _print_json({"success": False, "error": "worker requires active lease identity before context is exposed"}, stderr=True)
        return 1

    if not worker_execution.execution_gate_enabled():
        _print_json({
            "success": False,
            "mode": "worker_execution_disabled",
            "error": "worker execution requires HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION=1 from the trusted parent launch gate",
        })
        return 1

    context_path = os.getenv("HERMES_AGENT_RUNTIME_CONTEXT", "")
    if not context_path:
        _print_json({"success": False, "error": "worker execution requires brokered context path"}, stderr=True)
        return 1

    try:
        result = worker_execution.run_role_worker(
            job_id=args.job,
            attempt_id=attempt_id,
            lease_owner=lease_owner,
            context_path=context_path,
            agent_factory=agent_factory,
        )
    except Exception as exc:
        _print_json({"success": False, "error": str(exc)}, stderr=True)
        return 1

    _print_json(result.to_dict())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
