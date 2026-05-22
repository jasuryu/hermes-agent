"""CLI for Hermes Agent Runtime (`hermes runtime ...`)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
import time
from typing import Any
from urllib.parse import quote

from agent_runtime import approval_channel, cleanup, dashboard_mirror, db, observability, scheduler, scribe_sync, youtrack_sync
from agent_runtime.roles import roles_snapshot


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _run_to_dict(run) -> dict[str, Any]:
    return run.to_dict() if hasattr(run, "to_dict") else dict(run)


def _job_to_dict(job) -> dict[str, Any]:
    return job.to_dict() if hasattr(job, "to_dict") else dict(job)


@contextlib.contextmanager
def _connect_existing_runtime_db_readonly():
    db_path = db.runtime_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"runtime db not found: {db_path}")
    uri = "file:" + quote(str(db_path), safe="/:") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def _systemd_user_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "hermes-agent-runtime.service"


_APPROVE_COMMAND_ARGPARSE_TOKEN = object()


def _operator_cli_argparse_token_valid(args: argparse.Namespace) -> bool:
    return getattr(args, "_operator_cli_entrypoint", None) is _APPROVE_COMMAND_ARGPARSE_TOKEN


def _format_interval(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _validate_service_value(value: str) -> str:
    text = str(value)
    if any(ch in text for ch in ("\n", "\r", "\0")):
        raise ValueError("invalid runtime service unit value: control characters are not allowed")
    return text


def _validate_lease_owner(value: str) -> str:
    text = _validate_service_value(value or "agent-runtime-daemon")
    if not re.fullmatch(r"[A-Za-z0-9_.:@-]+", text):
        raise ValueError("invalid runtime service unit value: lease owner must be identifier-like")
    return text


def _systemd_quote(value: str) -> str:
    text = _validate_service_value(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def runtime_service_unit_text(*, interval: float = 5.0, lease_owner: str = "agent-runtime-daemon") -> str:
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    interval_text = _format_interval(max(0.0, float(interval)))
    lease = _validate_lease_owner(lease_owner)
    return "\n".join(
        [
            "[Unit]",
            "Description=Hermes Agent Runtime daemon (recovery-only)",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={_validate_service_value(str(project_root))}",
            f"Environment={_systemd_quote(f'HERMES_HOME={hermes_home}')}",
            f"ExecStart={_systemd_quote(python)} -m hermes_cli.main runtime daemon --interval {interval_text} --lease-owner {_systemd_quote(lease)}",
            "Restart=always",
            "RestartSec=10",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def _install_runtime_service(*, unit_text: str, write: bool, reload: bool) -> tuple[Path, str]:
    unit_path = _systemd_user_unit_path()
    if not write:
        return unit_path, ""
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_text, encoding="utf-8")
    unit_path.chmod(0o644)
    if reload:
        try:
            result = subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True, check=False, timeout=30)
        except subprocess.TimeoutExpired:
            return unit_path, "systemctl --user daemon-reload timed out"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "systemctl --user daemon-reload failed").strip()
            return unit_path, detail
    return unit_path, ""


def runtime_command(args: argparse.Namespace) -> int:
    action = getattr(args, "runtime_command", None) or "doctor"

    if action == "service-unit":
        try:
            unit = runtime_service_unit_text(
                interval=max(0.0, float(getattr(args, "interval", 5.0) or 0.0)),
                lease_owner=getattr(args, "lease_owner", "agent-runtime-daemon") or "agent-runtime-daemon",
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(unit, end="")
        return 0

    if action == "install-service":
        try:
            unit = runtime_service_unit_text(
                interval=max(0.0, float(getattr(args, "interval", 5.0) or 0.0)),
                lease_owner=getattr(args, "lease_owner", "agent-runtime-daemon") or "agent-runtime-daemon",
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        write = bool(getattr(args, "write", False))
        try:
            unit_path, reload_error = _install_runtime_service(unit_text=unit, write=write, reload=bool(getattr(args, "reload", False)))
        except OSError as exc:
            print(f"failed to install Runtime daemon unit: {exc}", file=sys.stderr)
            return 1
        if reload_error:
            print(f"Installed recovery-only Runtime daemon unit: {unit_path}")
            print(f"daemon-reload failed: {reload_error}", file=sys.stderr)
            return 1
        if write:
            print(f"Installed recovery-only Runtime daemon unit: {unit_path}")
            print("Not started automatically. Run: systemctl --user enable --now hermes-agent-runtime.service")
        else:
            print(f"DRY RUN: would write recovery-only Runtime daemon unit to {unit_path}")
            print(unit, end="")
        return 0

    if action == "approve-command":
        write = bool(getattr(args, "write", False))
        if write and getattr(args, "operator_confirm", "") != approval_channel.OPERATOR_CONFIRM_PHRASE:
            print(f"approve-command --write requires --operator-confirm {approval_channel.OPERATOR_CONFIRM_PHRASE}", file=sys.stderr)
            return 1
        try:
            packet = approval_channel.build_operator_approval_packet(
                target=getattr(args, "target", "") or "",
                commands=list(getattr(args, "commands", []) or []),
                reason=getattr(args, "reason", "") or "",
                blast_radius=getattr(args, "blast_radius", "") or "",
                rollback=getattr(args, "rollback", "") or "",
                verification=list(getattr(args, "verification", []) or []),
                approved_by=getattr(args, "approved_by", "") or "",
                approval_source=getattr(args, "approval_source", "") or "operator-cli",
                expires_in_seconds=getattr(args, "expires_in_seconds", None),
            )
            db_path = db.runtime_db_path()
            if not db_path.exists():
                raise FileNotFoundError(f"runtime db not found: {db_path}")
            approval_id = ""
            with db.connect() as conn:
                run_id = getattr(args, "run_id", "") or ""
                job_id = getattr(args, "job_id", "") or ""
                if db.get_run(conn, run_id) is None:
                    raise ValueError(f"runtime run not found: {run_id}")
                db._validate_optional_job_ref(conn, run_id=run_id, job_id=job_id or None)
                db.validate_approval_packet(packet)
                if write:
                    if not _operator_cli_argparse_token_valid(args):
                        raise PermissionError("approve-command --write must run through the hermes runtime approve-command operator CLI parser")
                    ts = int(time.time())
                    approval_id = f"appr_{secrets.token_hex(6)}"
                    conn.execute(
                        """
                        INSERT INTO runtime_approvals
                        (id, run_id, job_id, target, commands_json, command_hashes_json, reason,
                         blast_radius, rollback, verification_json, approved_by, approval_source,
                         approved_at, expires_at, scope_hash, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                        """,
                        (
                            approval_id,
                            run_id,
                            job_id or None,
                            packet.get("target", ""),
                            json.dumps(packet.get("commands") or [], ensure_ascii=False, sort_keys=True),
                            json.dumps(packet.get("command_hashes") or [], ensure_ascii=False, sort_keys=True),
                            packet.get("reason", ""),
                            packet.get("blast_radius", ""),
                            packet.get("rollback", ""),
                            json.dumps(packet.get("verification") or [], ensure_ascii=False, sort_keys=True),
                            packet.get("approved_by", ""),
                            packet.get("approval_source", ""),
                            int(packet.get("approved_at") or ts),
                            packet.get("expires_at"),
                            packet.get("scope_hash", ""),
                        ),
                    )
                    db.add_event(
                        conn,
                        run_id=run_id,
                        job_id=job_id or None,
                        kind="approval_recorded",
                        payload={"approval_id": approval_id, "target": packet.get("target", "")},
                        now=ts,
                    )
                payload = {
                    "written": write,
                    "approval_id": approval_id,
                    "run_id": run_id,
                    "job_id": job_id or None,
                    "operator_channel": "runtime-operator-cli",
                    "packet": packet,
                }
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except (PermissionError, ValueError, sqlite3.Error) as exc:
            print(f"failed to record Runtime approval: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            if write:
                print(f"Recorded Runtime approval {approval_id} for run {payload['run_id']} via trusted operator channel")
            else:
                print("DRY RUN: would record Runtime approval packet via trusted operator channel")
                print(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if action == "sync-obsidian":
        try:
            with _connect_existing_runtime_db_readonly() as conn:
                payload = scribe_sync.sync_runbook_to_obsidian(
                    conn,
                    getattr(args, "run_id", ""),
                    vault_path=getattr(args, "vault_path", "") or None,
                    write=bool(getattr(args, "write", False)),
                )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except sqlite3.Error as exc:
            print(f"failed to read runtime db: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"failed to write Obsidian runbook: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            if payload["written"]:
                print(f"Wrote documentation-only Runtime Obsidian mirror: {payload['path']}")
            else:
                print(f"DRY RUN: would write documentation-only Runtime Obsidian mirror to {payload['path']}")
                print(payload["markdown"], end="")
        return 0

    if action == "sync-youtrack":
        try:
            with _connect_existing_runtime_db_readonly() as conn:
                payload = youtrack_sync.sync_run_to_youtrack(
                    conn,
                    getattr(args, "run_id", ""),
                    issue_id=getattr(args, "issue_id", "") or None,
                    stage=getattr(args, "stage", "") or None,
                    write=bool(getattr(args, "write", False)),
                    ytctl=getattr(args, "ytctl", "ytctl") or "ytctl",
                )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"failed to sync Runtime mirror to YouTrack: {exc}", file=sys.stderr)
            return 1
        except sqlite3.Error as exc:
            print(f"failed to read runtime db: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"failed to run YouTrack sync command: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            if payload["written"]:
                print(f"Synced documentation-only Runtime YouTrack mirror to {payload['issue_id']}: {', '.join(payload['operations'])}")
            else:
                print(f"DRY RUN: would sync documentation-only Runtime YouTrack mirror to {payload['issue_id']}: {', '.join(payload['operations'])}")
                print(payload["comment"], end="")
        return 0

    if action == "mirror":
        try:
            with _connect_existing_runtime_db_readonly() as conn:
                payload = dashboard_mirror.build_dashboard_snapshot(conn)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except sqlite3.Error as exc:
            print(f"failed to read runtime db: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            print(f"Runtime mirror: runs={len(payload['runs'])} cards={len(payload['cards'])} mirror_only=true")
        return 0

    if action == "health":
        service_status = observability.probe_runtime_service()
        try:
            with _connect_existing_runtime_db_readonly() as conn:
                payload = observability.build_health_snapshot(conn, service_status=service_status)
        except FileNotFoundError:
            payload = observability.build_health_snapshot(None, service_status=service_status, db_missing=True)
        except sqlite3.Error as exc:
            payload = observability.build_health_snapshot(None, service_status=service_status)
            payload["status"] = "critical"
            payload.setdefault("alerts", []).append(
                observability._alert(
                    "critical",
                    "runtime_db_read_failed",
                    "Runtime DB could not be read.",
                    detail=str(exc),
                )
            )
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            print(f"Runtime health: {payload['status']}")
            for alert in payload.get("alerts", []):
                print(f"{alert.get('severity', 'info').upper()}: {alert.get('code')} — {alert.get('message')}")
        return 0 if payload.get("status") == "ok" else 1

    if action == "cleanup-sandboxes":
        try:
            payload = cleanup.cleanup_worker_sandboxes(
                parent=getattr(args, "parent", "") or None,
                max_age_seconds=int(getattr(args, "max_age_seconds", 86400) or 86400),
                execute=bool(getattr(args, "execute", False)),
            )
        except (ValueError, OSError) as exc:
            print(f"failed to cleanup Runtime worker sandboxes: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            mode = "removed" if payload["executed"] else "would remove"
            print(f"Runtime sandbox cleanup: {mode} {payload['candidates']} stale sandbox(s) under {payload['parent']}")
        return 0

    db.init_db()
    with db.connect() as conn:
        if action in {"init", "doctor"}:
            status = db.doctor_status(conn)
            status["roles"] = roles_snapshot()
            if getattr(args, "json", False):
                _print_json(status)
            else:
                print("Agent Runtime: OK" if status["ok"] else "Agent Runtime: NOT OK")
                print(f"DB: {status['db_path']}")
                print(f"Runs: {status['runs']}  Jobs: {status['jobs']}  Ready: {status['ready_jobs']}  Leased: {status['leased_jobs']}")
                print(f"Open findings: {status['open_findings']}  Active approvals: {status['active_approvals']}")
            return 0

        if action == "create-run":
            run_id = db.create_run(
                conn,
                title=getattr(args, "title", "") or "Untitled runtime run",
                objective=getattr(args, "objective", "") or "",
                owner_source=getattr(args, "owner_source", "") or "",
                public_ref=getattr(args, "public_ref", "") or "",
            )
            run = db.get_run(conn, run_id)
            payload = _run_to_dict(run)
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                print(f"Created runtime run {run_id}: {run.title}")
            return 0

        if action == "runs":
            payload = [_run_to_dict(r) for r in db.list_runs(conn)]
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                if not payload:
                    print("No runtime runs.")
                for item in payload:
                    ref = f" [{item['public_ref']}]" if item.get("public_ref") else ""
                    print(f"{item['id']}  {item['status']}{ref}  {item['title']}")
            return 0

        if action == "show":
            run_id = getattr(args, "run_id", "")
            run = db.get_run(conn, run_id)
            if run is None:
                print(f"runtime run not found: {run_id}", file=sys.stderr)
                return 1
            payload = {
                "run": _run_to_dict(run),
                "jobs": [_job_to_dict(j) for j in db.list_jobs(conn, run_id)],
                "events": [e.to_dict() for e in db.list_events(conn, run_id=run_id, limit=200)],
            }
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                print(f"{run.id}  {run.status}  {run.title}")
                if run.public_ref:
                    print(f"Public ref: {run.public_ref}")
                print(f"Jobs: {len(payload['jobs'])}  Events: {len(payload['events'])}")
            return 0

        if action == "create-job":
            job_id = db.create_job(
                conn,
                run_id=getattr(args, "run_id", ""),
                role=getattr(args, "role", "explorer"),
                title=getattr(args, "title", "") or "Untitled runtime job",
                body=getattr(args, "body", "") or "",
                depends_on=getattr(args, "depends_on", []) or [],
            )
            job = db.get_job(conn, job_id)
            payload = _job_to_dict(job)
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                print(f"Created runtime job {job_id}: {job.title}")
            return 0

        if action == "events":
            events = [e.to_dict() for e in db.list_events(conn, run_id=getattr(args, "run_id", None), limit=200)]
            if getattr(args, "json", False):
                _print_json(events)
            else:
                for event in events:
                    print(f"{event['id']} {event['kind']} {event.get('job_id') or ''}")
            return 0

        if action == "dispatch-once":
            result = scheduler.dispatch_once(
                conn,
                lease_owner=getattr(args, "lease_owner", "agent-runtime-daemon") or "agent-runtime-daemon",
                max_claims=max(1, int(getattr(args, "max_claims", 1) or 1)),
                spawn=bool(getattr(args, "spawn", False)),
                enable_spawn=bool(getattr(args, "enable_spawn", False)),
                isolation_backend=getattr(args, "isolation_backend", "disabled") or "disabled",
            )
            conn.commit()
            payload = result.to_dict()
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                print(f"Dispatch tick: claimed={result.claimed} recovered={result.recovered} promoted={result.promoted} spawned={result.spawned}")
                for error in result.errors:
                    print(f"ERROR: {error}", file=sys.stderr)
            return 1 if result.errors else 0

        if action == "daemon":
            max_ticks = getattr(args, "max_ticks", None)
            interval = max(0.0, float(getattr(args, "interval", 5.0) or 0.0))
            lease_owner = getattr(args, "lease_owner", "agent-runtime-daemon") or "agent-runtime-daemon"
            results: list[dict[str, Any]] = []
            ticks = 0
            had_errors = False
            try:
                while max_ticks is None or ticks < int(max_ticks):
                    result = scheduler.dispatch_once(
                        conn,
                        lease_owner=lease_owner,
                        spawn=bool(getattr(args, "spawn", False)),
                        enable_spawn=bool(getattr(args, "enable_spawn", False)),
                        isolation_backend=getattr(args, "isolation_backend", "disabled") or "disabled",
                    )
                    conn.commit()
                    results.append(result.to_dict())
                    if result.errors:
                        had_errors = True
                        if not getattr(args, "json", False):
                            for error in result.errors:
                                print(f"ERROR: {error}", file=sys.stderr)
                    ticks += 1
                    if max_ticks is not None and ticks >= int(max_ticks):
                        break
                    if interval:
                        time.sleep(interval)
            except KeyboardInterrupt:
                pass
            payload = {"ticks": ticks, "results": results}
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                print(f"Runtime daemon stopped after {ticks} tick(s).")
            return 1 if had_errors else 0

    print(f"unknown runtime command: {action}", file=sys.stderr)
    return 1


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = parent_subparsers.add_parser(
        "runtime",
        help="Agent Runtime runs/jobs/events/finding control plane",
        description="Manage the Hermes Agent Runtime machine execution truth.",
    )
    sub = parser.add_subparsers(dest="runtime_command")

    p_init = sub.add_parser("init", help="Initialize runtime DB and print status")
    p_init.add_argument("--json", action="store_true")

    p_doctor = sub.add_parser("doctor", help="Check runtime DB/state health")
    p_doctor.add_argument("--json", action="store_true")

    p_runs = sub.add_parser("runs", help="List runtime runs")
    p_runs.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Show one runtime run")
    p_show.add_argument("run_id")
    p_show.add_argument("--json", action="store_true")

    p_create = sub.add_parser("create-run", help="Create a runtime run")
    p_create.add_argument("title")
    p_create.add_argument("--objective", default="")
    p_create.add_argument("--owner-source", default="")
    p_create.add_argument("--public-ref", default="")
    p_create.add_argument("--json", action="store_true")

    p_job = sub.add_parser("create-job", help="Create a bounded runtime job")
    p_job.add_argument("run_id")
    p_job.add_argument("title")
    p_job.add_argument("--role", default="explorer")
    p_job.add_argument("--body", default="")
    p_job.add_argument("--depends-on", action="append", default=[])
    p_job.add_argument("--json", action="store_true")

    p_events = sub.add_parser("events", help="List runtime events")
    p_events.add_argument("run_id", nargs="?")
    p_events.add_argument("--json", action="store_true")

    p_approve = sub.add_parser("approve-command", help="Preview or write an exact-scope Runtime approval packet from a trusted operator channel")
    p_approve.add_argument("run_id")
    p_approve.add_argument("--job-id", default="", help="Optional job-scoped approval; omitted means run-level approval")
    p_approve.add_argument("--target", required=True, help="Exact target/scope, e.g. cluster/name namespace/name resource/name")
    p_approve.add_argument("--command", dest="commands", action="append", required=True, help="Exact command to approve; repeat for multiple commands")
    p_approve.add_argument("--reason", required=True)
    p_approve.add_argument("--blast-radius", required=True, help="Expected blast radius of the approved action")
    p_approve.add_argument("--rollback", required=True, help="Rollback command/procedure")
    p_approve.add_argument("--verification", action="append", required=True, help="Verification command/procedure; repeat for multiple checks")
    p_approve.add_argument("--approved-by", required=True, help="Human/operator identity")
    p_approve.add_argument("--approval-source", default="operator-cli", help="Trusted operator channel/source label")
    p_approve.add_argument("--expires-in-seconds", type=int, default=None)
    p_approve.add_argument("--write", action="store_true", help="Actually record the approval; omitted means dry-run preview")
    p_approve.add_argument("--operator-confirm", default="", help=f"Required with --write: {approval_channel.OPERATOR_CONFIRM_PHRASE}")
    p_approve.add_argument("--json", action="store_true")
    p_approve.set_defaults(_operator_cli_entrypoint=_APPROVE_COMMAND_ARGPARSE_TOKEN)

    p_sync = sub.add_parser("sync-obsidian", help="Render/sync a documentation-only Obsidian mirror for one runtime run")
    p_sync.add_argument("run_id")
    p_sync.add_argument("--vault-path", default="", help="Obsidian vault path; defaults to OBSIDIAN_VAULT_PATH or $HERMES_HOME/obsidian")
    p_sync.add_argument("--write", action="store_true", help="Actually write the runbook note; omitted means dry-run")
    p_sync.add_argument("--json", action="store_true")

    p_yt = sub.add_parser("sync-youtrack", help="Render/sync a documentation-only YouTrack public mirror for one runtime run")
    p_yt.add_argument("run_id")
    p_yt.add_argument("--issue-id", default="", help="YouTrack issue id; defaults to the runtime run public_ref")
    p_yt.add_argument("--stage", default="", help="Optional explicit YouTrack Stage value to set after posting the comment")
    p_yt.add_argument("--ytctl", default="ytctl", help="ytctl executable path/name")
    p_yt.add_argument("--write", action="store_true", help="Actually post/update YouTrack; omitted means dry-run")
    p_yt.add_argument("--json", action="store_true")

    p_mirror = sub.add_parser("mirror", help="Print a read-only dashboard/Kanban mirror snapshot from Runtime state")
    p_mirror.add_argument("--json", action="store_true")

    p_health = sub.add_parser("health", help="Read-only Runtime daemon health snapshot and alerts")
    p_health.add_argument("--json", action="store_true")

    p_cleanup = sub.add_parser("cleanup-sandboxes", help="Dry-run or execute stale worker sandbox cleanup")
    p_cleanup.add_argument("--parent", default="", help="Sandbox parent to scan; defaults to /tmp")
    p_cleanup.add_argument("--max-age-seconds", type=int, default=86400)
    p_cleanup.add_argument("--execute", action="store_true", help="Actually delete stale sandbox directories; omitted means dry-run")
    p_cleanup.add_argument("--json", action="store_true")

    p_dispatch = sub.add_parser("dispatch-once", help="Run one scheduler tick; worker spawn requires explicit gated flags")
    p_dispatch.add_argument("--lease-owner", default="agent-runtime-daemon")
    p_dispatch.add_argument("--max-claims", type=int, default=1)
    p_dispatch.add_argument("--spawn", action="store_true", help="Attempt worker spawn instead of recovery/promotion-only dry tick")
    p_dispatch.add_argument("--enable-spawn", action="store_true", help="Explicit operator gate required with --spawn")
    p_dispatch.add_argument("--isolation-backend", default="disabled", help="Reviewed worker isolation backend, e.g. bubblewrap")
    p_dispatch.add_argument("--json", action="store_true")

    p_daemon = sub.add_parser("daemon", help="Run a bounded/unbounded scheduler loop; worker spawn requires explicit gated flags")
    p_daemon.add_argument("--lease-owner", default="agent-runtime-daemon")
    p_daemon.add_argument("--interval", type=float, default=5.0)
    p_daemon.add_argument("--max-ticks", type=int, default=None)
    p_daemon.add_argument("--spawn", action="store_true", help="Attempt worker spawn instead of recovery/promotion-only ticks")
    p_daemon.add_argument("--enable-spawn", action="store_true", help="Explicit operator gate required with --spawn")
    p_daemon.add_argument("--isolation-backend", default="disabled", help="Reviewed worker isolation backend, e.g. bubblewrap")
    p_daemon.add_argument("--json", action="store_true")

    p_unit = sub.add_parser("service-unit", help="Print a recovery-only user systemd unit for the runtime daemon")
    p_unit.add_argument("--lease-owner", default="agent-runtime-daemon")
    p_unit.add_argument("--interval", type=float, default=5.0)

    p_install = sub.add_parser("install-service", help="Install a recovery-only user systemd unit for the runtime daemon")
    p_install.add_argument("--lease-owner", default="agent-runtime-daemon")
    p_install.add_argument("--interval", type=float, default=5.0)
    p_install.add_argument("--write", action="store_true", help="Actually write the unit file; omitted means dry-run")
    p_install.add_argument("--reload", action="store_true", help="Run systemctl --user daemon-reload after writing")

    parser.set_defaults(func=lambda args: runtime_command(args))
    return parser
