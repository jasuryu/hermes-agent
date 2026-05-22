"""Tests for the Agent Runtime CLI wrapper."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli.runtime import runtime_command


@pytest.fixture(autouse=True)
def _runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def _args(**kwargs):
    defaults = {
        "runtime_command": None,
        "json": False,
        "title": "Runtime run",
        "objective": "",
        "owner_source": "test",
        "public_ref": "",
        "run_id": "",
        "role": "explorer",
        "body": "",
        "depends_on": [],
        "lease_owner": "test-daemon",
        "max_claims": 1,
        "spawn": False,
        "enable_spawn": False,
        "isolation_backend": "disabled",
        "interval": 0.0,
        "max_ticks": 1,
        "write": False,
        "reload": False,
        "vault_path": "",
        "issue_id": "",
        "stage": "",
        "ytctl": "ytctl",
        "parent": "",
        "max_age_seconds": 86400,
        "execute": False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_runtime_doctor_json_reports_empty_runtime(capsys):
    rc = runtime_command(_args(runtime_command="doctor", json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["runs"] == 0
    assert payload["open_findings"] == 0


def test_runtime_create_run_and_show_json(capsys):
    rc = runtime_command(
        _args(
            runtime_command="create-run",
            json=True,
            title="Final Agent Runtime",
            objective="Implement final architecture",
            public_ref="HP-88",
        )
    )
    assert rc == 0
    created = json.loads(capsys.readouterr().out)

    rc = runtime_command(_args(runtime_command="show", json=True, run_id=created["id"]))

    assert rc == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["run"]["title"] == "Final Agent Runtime"
    assert shown["run"]["public_ref"] == "HP-88"
    assert shown["jobs"] == []


def test_runtime_dispatch_once_json_does_not_claim_without_execution(capsys):
    from agent_runtime import db

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Dispatch CLI run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Dispatch me")

    rc = runtime_command(_args(runtime_command="dispatch-once", json=True, lease_owner="cli-test"))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["claimed"] == 0
    assert payload["spawned"] == 0
    assert payload["claims"] == []
    with db.connect() as conn:
        assert db.get_job(conn, job_id).status == "ready"
        assert db.get_job(conn, job_id).attempt_count == 0


def test_runtime_dispatch_once_spawn_refusal_returns_nonzero(capsys):
    rc = runtime_command(_args(runtime_command="dispatch-once", json=True, spawn=True))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["claimed"] == 0
    assert payload["spawned"] == 0
    assert payload["errors"] == [
        "spawn mode requires explicit enable_spawn=True and a reviewed isolation backend",
    ]


def test_runtime_dispatch_once_subprocess_propagates_nonzero_exit(tmp_path):
    home = tmp_path / "cli-home"
    home.mkdir()
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "runtime", "dispatch-once", "--spawn", "--json"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["errors"] == [
        "spawn mode requires explicit enable_spawn=True and a reviewed isolation backend",
    ]


def test_runtime_daemon_spawn_refusal_returns_nonzero(capsys):
    rc = runtime_command(_args(runtime_command="daemon", json=True, spawn=True, max_ticks=1, interval=0.0))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ticks"] == 1
    assert payload["results"][0]["errors"] == [
        "spawn mode requires explicit enable_spawn=True and a reviewed isolation backend",
    ]


def test_runtime_daemon_non_json_prints_scheduler_errors(capsys):
    rc = runtime_command(_args(runtime_command="daemon", json=False, spawn=True, max_ticks=1, interval=0.0))

    assert rc == 1
    captured = capsys.readouterr()
    assert "Runtime daemon stopped after 1 tick" in captured.out
    assert "ERROR: spawn mode requires explicit enable_spawn=True" in captured.err


def test_runtime_daemon_json_runs_bounded_ticks(capsys):
    from agent_runtime import db

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Daemon CLI run")
        db.create_job(conn, run_id=run_id, role="explorer", title="Daemon claim")

    rc = runtime_command(_args(runtime_command="daemon", json=True, lease_owner="daemon-test", max_ticks=1, interval=0.0))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ticks"] == 1
    assert payload["results"][0]["claimed"] == 0


def test_runtime_dispatch_once_passes_explicit_spawn_gate(capsys, monkeypatch):
    from agent_runtime import scheduler

    observed = {}

    class FakeResult:
        claimed = 0
        recovered = 0
        promoted = 0
        spawned = 0
        errors = ()

        def to_dict(self):
            return {"claimed": 0, "spawned": 0, "errors": []}

    def fake_dispatch_once(_conn, **kwargs):
        observed.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(scheduler, "dispatch_once", fake_dispatch_once)

    rc = runtime_command(
        _args(
            runtime_command="dispatch-once",
            json=True,
            spawn=True,
            enable_spawn=True,
            isolation_backend="bubblewrap",
        )
    )

    assert rc == 0
    json.loads(capsys.readouterr().out)
    assert observed["spawn"] is True
    assert observed["enable_spawn"] is True
    assert observed["isolation_backend"] == "bubblewrap"

def test_runtime_approve_command_dry_run_does_not_write(capsys):
    from agent_runtime import db, policy

    db.init_db()
    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval CLI dry-run")

    rc = runtime_command(
        _args(
            runtime_command="approve-command",
            json=True,
            run_id=run_id,
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
            approval_source="telegram-owner",
            write=False,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] is False
    assert payload["approval_id"] == ""
    assert payload["packet"]["command_hashes"] == [policy.command_hash(command)]
    with db.connect() as conn:
        assert db.doctor_status(conn)["active_approvals"] == 0


def test_runtime_approve_command_write_requires_operator_confirm(capsys):
    from agent_runtime import db

    db.init_db()
    command = "kubectl -n prod rollout restart deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval CLI confirm")

    rc = runtime_command(
        _args(
            runtime_command="approve-command",
            json=True,
            run_id=run_id,
            target="cluster=whale namespace=prod deploy/api",
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
            approval_source="telegram-owner",
            write=True,
            operator_confirm="wrong",
        )
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "requires --operator-confirm APPROVE_RUNTIME_APPROVAL" in captured.err
    assert "kubectl" not in captured.err
    with db.connect() as conn:
        assert db.doctor_status(conn)["active_approvals"] == 0


def test_runtime_approve_command_direct_handler_cannot_write_even_with_spoofed_argv(capsys, monkeypatch):
    from agent_runtime import db

    db.init_db()
    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval direct handler")
        job_id = db.create_job(conn, run_id=run_id, role="ops_worker", title="Restart api")

    monkeypatch.setattr(sys, "argv", ["hermes", "runtime", "approve-command", run_id, "--write"])
    rc = runtime_command(
        _args(
            runtime_command="approve-command",
            json=True,
            run_id=run_id,
            job_id=job_id,
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
            approval_source="telegram-owner",
            expires_in_seconds=3600,
            write=True,
            operator_confirm="APPROVE_RUNTIME_APPROVAL",
        )
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "operator CLI parser" in captured.err
    assert "kubectl" not in captured.err
    with db.connect() as conn:
        assert db.doctor_status(conn)["active_approvals"] == 0


def test_runtime_approve_command_subprocess_write_records_exact_scope(capsys):
    from agent_runtime import db, policy

    db.init_db()
    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval CLI write")
        job_id = db.create_job(conn, run_id=run_id, role="ops_worker", title="Restart api")

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "runtime",
            "approve-command",
            run_id,
            "--job-id",
            job_id,
            "--target",
            target,
            "--command",
            command,
            "--reason",
            "restart stuck deployment",
            "--blast-radius",
            "api pods restart",
            "--rollback",
            "kubectl -n prod rollout undo deploy/api",
            "--verification",
            "kubectl -n prod rollout status deploy/api",
            "--approved-by",
            "Jasur",
            "--approval-source",
            "telegram-owner",
            "--expires-in-seconds",
            "3600",
            "--write",
            "--operator-confirm",
            "APPROVE_RUNTIME_APPROVAL",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["written"] is True
    assert payload["approval_id"].startswith("appr_")
    assert payload["packet"]["scope_hash"] == policy.approval_scope_hash(target, [command])
    with db.connect() as conn:
        approval = db.find_approval_for_command(conn, run_id=run_id, job_id=job_id, target=target, command=command)
        assert approval is not None
        assert approval["id"] == payload["approval_id"]
        assert approval["approval_source"] == "telegram-owner"
        assert db.find_approval_for_command(conn, run_id=run_id, target=target, command=command) is None


def test_runtime_approve_command_subprocess_requires_confirm(tmp_path):
    home = tmp_path / "cli-home"
    home.mkdir()
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)

    create = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "runtime", "create-run", "Approval subprocess", "--json"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert create.returncode == 0
    run_id = json.loads(create.stdout)["id"]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "runtime",
            "approve-command",
            run_id,
            "--target",
            "cluster=whale namespace=prod deploy/api",
            "--command",
            "kubectl -n prod rollout restart deploy/api",
            "--reason",
            "restart stuck deployment",
            "--blast-radius",
            "api pods restart",
            "--rollback",
            "kubectl -n prod rollout undo deploy/api",
            "--verification",
            "kubectl -n prod rollout status deploy/api",
            "--approved-by",
            "Jasur",
            "--write",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "requires --operator-confirm APPROVE_RUNTIME_APPROVAL" in result.stderr
    assert "kubectl" not in result.stderr


def test_runtime_service_unit_defaults_to_recovery_only_no_spawn(capsys, tmp_path):
    home = tmp_path / ".hermes"
    rc = runtime_command(_args(runtime_command="service-unit", interval=5.0, lease_owner="runtime-daemon"))

    assert rc == 0
    unit = capsys.readouterr().out
    assert "Description=Hermes Agent Runtime daemon" in unit
    assert f'Environment="HERMES_HOME={home}"' in unit
    assert "ExecStart=" in unit
    assert "runtime daemon" in unit
    assert '--lease-owner "runtime-daemon"' in unit
    assert "--interval 5" in unit
    assert "--spawn" not in unit
    assert "--enable-spawn" not in unit


def test_runtime_install_service_dry_run_does_not_write_unit(capsys, tmp_path):
    unit_path = tmp_path / ".config" / "systemd" / "user" / "hermes-agent-runtime.service"

    rc = runtime_command(_args(runtime_command="install-service", interval=5.0, write=False))

    assert rc == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "hermes-agent-runtime.service" in output
    assert "--spawn" not in output
    assert not unit_path.exists()


def test_runtime_service_unit_rejects_lease_owner_injection(capsys):
    rc = runtime_command(_args(runtime_command="service-unit", lease_owner="runtime-daemon\nExecStart=/bin/sh"))

    assert rc == 1
    captured = capsys.readouterr()
    assert "invalid runtime service unit value" in captured.err
    assert "ExecStart=/bin/sh" not in captured.out


def test_runtime_service_unit_dry_run_does_not_create_runtime_db(capsys):
    from agent_runtime import db

    db_path = db.runtime_db_path()
    assert not db_path.exists()

    rc = runtime_command(_args(runtime_command="service-unit", interval=5.0))

    assert rc == 0
    capsys.readouterr()
    assert not db_path.exists()


def test_runtime_install_service_reports_daemon_reload_failure(capsys, monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["systemctl"], returncode=1, stdout="", stderr="no user bus")

    monkeypatch.setattr("hermes_cli.runtime.subprocess.run", fake_run)

    rc = runtime_command(_args(runtime_command="install-service", write=True, reload=True))

    assert rc == 1
    captured = capsys.readouterr()
    assert "daemon-reload failed" in captured.err
    assert "no user bus" in captured.err


def test_runtime_install_service_reports_daemon_reload_timeout(capsys, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["systemctl", "--user", "daemon-reload"], timeout=30)

    monkeypatch.setattr("hermes_cli.runtime.subprocess.run", fake_run)

    rc = runtime_command(_args(runtime_command="install-service", write=True, reload=True))

    assert rc == 1
    captured = capsys.readouterr()
    assert "daemon-reload failed" in captured.err
    assert "timed out" in captured.err


def test_runtime_install_service_write_failure_returns_nonzero(capsys, tmp_path):
    config_path = tmp_path / ".config"
    config_path.write_text("not a directory", encoding="utf-8")

    rc = runtime_command(_args(runtime_command="install-service", write=True, reload=False))

    assert rc == 1
    captured = capsys.readouterr()
    assert "failed to install Runtime daemon unit" in captured.err


def test_runtime_sync_obsidian_dry_run_does_not_write_note(capsys, tmp_path):
    from agent_runtime import db

    vault = tmp_path / "vault"
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime docs sync", public_ref="HP-88")

    rc = runtime_command(_args(runtime_command="sync-obsidian", run_id=run_id, vault_path=str(vault), write=False))

    assert rc == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "Runtime docs sync" in output
    assert "not an execution queue" in output
    assert not list(vault.glob("**/*.md"))


def test_runtime_sync_obsidian_write_json_writes_note(capsys, tmp_path):
    from agent_runtime import db

    vault = tmp_path / "vault"
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime docs sync", public_ref="HP-88")

    rc = runtime_command(_args(runtime_command="sync-obsidian", run_id=run_id, vault_path=str(vault), write=True, json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["written"] is True
    assert payload["relative_path"].startswith("01 Hermes/Agent Runtime/Runs/")
    note_path = Path(payload["path"])
    assert note_path.exists()
    assert note_path.is_relative_to(vault)
    assert "documentation mirror only" in note_path.read_text(encoding="utf-8")


def test_runtime_sync_obsidian_unknown_run_returns_nonzero(capsys, tmp_path):
    rc = runtime_command(_args(runtime_command="sync-obsidian", run_id="run_missing", vault_path=str(tmp_path / "vault"), json=True))

    assert rc == 1
    captured = capsys.readouterr()
    assert "runtime db not found" in captured.err


def test_runtime_sync_obsidian_missing_db_does_not_create_runtime_db(capsys, tmp_path):
    from agent_runtime import db

    db_path = db.runtime_db_path()
    assert not db_path.exists()

    rc = runtime_command(_args(runtime_command="sync-obsidian", run_id="run_missing", vault_path=str(tmp_path / "vault"), json=True))

    assert rc == 1
    capsys.readouterr()
    assert not db_path.exists()


def test_runtime_sync_obsidian_write_failure_returns_nonzero(capsys, tmp_path):
    from agent_runtime import db

    vault_file = tmp_path / "vault-file"
    vault_file.write_text("not a directory", encoding="utf-8")
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime docs sync", public_ref="HP-88")

    rc = runtime_command(_args(runtime_command="sync-obsidian", run_id=run_id, vault_path=str(vault_file), write=True, json=True))

    assert rc == 1
    captured = capsys.readouterr()
    assert "failed to write Obsidian runbook" in captured.err


def test_runtime_sync_obsidian_corrupt_db_returns_nonzero(capsys, tmp_path):
    from agent_runtime import db

    db_path = db.runtime_db_path()
    db_path.parent.mkdir(parents=True)
    db_path.write_text("not sqlite", encoding="utf-8")

    rc = runtime_command(_args(runtime_command="sync-obsidian", run_id="run_missing", vault_path=str(tmp_path / "vault"), json=True))

    assert rc == 1
    captured = capsys.readouterr()
    assert "failed to read runtime db" in captured.err


def test_runtime_mirror_json_is_read_only_and_does_not_create_missing_db(capsys):
    from agent_runtime import db

    db_path = db.runtime_db_path()
    assert not db_path.exists()

    rc = runtime_command(_args(runtime_command="mirror", json=True))

    assert rc == 1
    captured = capsys.readouterr()
    assert "runtime db not found" in captured.err
    assert not db_path.exists()


def test_runtime_mirror_json_returns_dashboard_snapshot(capsys):
    from agent_runtime import db

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime mirror", public_ref="HP-88")
        db.create_job(conn, run_id=run_id, role="explorer", title="Mirror me", body="PRIVATE BODY")

    rc = runtime_command(_args(runtime_command="mirror", json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["mirror_only"] is True
    assert payload["runs"][0]["id"] == run_id
    assert "PRIVATE BODY" not in json.dumps(payload)


def test_runtime_health_json_redacts_db_read_error_detail(capsys, monkeypatch):
    def broken_connect():
        raise sqlite3.DatabaseError("failed OPENAI_API_KEY=DBERRSECRET")

    monkeypatch.setattr("hermes_cli.runtime._connect_existing_runtime_db_readonly", broken_connect)
    monkeypatch.setattr(
        "hermes_cli.runtime.observability.probe_runtime_service",
        lambda: {"ActiveState": "active", "SubState": "running", "NRestarts": "0"},
    )

    rc = runtime_command(_args(runtime_command="health", json=True))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "critical"
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "DBERRSECRET" not in encoded
    assert any(alert["code"] == "runtime_db_read_failed" for alert in payload["alerts"])


def test_runtime_health_json_alerts_missing_db_without_creating_it(capsys, monkeypatch):
    from agent_runtime import db

    db_path = db.runtime_db_path()
    assert not db_path.exists()
    monkeypatch.setattr(
        "hermes_cli.runtime.observability.probe_runtime_service",
        lambda: {"ActiveState": "active", "SubState": "running", "NRestarts": "0"},
    )

    rc = runtime_command(_args(runtime_command="health", json=True))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert any(alert["code"] == "runtime_db_missing" for alert in payload["alerts"])
    assert not db_path.exists()


def test_runtime_cleanup_sandboxes_dry_run_does_not_delete(capsys, tmp_path):
    parent = tmp_path / "tmp"
    parent.mkdir()
    stale = parent / "hermes-agent-runtime-workers-job_old-att_old-abc"
    stale.mkdir()
    os.utime(stale, (1_700_000_000 - 10_000, 1_700_000_000 - 10_000))

    rc = runtime_command(
        _args(
            runtime_command="cleanup-sandboxes",
            json=True,
            parent=str(parent),
            max_age_seconds=3600,
            execute=False,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["candidates"] == 1
    assert stale.exists()


def test_runtime_sync_youtrack_dry_run_json_uses_public_ref_issue(capsys):
    from agent_runtime import db

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime YouTrack CLI", public_ref="HP-88")

    rc = runtime_command(_args(runtime_command="sync-youtrack", run_id=run_id, json=True, write=False))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["written"] is False
    assert payload["issue_id"] == "HP-88"
    assert payload["operations"] == ["comment"]
    assert "not an execution queue" in payload["comment"]


def test_runtime_sync_youtrack_missing_db_does_not_create_runtime_db(capsys):
    from agent_runtime import db

    db_path = db.runtime_db_path()
    assert not db_path.exists()

    rc = runtime_command(_args(runtime_command="sync-youtrack", run_id="run_missing", issue_id="HP-88", json=True))

    assert rc == 1
    capsys.readouterr()
    assert not db_path.exists()


def test_runtime_sync_youtrack_write_passes_explicit_issue_stage_and_ytctl(capsys, monkeypatch):
    from agent_runtime import db
    from hermes_cli import runtime as runtime_module

    observed = {}

    def fake_sync(conn, run_id, **kwargs):
        observed.update(kwargs)
        return {
            "success": True,
            "run_id": run_id,
            "issue_id": kwargs["issue_id"],
            "written": kwargs["write"],
            "operations": ["comment", "stage"],
            "stage": kwargs["stage"],
            "comment": "safe public mirror",
            "commands": [],
            "results": [],
            "mirror_only": True,
        }

    monkeypatch.setattr(runtime_module.youtrack_sync, "sync_run_to_youtrack", fake_sync)
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime YouTrack CLI", public_ref="HP-88")

    rc = runtime_command(
        _args(
            runtime_command="sync-youtrack",
            run_id=run_id,
            issue_id="HP-89",
            stage="Review",
            ytctl="/tmp/ytctl",
            write=True,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] is True
    assert payload["issue_id"] == "HP-89"
    assert observed["issue_id"] == "HP-89"
    assert observed["stage"] == "Review"
    assert observed["write"] is True
    assert observed["ytctl"] == "/tmp/ytctl"
