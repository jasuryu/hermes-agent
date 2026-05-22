"""Tests for mandatory Agent Runtime ops command guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime import db, ops_guard, policy, worker_broker


@pytest.fixture
def runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db.init_db()
    return home


def _insert_approval_packet(conn, *, run_id: str, packet: dict, job_id: str | None = None, now: int = 2_000) -> str:
    approval_id = f"appr_test_{now}"
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
            job_id,
            packet["target"],
            json.dumps(packet["commands"]),
            json.dumps(packet["command_hashes"]),
            packet["reason"],
            packet["blast_radius"],
            packet["rollback"],
            json.dumps(packet["verification"]),
            packet["approved_by"],
            packet["approval_source"],
            int(packet["approved_at"]),
            packet.get("expires_at"),
            packet["scope_hash"],
        ),
    )
    return approval_id


def _ops_context(runtime_home, *, approval: dict | None = None, approval_job_id: str | None = None):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Ops guard run")
        job_id = db.create_job(conn, run_id=run_id, role="ops_worker", title="Ops guard job")
        claim = db.claim_next_job(conn, lease_owner="ops-worker", now=3_001)
        assert claim is not None
        if approval is not None:
            scoped_job_id = approval_job_id
            if approval_job_id == "__other__":
                scoped_job_id = db.create_job(conn, run_id=run_id, role="ops_worker", title="Other ops job")
            _insert_approval_packet(conn, run_id=run_id, job_id=scoped_job_id, packet=approval, now=3_000)
        context = worker_broker.build_worker_context(
            conn,
            job_id=job_id,
            lease_owner="ops-worker",
            attempt_id=claim.attempt_id,
            now=3_002,
        )
    return context, job_id


def test_ops_guard_allows_readonly_discovery_without_approval(runtime_home):
    context, _job_id = _ops_context(runtime_home)

    result = ops_guard.guard_ops_command(
        context,
        command="kubectl auth can-i get pods",
        target="cluster=whale namespace=prod",
        now=3_003,
    )

    assert result.allowed is True
    assert result.requires_approval is False
    assert result.approval_id == ""


def test_ops_guard_blocks_missing_context_expiry_even_for_readonly(runtime_home):
    context, _job_id = _ops_context(runtime_home)
    context.pop("expires_at", None)

    result = ops_guard.guard_ops_command(
        context,
        command="kubectl auth can-i get pods",
        target="cluster=whale namespace=prod",
        now=3_003,
    )

    assert result.allowed is False
    assert result.category == "invalid_context"
    assert "expires" in result.reason


def test_ops_guard_blocks_expired_context_before_runner(runtime_home):
    context, _job_id = _ops_context(runtime_home)
    context["expires_at"] = 3_003

    def forbidden_runner(*_args, **_kwargs):
        raise AssertionError("expired context must block before execution")

    result = ops_guard.guarded_ops_terminal(
        context,
        command="kubectl auth can-i get pods",
        target="cluster=whale namespace=prod",
        runner=forbidden_runner,
        now=3_003,
    )

    assert result["status"] == "blocked"
    assert result["guard"]["allowed"] is False
    assert result["guard"]["category"] == "invalid_context"
    assert "expired" in result["error"]


def test_ops_guard_blocks_sensitive_read_without_exact_approval(runtime_home):
    context, _job_id = _ops_context(runtime_home)

    result = ops_guard.guard_ops_command(
        context,
        command="kubectl -n prod get secrets",
        target="cluster=whale namespace=prod",
        now=3_003,
    )

    assert result.allowed is False
    assert result.requires_approval is True
    assert "approval" in result.reason


def test_ops_guard_allows_only_exact_command_and_target_from_context_snapshot(runtime_home):
    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    approval = policy.build_approval_packet(
        target=target,
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
        expires_at=9_999_999_999,
    )
    context, _job_id = _ops_context(runtime_home, approval=approval)

    allowed = ops_guard.guard_ops_command(context, command=command, target=target, now=3_003)
    wrong_command = ops_guard.guard_ops_command(context, command="kubectl -n prod delete deploy/api", target=target, now=3_003)
    wrong_target = ops_guard.guard_ops_command(context, command=command, target="cluster=whale namespace=stage deploy/api", now=3_003)

    assert allowed.allowed is True
    assert allowed.approval_id.startswith("appr_test_")
    assert wrong_command.allowed is False
    assert wrong_target.allowed is False


def test_ops_guard_rejects_inactive_approval_snapshot(runtime_home):
    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    approval = policy.build_approval_packet(
        target=target,
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
        expires_at=9_999_999_999,
    )
    context, _job_id = _ops_context(runtime_home, approval=approval)
    context["approvals"][0]["status"] = "revoked"

    result = ops_guard.guard_ops_command(context, command=command, target=target, now=3_003)

    assert result.allowed is False
    assert "approval" in result.reason


def test_ops_guard_does_not_reuse_job_scoped_approval_for_other_job(runtime_home):
    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    approval = policy.build_approval_packet(
        target=target,
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
        expires_at=9_999_999_999,
    )
    context, job_id = _ops_context(runtime_home, approval=approval, approval_job_id="__other__")
    assert context["job"]["id"] == job_id

    result = ops_guard.guard_ops_command(context, command=command, target=target, now=3_003)

    assert context["approvals"] == []
    assert result.allowed is False
    assert "approval" in result.reason


def test_guarded_ops_terminal_runs_only_after_guard_allows(runtime_home):
    context, _job_id = _ops_context(runtime_home)
    calls = []

    def fake_runner(argv, *, cwd, timeout, env):
        calls.append({"argv": argv, "cwd": cwd, "timeout": timeout, "env": env})
        return ops_guard.OpsCommandExecution(exit_code=0, output="ok", error="")

    result = ops_guard.guarded_ops_terminal(
        context,
        command="kubectl auth can-i get pods",
        target="cluster=whale namespace=prod",
        runner=fake_runner,
        now=3_003,
    )

    assert result["status"] == "ok"
    assert result["guard"]["allowed"] is True
    assert calls[0]["argv"] == ["kubectl", "auth", "can-i", "get", "pods"]


def test_guarded_ops_terminal_refuses_compound_shell_even_with_runner(runtime_home):
    context, _job_id = _ops_context(runtime_home)

    def forbidden_runner(*_args, **_kwargs):
        raise AssertionError("blocked command must not execute")

    result = ops_guard.guarded_ops_terminal(
        context,
        command="kubectl get pods && kubectl delete pod/api",
        target="cluster=whale namespace=prod",
        runner=forbidden_runner,
        now=3_003,
    )

    assert result["status"] == "blocked"
    assert result["guard"]["allowed"] is False
