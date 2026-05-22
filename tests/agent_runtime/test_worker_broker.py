"""Tests for trusted Agent Runtime worker context broker primitives."""

from __future__ import annotations

import json
import os
import subprocess
import stat
from pathlib import Path

import pytest

from agent_runtime import db, worker_broker


@pytest.fixture
def runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db.init_db()
    return home


def test_worker_broker_materializes_sanitized_context_for_active_lease(runtime_home, tmp_path):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Broker run", objective="Do bounded work", public_ref="HP-88", now=100)
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Broker job", body="Implement safe slice", now=101)
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", now=102)
        assert claim is not None
        bundle = worker_broker.materialize_worker_context(
            conn,
            job_id=job_id,
            lease_owner="trusted-daemon",
            attempt_id=claim.attempt_id,
            workspace_root=tmp_path / "workers",
            hermes_home=runtime_home,
            now=103,
        )

    assert bundle.sandbox.root.exists()
    assert bundle.sandbox.root.name.startswith("workers-")
    assert not bundle.sandbox.root.resolve().is_relative_to(runtime_home.resolve())
    assert bundle.context_path.exists()
    assert stat.S_IMODE(bundle.context_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(bundle.sandbox.root.stat().st_mode) == 0o700

    payload = json.loads(bundle.context_path.read_text())
    assert payload["run"]["id"] == run_id
    assert payload["run"]["public_ref"] == "HP-88"
    assert payload["job"]["id"] == job_id
    assert payload["job"]["role"] == "code_worker"
    assert payload["lease"]["attempt_id"] == claim.attempt_id
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert "db_path" not in serialized
    assert "hermes_home" not in serialized
    assert "runtime.db" not in serialized
    assert "approval_writer" not in serialized


def test_worker_broker_rejects_foreign_and_expired_lease(runtime_home, tmp_path):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Lease broker run", now=200)
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Lease job", now=201)
        claim = db.claim_next_job(conn, lease_owner="real-daemon", lease_ttl_seconds=1, now=202)
        assert claim is not None

        with pytest.raises(ValueError, match="active lease"):
            worker_broker.build_worker_context(
                conn,
                job_id=job_id,
                lease_owner="foreign-daemon",
                attempt_id=claim.attempt_id,
                now=202,
            )
        with pytest.raises(ValueError, match="active lease"):
            worker_broker.build_worker_context(
                conn,
                job_id=job_id,
                lease_owner="real-daemon",
                attempt_id=claim.attempt_id,
                now=204,
            )


def test_worker_broker_rejects_sandbox_under_hermes_home(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Sandbox run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Sandbox job")
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon")
        assert claim is not None
        with pytest.raises(ValueError, match="must not live under HERMES_HOME"):
            worker_broker.materialize_worker_context(
                conn,
                job_id=job_id,
                lease_owner="trusted-daemon",
                attempt_id=claim.attempt_id,
                workspace_root=runtime_home / "agent-runtime" / "workers",
                hermes_home=runtime_home,
            )


def test_worker_broker_creates_unique_private_sandbox_per_call(runtime_home, tmp_path):
    first = worker_broker.create_worker_sandbox(
        workspace_root=tmp_path / "workers",
        job_id="job_same",
        attempt_id="att_same",
        hermes_home=runtime_home,
    )
    second = worker_broker.create_worker_sandbox(
        workspace_root=tmp_path / "workers",
        job_id="job_same",
        attempt_id="att_same",
        hermes_home=runtime_home,
    )

    assert first.root != second.root
    assert stat.S_IMODE(first.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(second.root.stat().st_mode) == 0o700


def test_worker_broker_rejects_symlink_workspace_root(runtime_home, tmp_path):
    target = tmp_path / "target-workers"
    target.mkdir()
    link = tmp_path / "workers-link"
    os.symlink(target, link, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        worker_broker.create_worker_sandbox(
            workspace_root=link,
            job_id="job_123",
            attempt_id="att_123",
            hermes_home=runtime_home,
        )


def test_operator_approval_reaches_matching_worker_context(runtime_home):
    from agent_runtime import ops_guard, policy

    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval context run", now=250)
        job_id = db.create_job(conn, run_id=run_id, role="ops_worker", title="Restart api", now=251)
        packet = policy.build_approval_packet(
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
            approval_source="operator-cli",
        )
        conn.execute(
            """
            INSERT INTO runtime_approvals
            (id, run_id, job_id, target, commands_json, command_hashes_json, reason,
             blast_radius, rollback, verification_json, approved_by, approval_source,
             approved_at, expires_at, scope_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                "appr_operator_context",
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
                252,
                packet.get("expires_at"),
                packet["scope_hash"],
            ),
        )
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", now=253)
        assert claim is not None
        context = worker_broker.build_worker_context(
            conn,
            job_id=job_id,
            lease_owner="trusted-daemon",
            attempt_id=claim.attempt_id,
            now=254,
        )

    assert len(context["approvals"]) == 1
    guard = ops_guard.guard_ops_command(context, command=command, target=target, now=254)
    assert guard.allowed is True
    assert guard.approval_id.startswith("appr_")


def test_worker_result_broker_completes_successful_worker_stdout(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Result run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Result job")
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", now=300)
        assert claim is not None

        record = worker_broker.record_worker_result(
            conn,
            job_id=job_id,
            lease_owner="trusted-daemon",
            attempt_id=claim.attempt_id,
            exit_code=0,
            stdout=json.dumps({"success": True, "summary": "worker completed"}),
            stderr="",
            now=301,
        )

        assert record.success is True
        job = db.get_job(conn, job_id)
        attempt = conn.execute("SELECT status, summary FROM runtime_attempts WHERE id=?", (claim.attempt_id,)).fetchone()
        assert job.status == "succeeded"
        assert job.result_summary == "worker completed"
        assert attempt["status"] == "succeeded"
        assert attempt["summary"] == "worker completed"


def test_worker_result_broker_fails_nonzero_or_malformed_results(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Bad result run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Bad result job", max_attempts=1)
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", now=400)
        assert claim is not None

        record = worker_broker.record_worker_result(
            conn,
            job_id=job_id,
            lease_owner="trusted-daemon",
            attempt_id=claim.attempt_id,
            exit_code=1,
            stdout=json.dumps({"success": True, "summary": "do not trust nonzero"}),
            stderr="Traceback: boom",
            now=401,
        )

        assert record.success is False
        job = db.get_job(conn, job_id)
        attempt = conn.execute("SELECT status, error FROM runtime_attempts WHERE id=?", (claim.attempt_id,)).fetchone()
        assert job.status == "failed"
        assert "exit code 1" in attempt["error"]
        assert "Traceback" in attempt["error"]


def test_worker_result_broker_rejects_stale_lease_without_mutation(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Stale result run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Stale result job")
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", lease_ttl_seconds=1, now=500)
        assert claim is not None
        db.recover_expired_leases(conn, now=502)

        with pytest.raises(ValueError, match="lease|active attempt"):
            worker_broker.record_worker_result(
                conn,
                job_id=job_id,
                lease_owner="trusted-daemon",
                attempt_id=claim.attempt_id,
                exit_code=0,
                stdout=json.dumps({"success": True, "summary": "too late"}),
                stderr="",
                now=503,
            )

        job = db.get_job(conn, job_id)
        assert job.status == "ready"
        assert job.result_summary != "too late"


def test_worker_result_broker_rejects_success_after_lease_expiry_without_recovery(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Expired success run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Expired success job")
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", lease_ttl_seconds=1, now=540)
        assert claim is not None

        with pytest.raises(ValueError, match="lease|active attempt"):
            worker_broker.record_worker_result(
                conn,
                job_id=job_id,
                lease_owner="trusted-daemon",
                attempt_id=claim.attempt_id,
                exit_code=0,
                stdout=json.dumps({"success": True, "summary": "too late success"}),
                stderr="",
                now=542,
            )

        job = db.get_job(conn, job_id)
        attempt = conn.execute("SELECT status, summary FROM runtime_attempts WHERE id=?", (claim.attempt_id,)).fetchone()
        assert job is not None
        assert job.status == "leased"
        assert job.result_summary != "too late success"
        assert attempt["status"] == "running"
        assert attempt["summary"] == ""


def test_worker_result_broker_rejects_direct_failure_after_lease_expiry(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Expired direct failure run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Expired direct failure job")
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", lease_ttl_seconds=1, now=560)
        assert claim is not None

        with pytest.raises(ValueError, match="lease|active attempt"):
            worker_broker.record_worker_result(
                conn,
                job_id=job_id,
                lease_owner="trusted-daemon",
                attempt_id=claim.attempt_id,
                exit_code=1,
                stdout=json.dumps({"success": False, "error": "late failure"}),
                stderr="late failure",
                now=562,
            )

        job = db.get_job(conn, job_id)
        attempt = conn.execute("SELECT status, error FROM runtime_attempts WHERE id=?", (claim.attempt_id,)).fetchone()
        assert job is not None
        assert job.status == "leased"
        assert job.result_summary == ""
        assert attempt["status"] == "running"
        assert attempt["error"] == ""


def test_reap_worker_process_records_success_via_parent_broker(runtime_home):
    class FakeProcess:
        returncode = 0

        def communicate(self, timeout=None):
            assert timeout == 10
            return json.dumps({"success": True, "summary": "reaped result"}), ""

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Reaper run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Reaper job")
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", now=600)
        assert claim is not None

        record = worker_broker.reap_worker_process(
            conn,
            process=FakeProcess(),
            job_id=job_id,
            lease_owner="trusted-daemon",
            attempt_id=claim.attempt_id,
            timeout=10,
            now=601,
        )

        assert record.success is True
        assert db.get_job(conn, job_id).status == "succeeded"


def test_reap_worker_process_kills_timeout_and_fails_attempt(runtime_home):
    class HangingProcess:
        returncode = None
        killed = False

        def communicate(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired(cmd=["worker"], timeout=float(timeout or 0))
            self.returncode = -9
            return "", "worker timed out"

        def kill(self):
            self.killed = True

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Timeout run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Timeout job", max_attempts=1)
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", now=700)
        assert claim is not None
        proc = HangingProcess()

        record = worker_broker.reap_worker_process(
            conn,
            process=proc,
            job_id=job_id,
            lease_owner="trusted-daemon",
            attempt_id=claim.attempt_id,
            timeout=1,
            now=701,
        )

        assert proc.killed is True
        assert record.success is False
        assert "timed out" in record.error
        assert db.get_job(conn, job_id).status == "failed"


def test_reap_worker_process_timeout_after_lease_expiry_fails_and_retries_cleanly(runtime_home):
    class HangingProcess:
        returncode = None
        killed = False

        def communicate(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired(cmd=["worker"], timeout=float(timeout or 0))
            self.returncode = -9
            return "", "worker timed out"

        def kill(self):
            self.killed = True

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Expired timeout run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Expired timeout job", max_attempts=2)
        claim = db.claim_next_job(conn, lease_owner="trusted-daemon", lease_ttl_seconds=1, now=800)
        assert claim is not None
        proc = HangingProcess()

        record = worker_broker.reap_worker_process(
            conn,
            process=proc,
            job_id=job_id,
            lease_owner="trusted-daemon",
            attempt_id=claim.attempt_id,
            timeout=1,
            now=802,
        )

        assert proc.killed is True
        assert record.success is False
        assert "timed out" in record.error
        job = db.get_job(conn, job_id)
        assert job is not None
        assert job.status == "ready"
        assert job.lease_owner is None
        assert job.attempt_count == 1
        attempt = conn.execute("SELECT status, error FROM runtime_attempts WHERE id=?", (claim.attempt_id,)).fetchone()
        assert attempt["status"] == "failed"
        assert "timed out" in attempt["error"]

        retry_claim = db.claim_next_job(conn, lease_owner="trusted-daemon", now=803)
        assert retry_claim is not None
        assert retry_claim.job_id == job_id
        assert retry_claim.attempt_id != claim.attempt_id
        retry_job = db.get_job(conn, job_id)
        assert retry_job is not None
        assert retry_job.attempt_count == 2
