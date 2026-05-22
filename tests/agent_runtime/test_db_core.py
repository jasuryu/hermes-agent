"""Tests for the Agent Runtime durable state layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime import db


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


def test_init_db_creates_runtime_tables(runtime_home):
    with db.connect() as conn:
        names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        }

    assert {
        "runtime_runs",
        "runtime_jobs",
        "runtime_job_dependencies",
        "runtime_attempts",
        "runtime_events",
        "runtime_artifacts",
        "runtime_findings",
        "runtime_approvals",
        "runtime_decisions",
    } <= names
    assert db.runtime_db_path() == runtime_home / "agent-runtime" / "runtime.db"


def test_create_run_job_dependency_and_promote(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title="Final runtime",
            objective="Build orchestrator-centered runtime",
            owner_source="telegram:test",
            public_ref="HP-88",
        )
        parent = db.create_job(
            conn,
            run_id=run_id,
            role="explorer",
            title="Discovery",
            body="Collect facts",
        )
        child = db.create_job(
            conn,
            run_id=run_id,
            role="code_worker",
            title="Implement core",
            body="Write code",
            depends_on=[parent],
        )

        assert db.get_run(conn, run_id).public_ref == "HP-88"
        assert db.get_job(conn, parent).status == "ready"
        assert db.get_job(conn, child).status == "planned"

        claim = db.claim_next_job(conn, lease_owner="runtime-daemon", now=1_000)
        assert claim is not None
        assert claim.job_id == parent
        db.complete_job(
            conn,
            parent,
            summary="discovery done",
            lease_owner="runtime-daemon",
            attempt_id=claim.attempt_id,
            now=1_001,
        )
        promoted = db.promote_ready_jobs(conn, run_id=run_id)

        assert promoted == [child]
        assert db.get_job(conn, child).status == "ready"
        events = db.list_events(conn, run_id=run_id)
        assert [event.kind for event in events] == [
            "run_created",
            "job_created",
            "job_created",
            "job_leased",
            "job_succeeded",
            "job_ready",
        ]


def test_claim_heartbeat_and_expired_lease_recovery(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Lease run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Lease me")

        claimed = db.claim_next_job(
            conn,
            lease_owner="runtime-daemon",
            lease_ttl_seconds=10,
            now=1_000,
        )
        assert claimed is not None
        assert claimed.job_id == job_id
        assert db.get_job(conn, job_id).status == "leased"
        assert db.claim_next_job(conn, lease_owner="other", now=1_001) is None

        db.heartbeat(conn, job_id, lease_owner="runtime-daemon", attempt_id=claimed.attempt_id, lease_ttl_seconds=10, now=1_005)
        assert db.get_job(conn, job_id).lease_expires_at == 1_015

        recovered = db.recover_expired_leases(conn, now=1_016)
        assert recovered == [job_id]
        assert db.get_job(conn, job_id).status == "ready"
        assert db.get_job(conn, job_id).lease_owner is None


def test_heartbeat_after_expiry_is_rejected(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Heartbeat run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Heartbeat")
        claimed = db.claim_next_job(conn, lease_owner="runtime-daemon", lease_ttl_seconds=1, now=100)
        assert claimed is not None

        with pytest.raises(ValueError, match="lease"):
            db.heartbeat(conn, job_id, lease_owner="runtime-daemon", attempt_id=claimed.attempt_id, now=102)


def test_approval_packets_are_exact_scope_runtime_state(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval run")
        packet = policy.build_approval_packet(
            target="cluster=whale namespace=prod deploy/api",
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
        )
        approval_id = _insert_approval_packet(conn, run_id=run_id, packet=packet, now=2_000)

        approval = db.find_approval_for_command(
            conn,
            run_id=run_id,
            target="cluster=whale namespace=prod deploy/api",
            command=command,
            now=2_001,
        )
        assert approval is not None
        assert approval["id"] == approval_id
        assert db.find_approval_for_command(
            conn,
            run_id=run_id,
            target="cluster=whale namespace=prod deploy/api",
            command="kubectl -n prod delete deploy/api",
            now=2_001,
        ) is None


def test_record_approval_rejects_env_only_writer_forgery(runtime_home, monkeypatch):
    from agent_runtime import policy

    monkeypatch.setenv("HERMES_AGENT_RUNTIME_APPROVAL_WRITER", "1")
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_APPROVAL_NONCE", "nonce")
    command = "kubectl -n prod rollout restart deploy/api"
    packet = policy.build_approval_packet(
        target="cluster=whale namespace=prod deploy/api",
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
    )
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval gate run")
        with pytest.raises(PermissionError, match="not authorized"):
            db.record_approval(conn, run_id=run_id, packet=packet)


def test_record_approval_rejects_forged_and_direct_imported_writers(runtime_home):
    from agent_runtime import approval_channel, policy

    assert not hasattr(approval_channel, "operator_approval_writer")
    assert not hasattr(approval_channel, "_OPERATOR_APPROVAL_WRITER")
    assert not hasattr(approval_channel, "_operator_approval_writer_for_cli")

    command = "kubectl -n prod rollout restart deploy/api"
    packet = policy.build_approval_packet(
        target="cluster=whale namespace=prod deploy/api",
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
    )

    class FakeWriter:
        channel = "runtime-operator-cli"

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Trusted approval writer run")
        with pytest.raises(PermissionError, match="not authorized"):
            db.record_approval(conn, run_id=run_id, packet=packet, approval_writer=FakeWriter())


def test_record_approval_rejects_mirror_source_labels(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    packet = policy.build_approval_packet(
        target="cluster=whale namespace=prod deploy/api",
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
        approval_source="youtrack",
    )
    with pytest.raises(ValueError, match="approval_source"):
        db.validate_approval_packet(packet)


def test_record_approval_requires_strict_trusted_source_allowlist(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    packet = policy.build_approval_packet(
        target="cluster=whale namespace=prod deploy/api",
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
        approval_source="operator-fake",
    )
    with pytest.raises(ValueError, match="trusted operator channel"):
        db.validate_approval_packet(packet)


def test_record_approval_rejects_incomplete_audit_packet(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    packet = policy.build_approval_packet(
        target="cluster=whale namespace=prod deploy/api",
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
        approval_source="operator-cli",
    )
    packet["verification"] = ["   "]
    with pytest.raises(ValueError, match="verification"):
        db.validate_approval_packet(packet)


def test_find_approval_requires_matching_target_and_scope(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    wrong_target = "cluster=whale namespace=stage deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval scope run")
        packet = policy.build_approval_packet(
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
        )
        _insert_approval_packet(conn, run_id=run_id, packet=packet, now=2_000)

        assert db.find_approval_for_command(conn, run_id=run_id, target=wrong_target, command=command, now=2_001) is None
        assert db.find_approval_for_command(
            conn,
            run_id=run_id,
            target=target,
            command=command,
            scope_hash="wrong-scope",
            now=2_001,
        ) is None
        assert db.find_approval_for_command(
            conn,
            run_id=run_id,
            target=target,
            command=command,
            scope_hash=packet["scope_hash"],
            now=2_001,
        ) is not None


def test_find_approval_for_command_honors_job_scope(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval job scope run")
        job_a = db.create_job(conn, run_id=run_id, role="ops_worker", title="Job A")
        job_b = db.create_job(conn, run_id=run_id, role="ops_worker", title="Job B")
        packet = policy.build_approval_packet(
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
        )
        approval_id = _insert_approval_packet(conn, run_id=run_id, job_id=job_a, packet=packet, now=2_050)

        scoped = db.find_approval_for_command(
            conn,
            run_id=run_id,
            job_id=job_a,
            target=target,
            command=command,
            now=2_051,
        )
        assert scoped is not None
        assert scoped["id"] == approval_id
        assert db.find_approval_for_command(
            conn,
            run_id=run_id,
            job_id=job_b,
            target=target,
            command=command,
            now=2_051,
        ) is None
        assert db.find_approval_for_command(
            conn,
            run_id=run_id,
            target=target,
            command=command,
            now=2_051,
        ) is None


def test_find_approval_for_command_run_level_visible_with_or_without_job(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval run scope run")
        job_id = db.create_job(conn, run_id=run_id, role="ops_worker", title="Job")
        packet = policy.build_approval_packet(
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
        )
        approval_id = _insert_approval_packet(conn, run_id=run_id, packet=packet, now=2_060)

        without_job = db.find_approval_for_command(
            conn,
            run_id=run_id,
            target=target,
            command=command,
            now=2_061,
        )
        with_job = db.find_approval_for_command(
            conn,
            run_id=run_id,
            job_id=job_id,
            target=target,
            command=command,
            now=2_061,
        )
        assert without_job is not None
        assert without_job["id"] == approval_id
        assert with_job is not None
        assert with_job["id"] == approval_id


def test_find_approval_rejects_malformed_scope_hash_even_when_not_supplied(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval malformed scope run")
        packet = policy.build_approval_packet(
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
        )
        packet["scope_hash"] = "forged-scope"
        _insert_approval_packet(conn, run_id=run_id, packet=packet, now=2_100)

        assert db.find_approval_for_command(
            conn,
            run_id=run_id,
            target=target,
            command=command,
            now=2_101,
        ) is None


def test_find_approval_rejects_malformed_command_hashes(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    injected = "kubectl -n prod delete deploy/api"
    target = "cluster=whale namespace=prod deploy/api"
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Approval malformed hashes run")
        packet = policy.build_approval_packet(
            target=target,
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
        )
        packet["command_hashes"] = packet["command_hashes"] + [policy.command_hash(injected)]
        _insert_approval_packet(conn, run_id=run_id, packet=packet, now=2_200)

        assert db.find_approval_for_command(
            conn,
            run_id=run_id,
            target=target,
            command=injected,
            now=2_201,
        ) is None


def test_close_run_records_terminal_summary(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Close run")
        db.close_run(conn, run_id=run_id, status="done", summary="MVP completed", now=3_000)

        run = db.get_run(conn, run_id)
        assert run.status == "done"
        assert run.summary == "MVP completed"
        assert run.closed_at == 3_000
        assert db.list_events(conn, run_id=run_id)[-1].kind == "run_closed"


def test_close_run_records_reason_on_active_attempts(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Close leased run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Leased job")
        claim = db.claim_next_job(conn, lease_owner="worker", now=100)
        assert claim is not None

        db.close_run(conn, run_id=run_id, status="cancelled", now=101)

        attempt = conn.execute("SELECT status, error FROM runtime_attempts WHERE id=?", (claim.attempt_id,)).fetchone()
        assert attempt["status"] == "cancelled"
        assert attempt["error"] == "run closed: cancelled"
        assert db.get_job(conn, job_id).status == "cancelled"


def test_close_run_rejects_nonterminal_attention_status(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Attention is active")

        with pytest.raises(ValueError, match="terminal"):
            db.close_run(conn, run_id=run_id, status="attention")


def test_cross_run_dependencies_are_rejected(runtime_home):
    with db.connect() as conn:
        run_a = db.create_run(conn, title="A")
        run_b = db.create_run(conn, title="B")
        parent = db.create_job(conn, run_id=run_a, role="explorer", title="Parent")

        with pytest.raises(ValueError, match="same run"):
            db.create_job(conn, run_id=run_b, role="code_worker", title="Bad child", depends_on=[parent])


def test_create_job_rejects_terminal_runs(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Closed")
        db.close_run(conn, run_id=run_id, status="done")

        with pytest.raises(ValueError, match="terminal run"):
            db.create_job(conn, run_id=run_id, role="code_worker", title="Too late")


def test_orchestrator_cannot_be_created_as_runtime_job(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="No recursive orchestrator")

        with pytest.raises(ValueError, match="worker role"):
            db.create_job(conn, run_id=run_id, role="orchestrator", title="Do everything")


def test_role_aliases_are_stored_canonically(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Canonical role")
        job_id = db.create_job(conn, run_id=run_id, role="code-worker", title="Alias")

        assert db.get_job(conn, job_id).role == "code_worker"


def test_invalid_max_attempts_is_rejected(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Attempts")

        with pytest.raises(ValueError, match="max_attempts"):
            db.create_job(conn, run_id=run_id, role="code_worker", title="Bad", max_attempts=0)


def test_optional_job_references_must_belong_to_same_run(runtime_home):
    from agent_runtime import policy

    command = "kubectl -n prod rollout restart deploy/api"
    with db.connect() as conn:
        run_a = db.create_run(conn, title="A")
        job_a = db.create_job(conn, run_id=run_a, role="code_worker", title="A job")
        run_b = db.create_run(conn, title="B")
        packet = policy.build_approval_packet(
            target="cluster=whale namespace=prod deploy/api",
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
        )

        with pytest.raises(ValueError, match="same run"):
            db.add_finding(conn, run_id=run_b, job_id=job_a, severity="low", category="test", summary="bad")
        with pytest.raises(ValueError, match="same run"):
            db.record_decision(conn, run_id=run_b, job_id=job_a, kind="bad")


def test_record_decision_rejects_cross_run_linked_findings(runtime_home):
    with db.connect() as conn:
        run_a = db.create_run(conn, title="A")
        finding_a = db.add_finding(conn, run_id=run_a, severity="medium", category="review", summary="A finding")
        run_b = db.create_run(conn, title="B")

        with pytest.raises(ValueError, match="same run"):
            db.record_decision(conn, run_id=run_b, kind="accept_risk", linked_findings=[finding_a])


def test_claim_skips_exhausted_jobs_and_closed_runs(runtime_home):
    with db.connect() as conn:
        closed_run = db.create_run(conn, title="Closed")
        closed_job = db.create_job(conn, run_id=closed_run, role="code_worker", title="Closed job")
        db.close_run(conn, run_id=closed_run, status="done")

        active_run = db.create_run(conn, title="Active")
        exhausted_job = db.create_job(conn, run_id=active_run, role="code_worker", title="Exhausted", max_attempts=1)
        assert db.claim_next_job(conn, lease_owner="worker", now=10) is not None
        db.recover_expired_leases(conn, now=10_000)

        assert db.claim_next_job(conn, lease_owner="worker", now=10_001) is None
        assert db.get_job(conn, closed_job).status == "cancelled"
        assert db.get_job(conn, exhausted_job).attempt_count == 1


def test_heartbeat_rejects_closed_run(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Closed heartbeat")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Worker")
        claim = db.claim_next_job(conn, lease_owner="worker", lease_ttl_seconds=10, now=100)
        assert claim is not None
        db.close_run(conn, run_id=run_id, status="cancelled", now=101)

        with pytest.raises(ValueError, match="active"):
            db.heartbeat(conn, job_id, lease_owner="worker", attempt_id=claim.attempt_id, now=102)


def test_recover_expired_leases_does_not_resurrect_closed_run_jobs(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Closed recovery")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Worker")
        claim = db.claim_next_job(conn, lease_owner="worker", lease_ttl_seconds=1, now=100)
        assert claim is not None
        db.close_run(conn, run_id=run_id, status="cancelled", now=101)

        assert db.recover_expired_leases(conn, now=102) == []
        assert db.get_job(conn, job_id).status == "cancelled"


def test_claim_rechecks_run_status_during_update(runtime_home, monkeypatch):
    original_job_from_row = db._job_from_row
    closed = {"done": False}

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Race run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Race job")
        conn.commit()

        def close_run_after_select(row):
            job = original_job_from_row(row)
            if job is not None and not closed["done"]:
                with db.connect() as other_conn:
                    db.close_run(other_conn, run_id=run_id, status="done")
                closed["done"] = True
            return job

        monkeypatch.setattr(db, "_job_from_row", close_run_after_select)
        claim = db.claim_next_job(conn, lease_owner="worker", now=11)
        monkeypatch.setattr(db, "_job_from_row", original_job_from_row)

        assert claim is None
        assert db.get_job(conn, job_id).status == "cancelled"


def test_complete_job_rejects_terminal_job_state(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Terminal job run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Failed job")
        conn.execute("UPDATE runtime_jobs SET status='failed' WHERE id=?", (job_id,))

        with pytest.raises(ValueError, match="completion"):
            db.complete_job(conn, job_id)


def test_complete_job_rejects_ready_job_without_active_attempt(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Ready completion run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Ready job")

        with pytest.raises(ValueError, match="lease|attempt|completion"):
            db.complete_job(conn, job_id, summary="manual shortcut")

        assert db.get_job(conn, job_id).status == "ready"


def test_complete_job_rejects_stale_lease_identity_after_recovery(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Recovered lease run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Recovered job")
        claim = db.claim_next_job(conn, lease_owner="worker-a", lease_ttl_seconds=1, now=100)
        assert claim is not None
        assert db.recover_expired_leases(conn, now=102) == [job_id]

        with pytest.raises(ValueError, match="lease|completion"):
            db.complete_job(conn, job_id, lease_owner="worker-a", attempt_id=claim.attempt_id, now=103)

        assert db.get_job(conn, job_id).status == "ready"


def test_complete_job_rejects_planned_dependency_children(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Dependency run")
        parent_id = db.create_job(conn, run_id=run_id, role="explorer", title="Parent")
        child_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Child", depends_on=[parent_id])

        assert db.get_job(conn, child_id).status == "planned"
        with pytest.raises(ValueError, match="completion"):
            db.complete_job(conn, child_id)

        assert db.get_job(conn, child_id).status == "planned"
