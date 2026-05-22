"""Tests for the Agent Runtime worker entrypoint skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime import db, worker_broker, worker_main


@pytest.fixture
def runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db.init_db()
    return home


def test_worker_main_skeleton_refuses_to_execute_without_mutating_job(runtime_home, capsys, monkeypatch):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Worker run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Worker job")
        claim = db.claim_next_job(conn, lease_owner="test-worker")
        assert claim is not None
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", claim.attempt_id)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "test-worker")

    assert worker_main.main(["--job", job_id]) == 1

    with db.connect() as conn:
        job = db.get_job(conn, job_id)
        attempt = conn.execute("SELECT status, error FROM runtime_attempts WHERE job_id=?", (job_id,)).fetchone()
        events = db.list_events(conn, job_id=job_id)

    assert job.status == "leased"
    assert attempt["status"] == "running"
    assert not any(event.kind == "job_succeeded" for event in events)
    out = capsys.readouterr()
    assert "worker_execution_disabled" in out.out
    assert "Worker run" not in out.out
    assert "Worker job" not in out.out


def test_worker_main_placeholder_never_exposes_db_context_even_with_lease_env(runtime_home, capsys, monkeypatch):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Hidden leased worker run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Hidden leased worker job")
        claim = db.claim_next_job(conn, lease_owner="real-worker")
        assert claim is not None
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", claim.attempt_id)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "real-worker")

    assert worker_main.main(["--job", job_id]) == 1

    out = capsys.readouterr()
    assert "Hidden leased worker run" not in out.out
    assert "Hidden leased worker job" not in out.out
    assert "worker_execution_disabled" in out.out


def test_worker_main_rejects_stale_lease_identity(runtime_home, monkeypatch):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Stale worker run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Worker job")
        claim = db.claim_next_job(conn, lease_owner="test-worker", lease_ttl_seconds=1, now=100)
        assert claim is not None
        db.recover_expired_leases(conn, now=102)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", claim.attempt_id)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "test-worker")

    assert worker_main.main(["--job", job_id]) == 1

    with db.connect() as conn:
        job = db.get_job(conn, job_id)
    assert job.status == "ready"


def test_worker_main_rejects_unleased_job_without_context(runtime_home, capsys):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Unleased worker run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Unleased worker job")

    assert worker_main.main(["--job", job_id]) == 1

    out = capsys.readouterr()
    assert "Unleased worker run" not in out.out
    assert "Unleased worker job" not in out.out
    assert "lease" in out.err


def test_worker_main_rejects_foreign_lease_identity_without_context(runtime_home, capsys, monkeypatch):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Foreign worker run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Worker job")
        claim = db.claim_next_job(conn, lease_owner="real-worker")
        assert claim is not None
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", claim.attempt_id)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "foreign-worker")

    assert worker_main.main(["--job", job_id]) == 1

    out = capsys.readouterr()
    assert "Foreign worker run" not in out.out
    assert "Worker job" not in out.out
    assert "worker_execution_disabled" in out.out


def test_worker_main_runs_role_specific_agent_when_execution_gate_enabled(runtime_home, capsys, monkeypatch):
    observed: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            observed["agent_kwargs"] = kwargs

        def run_conversation(self, prompt):
            observed["prompt"] = prompt
            return {"final_response": "completed worker task"}

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Executable run", objective="Deliver safe worker execution")
        job_id = db.create_job(
            conn,
            run_id=run_id,
            role="code_worker",
            title="Implement feature",
            body="Change only the bounded repo and run tests.",
        )
        claim = db.claim_next_job(conn, lease_owner="exec-worker")
        assert claim is not None
        bundle = worker_broker.materialize_worker_context(
            conn,
            job_id=job_id,
            lease_owner="exec-worker",
            attempt_id=claim.attempt_id,
            hermes_home=runtime_home,
        )

    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", claim.attempt_id)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "exec-worker")
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_CONTEXT", str(bundle.context_path))
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION", "1")

    assert worker_main.main(["--job", job_id], agent_factory=FakeAgent) == 0

    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["success"] is True
    assert payload["role"] == "code_worker"
    assert payload["summary"] == "completed worker task"
    assert observed["agent_kwargs"]["model"] == "gpt-5.3-codex"
    assert observed["agent_kwargs"]["enabled_toolsets"] == ["terminal", "file", "code_execution", "git"]
    assert observed["agent_kwargs"]["skip_memory"] is True
    assert observed["agent_kwargs"]["skip_context_files"] is True
    assert "Implement feature" in observed["prompt"]
    assert "Change only the bounded repo" in observed["prompt"]
    with db.connect() as conn:
        assert db.get_job(conn, job_id).status == "leased"


def test_worker_main_rejects_context_identity_mismatch_without_exposing_context(runtime_home, capsys, monkeypatch):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Mismatch hidden run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Mismatch hidden job")
        claim = db.claim_next_job(conn, lease_owner="exec-worker")
        assert claim is not None
        bundle = worker_broker.materialize_worker_context(
            conn,
            job_id=job_id,
            lease_owner="exec-worker",
            attempt_id=claim.attempt_id,
            hermes_home=runtime_home,
        )

    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", "wrong-attempt")
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "exec-worker")
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_CONTEXT", str(bundle.context_path))
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION", "1")

    assert worker_main.main(["--job", job_id]) == 1

    out = capsys.readouterr()
    assert "Mismatch hidden run" not in out.out
    assert "Mismatch hidden job" not in out.out
    assert "context identity" in out.err


def test_worker_main_runs_ops_worker_with_mandatory_command_guard_toolset(runtime_home, capsys, monkeypatch):
    observed: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            observed["agent_kwargs"] = kwargs

        def run_conversation(self, prompt):
            observed["prompt"] = prompt
            return {"final_response": "ops discovery complete"}

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Ops guarded run")
        job_id = db.create_job(conn, run_id=run_id, role="ops_worker", title="Inspect prod")
        claim = db.claim_next_job(conn, lease_owner="exec-worker")
        assert claim is not None
        bundle = worker_broker.materialize_worker_context(
            conn,
            job_id=job_id,
            lease_owner="exec-worker",
            attempt_id=claim.attempt_id,
            hermes_home=runtime_home,
        )

    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ATTEMPT_ID", claim.attempt_id)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_LEASE_OWNER", "exec-worker")
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_CONTEXT", str(bundle.context_path))
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION", "1")

    assert worker_main.main(["--job", job_id], agent_factory=FakeAgent) == 0

    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["success"] is True
    assert payload["role"] == "ops_worker"
    assert observed["agent_kwargs"]["enabled_toolsets"] == ["ops_terminal", "monitoring", "logs", "file_readonly"]
    assert "approval packet" in observed["agent_kwargs"]["ephemeral_system_prompt"]
    assert "Inspect prod" in observed["prompt"]
