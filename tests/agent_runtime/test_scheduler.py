"""Tests for Agent Runtime scheduler/spawner skeleton."""

from __future__ import annotations

import json
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from agent_runtime import db, scheduler, spawner, worker_broker


@pytest.fixture
def runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db.init_db()
    return home


def test_worker_invocation_is_scoped_by_runtime_env(runtime_home):
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=runtime_home.parent / "workers",
        job_id="job_123",
        attempt_id="att_123",
        hermes_home=runtime_home,
    )
    context_path = sandbox.root / "context.json"
    context_path.write_text("{}")
    context_path.chmod(0o600)
    invocation = spawner.build_worker_invocation(
        job_id="job_123",
        run_id="run_123",
        role="code_worker",
        attempt_id="att_123",
        lease_owner="daemon",
        context_path=context_path,
        sandbox=sandbox,
    )
    argv = list(invocation.argv)
    env = invocation.env

    assert argv[:2] == [spawner.python_executable(), "-c"]
    assert "runpy.run_module('agent_runtime.worker_main'" in argv[2]
    assert "job_123" in argv[2]
    assert invocation.cwd == sandbox.workdir
    assert env["HERMES_AGENT_RUNTIME_ATTEMPT_ID"] == "att_123"
    assert env["HERMES_AGENT_RUNTIME_LEASE_OWNER"] == "daemon"
    assert env["HERMES_AGENT_RUNTIME_CONTEXT"] == str(context_path)
    assert env["HOME"] == str(sandbox.home)
    assert env["TMPDIR"] == str(sandbox.tmp)
    assert env["XDG_CONFIG_HOME"] == str(sandbox.xdg_config_home)
    assert env["XDG_CACHE_HOME"] == str(sandbox.xdg_cache_home)
    assert "HERMES_AGENT_RUNTIME_JOB_ID" not in env
    assert "HERMES_AGENT_RUNTIME_RUN_ID" not in env
    assert "HERMES_AGENT_ROLE" not in env
    assert "HERMES_HOME" not in env
    assert "PATH" not in env
    assert "VIRTUAL_ENV" not in env


def test_worker_invocation_rejects_untrusted_sandbox_paths(runtime_home, tmp_path):
    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir(mode=0o700)
    sandbox = worker_broker.WorkerSandbox(
        root=sandbox_root,
        home=runtime_home,
        workdir=sandbox_root / "work",
        tmp=sandbox_root / "tmp",
        xdg_config_home=sandbox_root / "xdg-config",
        xdg_cache_home=sandbox_root / "xdg-cache",
    )
    context_path = sandbox.root / "context.json"
    context_path.write_text("{}")
    context_path.chmod(0o600)

    with pytest.raises(ValueError, match="sandbox"):
        spawner.build_worker_invocation(
            job_id="job_123",
            run_id="run_123",
            role="code_worker",
            attempt_id="att_123",
            lease_owner="daemon",
            context_path=context_path,
            sandbox=sandbox,
        )


def test_worker_invocation_requires_lease_identity(runtime_home):
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=runtime_home.parent / "workers",
        job_id="job_123",
        attempt_id="att_123",
        hermes_home=runtime_home,
    )
    with pytest.raises(ValueError, match="attempt_id and lease_owner"):
        spawner.build_worker_invocation(
            job_id="job_123",
            run_id="run_123",
            role="code_worker",
            context_path=sandbox.root / "context.json",
            sandbox=sandbox,
        )


def test_worker_invocation_requires_brokered_context(runtime_home):
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=runtime_home.parent / "workers",
        job_id="job_123",
        attempt_id="att_123",
        hermes_home=runtime_home,
    )
    with pytest.raises(ValueError, match="context_path"):
        spawner.build_worker_invocation(
            job_id="job_123",
            run_id="run_123",
            role="code_worker",
            attempt_id="att_123",
            lease_owner="daemon",
            sandbox=sandbox,
        )


def test_worker_invocation_does_not_inherit_secret_like_env(runtime_home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HOME", str(runtime_home.parent))
    monkeypatch.setenv("PATH", "/tmp/poison")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/venv")
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=runtime_home.parent / "workers",
        job_id="job_123",
        attempt_id="att_123",
        hermes_home=runtime_home,
    )
    context_path = sandbox.root / "context.json"
    context_path.write_text("{}")
    context_path.chmod(0o600)

    invocation = spawner.build_worker_invocation(
        job_id="job_123",
        run_id="run_123",
        role="code_worker",
        attempt_id="att_123",
        lease_owner="daemon",
        context_path=context_path,
        sandbox=sandbox,
    )
    env = invocation.env

    assert "OPENAI_API_KEY" not in env
    assert "HERMES_HOME" not in env
    assert env["HOME"] == str(sandbox.home)
    assert "PATH" not in env
    assert "VIRTUAL_ENV" not in env


def test_worker_invocation_filters_extra_env(runtime_home):
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=runtime_home.parent / "workers",
        job_id="job_123",
        attempt_id="att_good",
        hermes_home=runtime_home,
    )
    context_path = sandbox.root / "context.json"
    context_path.write_text("{}")
    context_path.chmod(0o600)
    invocation = spawner.build_worker_invocation(
        job_id="job_123",
        run_id="run_123",
        role="code_worker",
        attempt_id="att_good",
        lease_owner="lease_good",
        context_path=context_path,
        sandbox=sandbox,
        extra_env={
            "OPENAI_API_KEY": "sk-secret",
            "UNRELATED": "value",
            "PYTHONPATH": "/tmp/poison",
            "HERMES_AGENT_RUNTIME_APPROVAL_NONCE": "nonce",
            "HERMES_AGENT_RUNTIME_ATTEMPT_ID": "att_evil",
            "HERMES_AGENT_RUNTIME_LEASE_OWNER": "lease_evil",
            "HERMES_AGENT_ROLE": "ops_worker",
            "HERMES_AGENT_RUNTIME_TRACE": "1",
            "HOME": "/tmp/home",
            "PATH": "/tmp/bin",
        },
    )
    env = invocation.env

    assert "OPENAI_API_KEY" not in env
    assert "UNRELATED" not in env
    assert "PYTHONPATH" not in env
    assert env["HOME"] == str(sandbox.home)
    assert "PATH" not in env
    assert "HERMES_AGENT_RUNTIME_APPROVAL_NONCE" not in env
    assert env["HERMES_AGENT_RUNTIME_ATTEMPT_ID"] == "att_good"
    assert env["HERMES_AGENT_RUNTIME_LEASE_OWNER"] == "lease_good"
    assert "HERMES_AGENT_ROLE" not in env
    assert env["HERMES_AGENT_RUNTIME_TRACE"] == "1"


def test_dispatch_once_without_spawn_does_not_claim_or_burn_attempt(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Dispatch run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Do code")

        result = scheduler.dispatch_once(conn, lease_owner="test-daemon", spawn=False, now=4_000)

        assert result.claimed == 0
        assert result.spawned == 0
        assert result.claims == ()
        assert db.get_job(conn, job_id).status == "ready"
        assert db.get_job(conn, job_id).attempt_count == 0


def test_dispatch_spawn_mode_is_disabled_until_real_worker_execution_exists(runtime_home):
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Spawn disabled run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Do not spawn")

        result = scheduler.dispatch_once(conn, lease_owner="test-daemon", spawn=True, now=5_000)

        assert result.claimed == 0
        assert result.spawned == 0
        assert result.errors == (
            "spawn mode requires explicit enable_spawn=True and a reviewed isolation backend",
        )
        assert db.get_job(conn, job_id).status == "ready"
        assert db.get_job(conn, job_id).attempt_count == 0


def test_dispatch_spawn_enabled_claims_commits_launches_and_records_success(runtime_home, tmp_path):
    observed = {}

    class SuccessfulProcess:
        pid = 4321
        returncode = 0

        def communicate(self, timeout=None):
            observed["communicate_timeout"] = timeout
            return '{"success": true, "summary": "spawned worker completed"}', ""

    def fake_popen(argv, *, cwd, env, stdout, stderr, text):
        observed["argv"] = tuple(argv)
        observed["cwd"] = Path(cwd)
        observed["env"] = dict(env)
        observed["stdout"] = stdout
        observed["stderr"] = stderr
        observed["text"] = text
        # The claim and attempt must be committed before the worker process can
        # observe them through an independent DB connection.
        with db.connect() as other_conn:
            visible_job = db.get_job(other_conn, job_id)
            assert visible_job is not None
            assert visible_job.status == "leased"
            assert visible_job.lease_owner == "test-daemon"
        return SuccessfulProcess()

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Spawn enabled run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Launch worker")
        conn.commit()

        result = scheduler.dispatch_once(
            conn,
            lease_owner="test-daemon",
            spawn=True,
            enable_spawn=True,
            max_claims=1,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=fake_popen,
            workspace_root=str(tmp_path / "workers"),
            worker_timeout_seconds=42,
            now=6_000,
        )

        assert result.errors == ()
        assert result.claimed == 1
        assert result.spawned == 1
        assert len(result.claims) == 1
        assert result.claims[0].job_id == job_id
        assert observed["argv"][0] == "/usr/bin/bwrap"
        assert observed["cwd"].name == "work"
        assert observed["communicate_timeout"] == 42
        context_path = Path(observed["env"]["HERMES_AGENT_RUNTIME_CONTEXT"])
        assert context_path.is_file()
        assert stat.S_IMODE(context_path.lstat().st_mode) == 0o600
        context = json.loads(context_path.read_text())
        assert context["job"]["id"] == job_id
        assert context["lease"]["attempt_id"] == result.claims[0].attempt_id
        assert context["lease"]["lease_owner"] == "test-daemon"
        assert observed["env"]["HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION"] == "1"
        assert "HERMES_HOME" not in observed["env"]
        job = db.get_job(conn, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.result_summary == "spawned worker completed"
        attempt = conn.execute("SELECT status, pid, summary FROM runtime_attempts WHERE id=?", (result.claims[0].attempt_id,)).fetchone()
        assert attempt["status"] == "succeeded"
        assert attempt["pid"] == 4321
        assert attempt["summary"] == "spawned worker completed"


def test_dispatch_spawn_worker_timeout_stays_below_lease_ttl(runtime_home, tmp_path):
    observed = {}

    class SuccessfulProcess:
        pid = 1212
        returncode = 0

        def communicate(self, timeout=None):
            observed["timeout"] = timeout
            return '{"success": true, "summary": "bounded timeout"}', ""

    def fake_popen(*_args, **_kwargs):
        return SuccessfulProcess()

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Timeout bound run")
        db.create_job(conn, run_id=run_id, role="code_worker", title="Bound timeout")

        result = scheduler.dispatch_once(
            conn,
            lease_owner="test-daemon",
            spawn=True,
            enable_spawn=True,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=fake_popen,
            workspace_root=str(tmp_path / "workers"),
            lease_ttl_seconds=5,
            worker_timeout_seconds=50,
            now=6_500,
        )

        assert result.errors == ()
        assert result.claimed == 1
        assert result.claims[0].lease_expires_at == 6_505
        assert observed["timeout"] == 4


def test_dispatch_spawn_popen_failure_releases_claim_for_retry(runtime_home, tmp_path):
    def failing_popen(*_args, **_kwargs):
        raise OSError("cannot launch worker")

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Spawn failure run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Retry launch", max_attempts=2)

        result = scheduler.dispatch_once(
            conn,
            lease_owner="test-daemon",
            spawn=True,
            enable_spawn=True,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=failing_popen,
            workspace_root=str(tmp_path / "workers"),
            now=7_000,
        )

        assert result.claimed == 1
        assert result.spawned == 0
        assert result.errors and "cannot launch worker" in result.errors[0]
        job = db.get_job(conn, job_id)
        assert job is not None
        assert job.status == "ready"
        assert job.lease_owner is None
        assert job.attempt_count == 1
        attempt = conn.execute("SELECT status, error FROM runtime_attempts WHERE id=?", (result.claims[0].attempt_id,)).fetchone()
        assert attempt["status"] == "failed"
        assert "spawn failed" in attempt["error"]


def test_dispatch_spawn_pid_record_failure_kills_worker_and_retries(runtime_home, tmp_path):
    class KillingProcess:
        pid = 2222
        returncode = -9

        def __init__(self):
            self.killed = False

        def kill(self):
            self.killed = True

        def communicate(self, timeout=None):
            return "", "killed after pid record failure"

    process = KillingProcess()

    def fake_popen(*_args, **_kwargs):
        return process

    class FailingPidConnection:
        def __init__(self, inner):
            self.inner = inner
            self.failed = False

        def execute(self, sql, *args, **kwargs):
            if not self.failed and "UPDATE runtime_attempts" in str(sql) and "SET pid" in str(sql):
                self.failed = True
                raise sqlite3.OperationalError("pid update failed")
            return self.inner.execute(sql, *args, **kwargs)

        def commit(self):
            return self.inner.commit()

        def __getattr__(self, name):
            return getattr(self.inner, name)

    with db.connect() as raw_conn:
        run_id = db.create_run(raw_conn, title="Pid failure run")
        job_id = db.create_job(raw_conn, run_id=run_id, role="code_worker", title="Retry after pid failure", max_attempts=2)
        raw_conn.commit()
        conn = FailingPidConnection(raw_conn)

        result = scheduler.dispatch_once(
            cast(sqlite3.Connection, conn),
            lease_owner="test-daemon",
            spawn=True,
            enable_spawn=True,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=fake_popen,
            workspace_root=str(tmp_path / "workers"),
            now=7_500,
        )

        assert result.claimed == 1
        assert result.spawned == 1
        assert result.errors and "pid update failed" in result.errors[0]
        assert process.killed is True
        job = db.get_job(raw_conn, job_id)
        assert job is not None
        assert job.status == "ready"
        assert job.lease_owner is None
        attempt = raw_conn.execute("SELECT status, error FROM runtime_attempts WHERE id=?", (result.claims[0].attempt_id,)).fetchone()
        assert attempt["status"] == "failed"
        assert "post-spawn bookkeeping failed" in attempt["error"]


def test_dispatch_spawn_pid_record_zero_rows_kills_worker_without_stale_pid(runtime_home, tmp_path):
    class KillingProcess:
        pid = 2424
        returncode = -9

        def __init__(self):
            self.killed = False

        def kill(self):
            self.killed = True

        def communicate(self, timeout=None):
            return "", "killed after stale pid update"

    process = KillingProcess()

    def fake_popen(*_args, **_kwargs):
        return process

    class StalePidConnection:
        def __init__(self, inner, job_id):
            self.inner = inner
            self.job_id = job_id
            self.staled = False

        def execute(self, sql, *args, **kwargs):
            if not self.staled and "UPDATE runtime_attempts" in str(sql) and "SET pid" in str(sql):
                self.staled = True
                self.inner.execute(
                    "UPDATE runtime_jobs SET status='ready', lease_owner=NULL, lease_expires_at=NULL WHERE id=?",
                    (self.job_id,),
                )
                self.inner.commit()
            return self.inner.execute(sql, *args, **kwargs)

        def commit(self):
            return self.inner.commit()

        def __getattr__(self, name):
            return getattr(self.inner, name)

    with db.connect() as raw_conn:
        run_id = db.create_run(raw_conn, title="Stale pid run")
        job_id = db.create_job(raw_conn, run_id=run_id, role="code_worker", title="Do not mutate stale pid", max_attempts=2)
        raw_conn.commit()
        conn = StalePidConnection(raw_conn, job_id)

        result = scheduler.dispatch_once(
            cast(sqlite3.Connection, conn),
            lease_owner="test-daemon",
            spawn=True,
            enable_spawn=True,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=fake_popen,
            workspace_root=str(tmp_path / "workers"),
            now=7_550,
        )

        assert result.claimed == 1
        assert result.spawned == 1
        assert result.errors and "pid update lost active lease" in result.errors[0]
        assert process.killed is True
        attempt = raw_conn.execute("SELECT pid FROM runtime_attempts WHERE id=?", (result.claims[0].attempt_id,)).fetchone()
        assert attempt["pid"] is None
        job = db.get_job(raw_conn, job_id)
        assert job is not None
        assert job.status == "ready"


def test_dispatch_spawn_reaper_exception_fails_claim_for_retry(runtime_home, tmp_path, monkeypatch):
    class SuccessfulProcess:
        pid = 3333
        returncode = 0

        def communicate(self, timeout=None):
            return '{"success": true, "summary": "unrecorded"}', ""

    def fake_popen(*_args, **_kwargs):
        return SuccessfulProcess()

    def broken_reaper(*_args, **_kwargs):
        raise RuntimeError("broker write failed")

    monkeypatch.setattr(scheduler.worker_broker, "reap_worker_process", broken_reaper)

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Reaper failure run")
        job_id = db.create_job(conn, run_id=run_id, role="code_worker", title="Retry after reaper failure", max_attempts=2)

        result = scheduler.dispatch_once(
            conn,
            lease_owner="test-daemon",
            spawn=True,
            enable_spawn=True,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=fake_popen,
            workspace_root=str(tmp_path / "workers"),
            now=7_600,
        )

        assert result.claimed == 1
        assert result.spawned == 1
        assert result.errors and "broker write failed" in result.errors[0]
        job = db.get_job(conn, job_id)
        assert job is not None
        assert job.status == "ready"
        assert job.lease_owner is None
        attempt = conn.execute("SELECT status, error FROM runtime_attempts WHERE id=?", (result.claims[0].attempt_id,)).fetchone()
        assert attempt["status"] == "failed"
        assert "worker reaper failed" in attempt["error"]


def test_dispatch_spawn_respects_max_claims(runtime_home, tmp_path):
    class SuccessfulProcess:
        pid = 1111
        returncode = 0

        def communicate(self, timeout=None):
            return '{"success": true, "summary": "ok"}', ""

    def fake_popen(*_args, **_kwargs):
        return SuccessfulProcess()

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Max claims run")
        first = db.create_job(conn, run_id=run_id, role="code_worker", title="First")
        second = db.create_job(conn, run_id=run_id, role="code_worker", title="Second")

        result = scheduler.dispatch_once(
            conn,
            lease_owner="test-daemon",
            spawn=True,
            enable_spawn=True,
            max_claims=1,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=fake_popen,
            workspace_root=str(tmp_path / "workers"),
            now=8_000,
        )

        assert result.claimed == 1
        assert result.spawned == 1
        claimed_job_id = result.claims[0].job_id
        statuses = {job.id: job.status for job in db.list_jobs(conn, run_id)}
        assert statuses[claimed_job_id] == "succeeded"
        assert sorted(statuses.values()) == ["ready", "succeeded"]
        assert {first, second} == set(statuses)


def test_dispatch_spawn_queued_job_integration_records_real_child_success(runtime_home, tmp_path):
    """Queue one job and run the trusted broker/reaper path against a real child process.

    The child is deterministic and local; this proves the queued spawn plumbing can
    complete one non-empty job before any live LLM worker rollout is enabled.
    """

    launched = {}

    def deterministic_child_popen(_argv, *, cwd, env, stdout, stderr, text):
        launched["cwd"] = cwd
        launched["env_keys"] = sorted(env)
        code = "import json; print(json.dumps({'success': True, 'summary': 'deterministic queued integration ok'}))"
        return subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=cwd,
            env={"PATH": "/usr/bin:/bin"},
            stdout=stdout,
            stderr=stderr,
            text=text,
        )

    with db.connect() as conn:
        run_id = db.create_run(conn, title="Queued integration run")
        job_id = db.create_job(conn, run_id=run_id, role="explorer", title="Real child smoke", max_attempts=1)

        result = scheduler.dispatch_once(
            conn,
            lease_owner="integration-daemon",
            spawn=True,
            enable_spawn=True,
            max_claims=1,
            isolation_backend="bubblewrap",
            executable_resolver=lambda name: f"/usr/bin/{name}",
            popen_factory=deterministic_child_popen,
            workspace_root=str(tmp_path / "workers"),
            now=9_000,
        )

        assert result.claimed == 1
        assert result.spawned == 1
        assert result.errors == ()
        job = db.get_job(conn, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.result_summary == "deterministic queued integration ok"
        attempt = conn.execute("SELECT status, pid, summary, error FROM runtime_attempts WHERE job_id=?", (job_id,)).fetchone()
        assert attempt["status"] == "succeeded"
        assert attempt["pid"] is not None
        assert attempt["summary"] == "deterministic queued integration ok"
        assert attempt["error"] == ""
    assert Path(launched["cwd"]).is_dir()
    assert "HERMES_AGENT_RUNTIME_CONTEXT" in launched["env_keys"]
