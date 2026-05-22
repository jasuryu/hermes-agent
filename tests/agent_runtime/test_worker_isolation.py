"""Tests for Agent Runtime worker isolation preflight."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from agent_runtime import spawner, worker_broker, worker_isolation


def _private_file(path: Path, content: str = "{}") -> Path:
    path.write_text(content)
    path.chmod(0o600)
    return path


def test_worker_isolation_defaults_to_fail_closed_without_backend():
    assessment = worker_isolation.assess_worker_isolation(backend="disabled", executable_resolver=lambda _: None)

    assert assessment.available is False
    assert assessment.backend == "disabled"
    assert "disabled" in assessment.reason
    assert assessment.allows_spawn is False


def test_worker_isolation_requires_real_backend_not_scratch_env_only():
    assessment = worker_isolation.assess_worker_isolation(backend="env_only", executable_resolver=lambda _: "/bin/true")

    assert assessment.available is False
    assert assessment.allows_spawn is False
    assert "not a security boundary" in assessment.reason


def test_worker_isolation_accepts_configured_bubblewrap_backend():
    assessment = worker_isolation.assess_worker_isolation(backend="bubblewrap", executable_resolver=lambda name: f"/usr/bin/{name}")

    assert assessment.available is True
    assert assessment.backend == "bubblewrap"
    assert assessment.executable == "/usr/bin/bwrap"
    assert assessment.allows_spawn is True
    assert "reviewed launch policy" in assessment.reason


def test_bubblewrap_launch_plan_wraps_worker_and_enforces_workdir(tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=tmp_path / "workers",
        job_id="job_launch",
        attempt_id="att_launch",
        hermes_home=hermes_home,
    )
    context_path = _private_file(sandbox.root / "context.json")
    worker_argv = ["/usr/bin/python3", "-m", "agent_runtime.worker_main", "--job", "job_launch"]
    worker_env = {
        "HOME": str(sandbox.home),
        "TMPDIR": str(sandbox.tmp),
        "XDG_CONFIG_HOME": str(sandbox.xdg_config_home),
        "XDG_CACHE_HOME": str(sandbox.xdg_cache_home),
        "HERMES_AGENT_RUNTIME_CONTEXT": str(context_path),
        "HERMES_AGENT_RUNTIME_ATTEMPT_ID": "att_launch",
        "HERMES_AGENT_RUNTIME_LEASE_OWNER": "daemon",
    }

    plan = worker_isolation.build_launch_plan(
        backend="bubblewrap",
        worker_argv=worker_argv,
        worker_env=worker_env,
        cwd=sandbox.workdir,
        sandbox=sandbox,
        context_path=context_path,
        executable_resolver=lambda name: f"/usr/bin/{name}",
    )

    assert plan.backend == "bubblewrap"
    assert plan.executable == "/usr/bin/bwrap"
    assert plan.cwd == sandbox.workdir
    assert plan.env == worker_env
    assert plan.worker_argv == tuple(worker_argv)
    assert plan.argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in plan.argv
    assert "--dev" in plan.argv
    assert "/dev" in plan.argv
    assert "--chdir" in plan.argv
    assert plan.argv[plan.argv.index("--chdir") + 1] == str(sandbox.workdir)
    assert "HERMES_AGENT_RUNTIME_CONTEXT" in plan.argv
    assert plan.argv[plan.argv.index("HERMES_AGENT_RUNTIME_CONTEXT") + 1] == str(context_path)
    assert "HERMES_AGENT_RUNTIME_ATTEMPT_ID" in plan.argv
    assert "--ro-bind" in plan.argv
    assert "/usr" in plan.argv
    assert str(context_path) in plan.argv
    tmpfs_index = plan.argv.index("--tmpfs")
    sandbox_dir_index = plan.argv.index("--dir")
    assert plan.argv[sandbox_dir_index + 1] == str(sandbox.root)
    assert tmpfs_index < sandbox_dir_index
    for writable_path in (
        sandbox.home,
        sandbox.workdir,
        sandbox.tmp,
        sandbox.xdg_config_home,
        sandbox.xdg_cache_home,
    ):
        bind_index = next(
            i for i, token in enumerate(plan.argv[:-2])
            if token == "--bind" and plan.argv[i + 1] == str(writable_path)
        )
        assert plan.argv[bind_index + 2] == str(writable_path)
    context_ro_index = next(
        i for i, token in enumerate(plan.argv[:-2])
        if token == "--ro-bind" and plan.argv[i + 1] == str(context_path)
    )
    assert context_ro_index > sandbox_dir_index
    assert str(hermes_home) not in plan.argv
    assert plan.allows_spawn is True
    assert "reviewed bubblewrap launch policy" in plan.reason


def test_launch_plan_rejects_cwd_outside_sandbox(tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=tmp_path / "workers",
        job_id="job_launch",
        attempt_id="att_launch",
        hermes_home=hermes_home,
    )
    context_path = _private_file(sandbox.root / "context.json")

    with pytest.raises(ValueError, match="cwd"):
        worker_isolation.build_launch_plan(
            backend="bubblewrap",
            worker_argv=["python", "-m", "agent_runtime.worker_main"],
            worker_env={},
            cwd=hermes_home,
            sandbox=sandbox,
            context_path=context_path,
            executable_resolver=lambda name: f"/usr/bin/{name}",
        )


def test_launch_plan_rejects_env_paths_outside_sandbox(tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=tmp_path / "workers",
        job_id="job_launch",
        attempt_id="att_launch",
        hermes_home=hermes_home,
    )
    context_path = _private_file(sandbox.root / "context.json")

    for key, bad_value in (
        ("HOME", str(hermes_home)),
        ("TMPDIR", str(hermes_home / "tmp")),
        ("XDG_CONFIG_HOME", str(tmp_path / "outside-config")),
        ("HERMES_AGENT_RUNTIME_CONTEXT", str(sandbox.root / "other-context.json")),
        ("HERMES_AGENT_RUNTIME_DB_PATH", str(hermes_home / "agent-runtime" / "runtime.db")),
        ("KUBECONFIG", str(hermes_home / "kubeconfig")),
    ):
        env = {
            "HOME": str(sandbox.home),
            "TMPDIR": str(sandbox.tmp),
            "XDG_CONFIG_HOME": str(sandbox.xdg_config_home),
            "XDG_CACHE_HOME": str(sandbox.xdg_cache_home),
            "HERMES_AGENT_RUNTIME_CONTEXT": str(context_path),
            "HERMES_AGENT_RUNTIME_ATTEMPT_ID": "att_launch",
            "HERMES_AGENT_RUNTIME_LEASE_OWNER": "daemon",
            key: bad_value,
        }
        with pytest.raises(ValueError, match="env"):
            worker_isolation.build_launch_plan(
                backend="bubblewrap",
                worker_argv=["python", "-m", "agent_runtime.worker_main"],
                worker_env=env,
                cwd=sandbox.workdir,
                sandbox=sandbox,
                context_path=context_path,
                executable_resolver=lambda name: f"/usr/bin/{name}",
            )

def test_bubblewrap_launch_plan_runs_current_python_when_bwrap_installed(tmp_path):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        pytest.skip("bubblewrap executable is not installed")
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=tmp_path / "workers",
        job_id="job_real_bwrap",
        attempt_id="att_real_bwrap",
        hermes_home=hermes_home,
    )
    context_path = _private_file(sandbox.root / "context.json")
    worker_env = {
        "HOME": str(sandbox.home),
        "TMPDIR": str(sandbox.tmp),
        "XDG_CONFIG_HOME": str(sandbox.xdg_config_home),
        "XDG_CACHE_HOME": str(sandbox.xdg_cache_home),
        "HERMES_AGENT_RUNTIME_CONTEXT": str(context_path),
        "HERMES_AGENT_RUNTIME_ATTEMPT_ID": "att_real_bwrap",
        "HERMES_AGENT_RUNTIME_LEASE_OWNER": "daemon",
    }

    plan = worker_isolation.build_launch_plan(
        backend="bubblewrap",
        worker_argv=[sys.executable, "-c", "import sys; print('bwrap-ok:' + sys.executable)"],
        worker_env=worker_env,
        cwd=sandbox.workdir,
        sandbox=sandbox,
        context_path=context_path,
        executable_resolver=lambda _name: bwrap,
    )

    result = subprocess.run(list(plan.argv), text=True, capture_output=True, timeout=15)

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("bwrap-ok:")

def test_bubblewrap_launch_plan_runs_worker_main_module_when_bwrap_installed(tmp_path):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        pytest.skip("bubblewrap executable is not installed")
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    sandbox = worker_broker.create_worker_sandbox(
        workspace_root=tmp_path / "workers",
        job_id="job_worker_main_bwrap",
        attempt_id="att_worker_main_bwrap",
        hermes_home=hermes_home,
    )
    context_path = _private_file(sandbox.root / "context.json")
    invocation = spawner.build_worker_invocation(
        job_id="job_worker_main_bwrap",
        run_id="run_worker_main_bwrap",
        role="code_worker",
        attempt_id="att_worker_main_bwrap",
        lease_owner="daemon",
        context_path=context_path,
        sandbox=sandbox,
        enable_execution=False,
    )
    plan = worker_isolation.build_launch_plan(
        backend="bubblewrap",
        worker_argv=invocation.argv,
        worker_env=invocation.env,
        cwd=invocation.cwd,
        sandbox=sandbox,
        context_path=context_path,
        executable_resolver=lambda _name: bwrap,
    )

    result = subprocess.run(list(plan.argv), text=True, capture_output=True, timeout=15)

    assert result.returncode == 1
    assert "worker_execution_disabled" in result.stdout
    assert "No module named" not in result.stderr
