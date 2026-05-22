"""Tests for guarded Agent Runtime ops terminal toolset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def test_ops_terminal_toolset_resolves_only_when_worker_context_env_is_present(tmp_path, monkeypatch):
    context_path = tmp_path / "context.json"
    context_path.write_text('{"job":{"id":"job_1"},"constraints":{"runtime_db_access":"forbidden"},"approvals":[]}')
    context_path.chmod(0o600)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_CONTEXT", str(context_path))
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION", "1")

    import tools.agent_runtime_ops_tools  # noqa: F401 - ensure registry side effect
    from toolsets import resolve_toolset
    from tools.registry import registry, invalidate_check_fn_cache

    invalidate_check_fn_cache()
    names = set(resolve_toolset("ops_terminal"))
    assert names == {"runtime_ops_terminal"}
    schema = registry.get_definitions(names, quiet=True)
    schema_names = {item["function"]["name"] for item in schema}
    assert "runtime_ops_terminal" in schema_names


def test_ops_terminal_tool_is_not_in_default_core_toolset(monkeypatch):
    import tools.agent_runtime_ops_tools  # noqa: F401 - ensure registry side effect
    from toolsets import resolve_toolset

    core_names = set(resolve_toolset("hermes-cli"))
    assert "runtime_ops_terminal" not in core_names


def test_runtime_ops_terminal_handler_uses_context_guard_without_opening_runtime_db(tmp_path, monkeypatch):
    context_path = tmp_path / "context.json"
    context_path.write_text('{"job":{"id":"job_1"},"constraints":{"runtime_db_access":"forbidden"},"approvals":[]}')
    context_path.chmod(0o600)
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_CONTEXT", str(context_path))
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION", "1")

    from tools import agent_runtime_ops_tools as ops_tools

    observed = {}

    def fake_guard(context, **kwargs):
        observed["context"] = context
        observed["kwargs"] = kwargs
        return {"status": "blocked", "guard": {"allowed": False}, "exit_code": -1, "output": "", "error": "blocked"}

    monkeypatch.setattr(ops_tools.ops_guard, "guarded_ops_terminal", fake_guard)

    payload = json.loads(ops_tools.runtime_ops_terminal(command="kubectl get pods", target="cluster=whale namespace=prod"))

    assert payload["status"] == "blocked"
    assert observed["context"]["job"]["id"] == "job_1"
    assert observed["kwargs"]["command"] == "kubectl get pods"
    assert observed["kwargs"]["target"] == "cluster=whale namespace=prod"
