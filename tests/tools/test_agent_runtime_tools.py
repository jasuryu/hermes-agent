"""Tests for Agent Runtime orchestrator tools."""

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


def test_agent_runtime_toolset_resolves_runtime_tools(monkeypatch):
    monkeypatch.delenv("HERMES_AGENT_RUNTIME_APPROVAL_WRITER", raising=False)
    import tools.agent_runtime_tools  # noqa: F401 - ensure registry side effect
    from toolsets import resolve_toolset
    from tools.registry import registry, invalidate_check_fn_cache

    invalidate_check_fn_cache()
    names = set(resolve_toolset("agent_runtime"))
    assert {"runtime_create_run", "runtime_get_status", "runtime_record_decision"} <= names
    schema = registry.get_definitions(names, quiet=True)
    schema_names = {item["function"]["name"] for item in schema}
    assert "runtime_create_run" in schema_names
    assert "runtime_record_approval" not in schema_names


def test_agent_runtime_tools_are_not_in_default_core_toolset(monkeypatch):
    import tools.agent_runtime_tools  # noqa: F401 - ensure registry side effect
    from toolsets import resolve_toolset

    core_names = set(resolve_toolset("hermes-cli"))
    assert "runtime_create_run" not in core_names
    assert "runtime_create_job" not in core_names
    assert "runtime_record_decision" not in core_names


def test_approval_writer_tool_is_not_model_callable_even_with_env(monkeypatch):
    import tools.agent_runtime_tools  # noqa: F401 - ensure registry side effect
    from toolsets import resolve_toolset
    from tools.registry import registry, invalidate_check_fn_cache

    monkeypatch.setenv("HERMES_AGENT_RUNTIME_APPROVAL_WRITER", "1")
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_APPROVAL_NONCE", "nonce")
    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("agent_runtime")), quiet=True)
    schema_names = {item["function"]["name"] for item in schema}
    assert "runtime_record_approval" not in schema_names


def test_runtime_create_run_and_status_tool_roundtrip():
    from tools import agent_runtime_tools as rt

    created = json.loads(rt.runtime_create_run(title="Final runtime", public_ref="HP-88"))
    status = json.loads(rt.runtime_get_status(run_id=created["id"]))

    assert created["title"] == "Final runtime"
    assert status["run"]["public_ref"] == "HP-88"
    assert status["jobs"] == []


def test_runtime_record_decision_tool_persists_event():
    from tools import agent_runtime_tools as rt

    created = json.loads(rt.runtime_create_run(title="Decision run"))
    decision = json.loads(
        rt.runtime_record_decision(
            run_id=created["id"],
            kind="accept_risk",
            rationale="Sentinel finding reviewed by orchestrator",
        )
    )
    status = json.loads(rt.runtime_get_status(run_id=created["id"]))

    assert decision["kind"] == "accept_risk"
    assert any(event["kind"] == "decision_recorded" for event in status["events"])


def test_runtime_check_command_and_record_approval_tool_stays_disabled(monkeypatch):
    from tools import agent_runtime_tools as rt

    monkeypatch.setenv("HERMES_AGENT_RUNTIME_APPROVAL_WRITER", "1")
    monkeypatch.setenv("HERMES_AGENT_RUNTIME_APPROVAL_NONCE", "nonce")
    command = "kubectl -n prod rollout restart deploy/api"
    created = json.loads(rt.runtime_create_run(title="Approval tool run"))
    verdict = json.loads(rt.runtime_check_command(command=command))
    approval = json.loads(
        rt.runtime_record_approval(
            run_id=created["id"],
            target="cluster=whale namespace=prod deploy/api",
            commands=[command],
            reason="restart stuck deployment",
            blast_radius="api pods restart",
            rollback="kubectl -n prod rollout undo deploy/api",
            verification=["kubectl -n prod rollout status deploy/api"],
            approved_by="Jasur",
            approval_nonce="nonce",
        )
    )

    assert verdict["requires_approval"] is True
    assert approval["success"] is False
    assert "hermes runtime approve-command" in approval["error"]
