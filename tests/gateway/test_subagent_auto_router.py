from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _enabled_config() -> dict:
    return {
        "subagent_router": {
            "enabled": True,
            "platforms": ["telegram"],
            "levels": ["L2", "L3"],
            "allowed_sources": [
                {"platform": "telegram", "chat_id": "-1003941291843", "thread_id": "234"}
            ],
        }
    }


def test_subagent_router_disabled_by_default_does_not_create_runtime_run(runtime_home):
    from gateway.subagent_router import maybe_create_runtime_route
    from agent_runtime import db

    result = maybe_create_runtime_route(
        user_config={},
        platform_key="telegram",
        chat_id="-1003941291843",
        thread_id="234",
        session_id="sess_1",
        message_text="Давай проверим ещё способы купить золотые ETF из Узбекистана",
        owner_source="telegram:-1003941291843:234",
    )

    assert result is None
    db.init_db()
    with db.connect() as conn:
        assert db.doctor_status(conn)["runs"] == 0


def test_subagent_router_routes_l2_research_to_explorer_then_scribe(runtime_home):
    from gateway.subagent_router import maybe_create_runtime_route
    from agent_runtime import db

    result = maybe_create_runtime_route(
        user_config=_enabled_config(),
        platform_key="telegram",
        chat_id="-1003941291843",
        thread_id="234",
        session_id="sess_gold_etf",
        message_text="Давай проверим ещё какие есть способы находясь в Узбекистане купить золото, ETF или золотые ценные бумаги",
        owner_source="telegram:-1003941291843:234",
    )

    assert result is not None
    assert result.level == "L2"
    assert result.roles == ["explorer", "scribe"]
    assert "Runtime subagents" in result.response_text
    assert result.agent_result(history_len=0, user_text="hello")["final_response"] == result.response_text

    with db.connect() as conn:
        run = db.get_run(conn, result.run_id)
        jobs = db.list_jobs(conn, result.run_id)

    assert run is not None
    assert run.status == "running"
    assert run.owner_source == "telegram:-1003941291843:234"
    assert run.orchestrator_session_id == "sess_gold_etf"
    assert [job.role for job in jobs] == ["explorer", "scribe"]
    assert [job.status for job in jobs] == ["ready", "planned"]
    assert "ETF" in jobs[0].body


def test_subagent_router_routes_l3_code_work_without_orchestrator_job(runtime_home):
    from gateway.subagent_router import maybe_create_runtime_route
    from agent_runtime import db

    result = maybe_create_runtime_route(
        user_config=_enabled_config(),
        platform_key="telegram",
        chat_id="-1003941291843",
        thread_id="234",
        session_id="sess_router",
        message_text="Настрой deterministic router для L2 и L3 задач, добавь тесты, внедри в gateway и проверь dashboard",
        owner_source="telegram:-1003941291843:234",
    )

    assert result is not None
    assert result.level == "L3"
    assert result.roles == ["explorer", "code_worker", "sentinel", "scribe"]

    with db.connect() as conn:
        jobs = db.list_jobs(conn, result.run_id)

    assert "orchestrator" not in {job.role for job in jobs}
    assert [job.role for job in jobs] == ["explorer", "code_worker", "sentinel", "scribe"]
    assert jobs[0].status == "ready"
    assert jobs[1].status == "ready"
    assert jobs[2].status == "planned"
    assert jobs[3].status == "planned"


def test_workspace_registry_resolves_structured_metadata_for_configured_alias(tmp_path):
    from gateway.subagent_router import resolve_workspace

    finance_repo = tmp_path / "finance-control"
    finance_repo.mkdir()
    (finance_repo / ".env.example").write_text("PLACEHOLDER=1\n", encoding="utf-8")
    resolution = resolve_workspace(
        cfg={
            "workspaces": [
                {
                    "workspace_id": "finance-control",
                    "path": str(finance_repo),
                    "aliases": ["tez", "finance-control"],
                    "default_roles": ["code_worker", "sentinel"],
                }
            ]
        },
        role="code_worker",
        message_text="Почини TEZ import pipeline и добавь тесты",
    )

    assert resolution.workspace_kind == "dir"
    assert resolution.workspace_id == "finance-control"
    assert resolution.path == str(finance_repo)
    assert resolution.matched_by == "alias:tez"
    assert resolution.confidence == "high"
    assert resolution.mode == "writable_intent"
    assert resolution.to_job_workspace() == ("dir", str(finance_repo))


def test_workspace_registry_keeps_scribe_on_obsidian_before_generic_rules(tmp_path):
    from gateway.subagent_router import resolve_workspace

    hermes_repo = tmp_path / "hermes-agent"
    obsidian = tmp_path / "obsidian"
    hermes_repo.mkdir()
    obsidian.mkdir()

    resolution = resolve_workspace(
        cfg={
            "workspace_rules": [{"match": ["router"], "path": str(hermes_repo)}],
            "scribe_workspace": str(obsidian),
        },
        role="scribe",
        message_text="Запиши router handoff в документацию",
    )

    assert resolution.workspace_kind == "dir"
    assert resolution.workspace_id == "obsidian"
    assert resolution.path == str(obsidian)
    assert resolution.matched_by == "role:scribe"
    assert resolution.confidence == "high"


def test_workspace_registry_rejects_broad_and_secret_paths(tmp_path):
    from gateway.subagent_router import resolve_workspace

    secret_parent = tmp_path / ".env-project"
    secret_parent.mkdir()
    project_with_secret_child = tmp_path / "finance-control"
    project_with_secret_child.mkdir()
    (project_with_secret_child / "secrets").mkdir()

    home_resolution = resolve_workspace(
        cfg={"workspaces": [{"workspace_id": "home", "path": str(tmp_path), "aliases": ["home"]}]},
        role="code_worker",
        message_text="fix home project",
    )
    secret_resolution = resolve_workspace(
        cfg={"workspaces": [{"workspace_id": "secret", "path": str(secret_parent), "aliases": ["secret"]}]},
        role="code_worker",
        message_text="fix secret project",
    )
    secret_child_resolution = resolve_workspace(
        cfg={
            "workspaces": [
                {
                    "workspace_id": "finance-control",
                    "path": str(project_with_secret_child),
                    "aliases": ["finance-control"],
                }
            ]
        },
        role="code_worker",
        message_text="fix finance-control project",
    )

    assert home_resolution.workspace_kind == "scratch"
    assert home_resolution.confidence == "none"
    assert "broad" in home_resolution.reason
    assert secret_resolution.workspace_kind == "scratch"
    assert secret_resolution.confidence == "none"
    assert "secret" in secret_resolution.reason
    assert secret_child_resolution.workspace_kind == "scratch"
    assert secret_child_resolution.confidence == "none"
    assert "secret" in secret_child_resolution.reason


def test_runtime_route_writes_workspace_resolution_audit_to_job_body(tmp_path):
    from gateway.subagent_router import create_runtime_route
    from agent_runtime import db

    repo = tmp_path / "finance-control"
    repo.mkdir()

    result = create_runtime_route(
        level="L3",
        title="Finance workspace registry",
        objective="Реализуй TEZ import в finance-control и проверь тесты",
        owner_source="telegram:-1003941291843:234",
        router_config={
            "workspaces": [
                {
                    "workspace_id": "finance-control",
                    "path": str(repo),
                    "aliases": ["tez", "finance-control"],
                    "default_roles": ["code_worker", "sentinel"],
                }
            ]
        },
    )

    with db.connect() as conn:
        jobs = db.list_jobs(conn, result.run_id)
    code_job = next(job for job in jobs if job.role == "code_worker")

    assert code_job.workspace_kind == "dir"
    assert code_job.workspace_path == str(repo)
    assert "workspace_resolution:" in code_job.body
    assert "workspace_id: finance-control" in code_job.body
    assert "matched_by: alias:tez" in code_job.body
    assert "confidence: high" in code_job.body


def test_subagent_router_respects_source_allowlist(runtime_home):
    from gateway.subagent_router import maybe_create_runtime_route
    from agent_runtime import db

    result = maybe_create_runtime_route(
        user_config=_enabled_config(),
        platform_key="telegram",
        chat_id="999",
        thread_id="234",
        session_id="sess_other",
        message_text="Настрой большой проект с тестами и проверкой",
        owner_source="telegram:999:234",
    )

    assert result is None
    db.init_db()
    with db.connect() as conn:
        assert db.doctor_status(conn)["runs"] == 0


def test_subagent_router_does_not_auto_route_l4_production_mutation(runtime_home):
    from gateway.subagent_router import maybe_create_runtime_route
    from agent_runtime import db

    result = maybe_create_runtime_route(
        user_config=_enabled_config(),
        platform_key="telegram",
        chat_id="-1003941291843",
        thread_id="234",
        session_id="sess_prod",
        message_text="Сделай kubectl apply в production и перезапусти сервис прямо сейчас",
        owner_source="telegram:-1003941291843:234",
    )

    assert result is None
    db.init_db()
    with db.connect() as conn:
        assert db.doctor_status(conn)["runs"] == 0
