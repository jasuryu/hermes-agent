"""Tests for deterministic Agent Runtime → Obsidian runbook sync."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    vault = home / "obsidian"
    home.mkdir()
    vault.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def test_render_runtime_runbook_is_deterministic_mirror_without_raw_job_body():
    from agent_runtime import db
    from agent_runtime.scribe_sync import render_runbook_markdown

    standalone_secret = "sk-" + "a" * 16
    env_secret = "env-secret-value"
    standalone_key_name = "OPENAI_API_KEY"
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title="Runtime rollout",
            objective="Keep recovery-only daemon documented",
            owner_source="telegram:thread/234",
            public_ref="HP-88",
            risk_level="prod_sensitive",
            now=1_700_000_000,
        )
        job_id = db.create_job(
            conn,
            run_id=run_id,
            role="scribe",
            title="Write runbook",
            body="Internal prompt must stay out of notes. api_key=super-secret-value",
            now=1_700_000_010,
        )
        db.add_finding(
            conn,
            run_id=run_id,
            job_id=job_id,
            severity="high",
            category="safety",
            summary=f"Found token=secret-token-value, OPENAI_API_KEY={env_secret}, bare {standalone_key_name}, and {standalone_secret} in an unsafe draft",
            evidence_ref="local-only transcript",
            recommendation="Redact before publishing",
            now=1_700_000_020,
        )
        db.record_decision(
            conn,
            run_id=run_id,
            kind="proceed",
            rationale="Recovery-only docs sync is safe; password=another-secret-value must redact.",
            job_id=job_id,
            now=1_700_000_030,
        )

        markdown = render_runbook_markdown(conn, run_id, generated_at=1_700_000_040)

    assert "type: hermes-agent-runtime-runbook" in markdown
    assert "Runtime rollout" in markdown
    assert "documentation mirror only" in markdown
    assert "not an execution queue" in markdown
    assert "HP-88" in markdown
    assert "Jobs: 1 total" in markdown
    assert "`job_" in markdown
    assert "Write runbook" in markdown
    assert "Internal prompt must stay out of notes" not in markdown
    assert "super-secret-value" not in markdown
    assert "secret-token-value" not in markdown
    assert env_secret not in markdown
    assert "bare OPENAI_API_KEY" not in markdown
    assert standalone_secret not in markdown
    assert "another-secret-value" not in markdown
    assert "token=" not in markdown
    assert "password=" not in markdown
    assert "OPENAI_API_KEY" not in markdown
    assert "[REDACTED_KEY]=[REDACTED]" in markdown
    assert "bare [REDACTED_KEY]" in markdown
    assert "[REDACTED]" in markdown


def test_render_runtime_runbook_generated_at_is_stable_from_run_update_time():
    from agent_runtime import db
    from agent_runtime.scribe_sync import render_runbook_markdown

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Stable render", public_ref="HP-88", now=1_700_000_000)
        first = render_runbook_markdown(conn, run_id)
        second = render_runbook_markdown(conn, run_id)

    assert first == second
    assert "generated_at: 2023-11-14T22:13:20+00:00" in first


def test_sync_runtime_runbook_redacts_secret_like_run_metadata_from_path(tmp_path):
    from agent_runtime import db
    from agent_runtime.scribe_sync import sync_runbook_to_obsidian

    vault = tmp_path / "vault"
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title="Runtime OPENAI_API_KEY=super-secret-value",
            public_ref="ACCESS_TOKEN=public-secret-value",
            now=1_700_000_000,
        )
        payload = sync_runbook_to_obsidian(conn, run_id, vault_path=vault, write=False)

    assert "super-secret-value" not in payload["path"]
    assert "public-secret-value" not in payload["path"]
    assert "OPENAI_API_KEY" not in Path(payload["path"]).name
    assert "ACCESS_TOKEN" not in Path(payload["path"]).name
    assert "[REDACTED_KEY]=[REDACTED]" in Path(payload["path"]).name


def test_sync_runtime_runbook_sanitizes_malformed_run_id_in_path(tmp_path):
    from agent_runtime.scribe_sync import RUNBOOK_RELATIVE_DIR, sync_runbook_to_obsidian
    from agent_runtime import db

    vault = tmp_path / "vault"
    malformed_run_id = "../escape/OPENAI_API_KEY=super-secret-value"
    now = 1_700_000_000
    db.init_db()
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO runtime_runs
            (id, title, objective, owner_source, public_ref, status, risk_level,
             orchestrator_session_id, created_at, updated_at)
            VALUES (?, 'Malformed id', '', '', 'HP-88', 'running', 'medium', '', ?, ?)
            """,
            (malformed_run_id, now, now),
        )
        payload = sync_runbook_to_obsidian(conn, malformed_run_id, vault_path=vault, write=False)

    path = Path(payload["path"])
    assert path.is_relative_to(vault / RUNBOOK_RELATIVE_DIR)
    assert ".." not in path.name
    assert "/" not in path.name
    assert "super-secret-value" not in payload["path"]
    assert "super-secret-value" not in payload["markdown"]
    assert "OPENAI_API_KEY" not in path.name
    assert "[REDACTED_KEY]=[REDACTED]" in path.name
    assert "runtime_run_id:" in payload["markdown"]


def test_sync_runtime_runbook_dry_run_and_write_are_path_safe(tmp_path):
    from agent_runtime import db
    from agent_runtime.scribe_sync import sync_runbook_to_obsidian

    vault = tmp_path / "vault"
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title="../Unsafe / Runtime",
            objective="Path must remain below vault",
            public_ref="HP-88/../../evil",
            now=1_700_000_000,
        )

        dry = sync_runbook_to_obsidian(conn, run_id, vault_path=vault, write=False, generated_at=1_700_000_010)
        assert dry["written"] is False
        note_path = Path(dry["path"])
        assert note_path.name.endswith(f"{run_id}.md")
        assert note_path.is_relative_to(vault)
        assert ".." not in note_path.name
        assert "/" not in note_path.name
        assert not note_path.exists()

        written = sync_runbook_to_obsidian(conn, run_id, vault_path=vault, write=True, generated_at=1_700_000_020)
        written_path = Path(written["path"])

    assert written["written"] is True
    assert written_path.is_relative_to(vault)
    assert written_path.exists()
    content = written_path.read_text(encoding="utf-8")
    assert f' runtime_run_id: {run_id}' not in content
    assert f'runtime_run_id: "{run_id}"' in content
    assert "documentation mirror only" in content


def test_sync_runtime_runbook_rejects_symlink_escape(tmp_path):
    from agent_runtime import db
    from agent_runtime.scribe_sync import sync_runbook_to_obsidian

    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    (vault / "01 Hermes").symlink_to(outside, target_is_directory=True)

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Symlink escape", public_ref="HP-88")
        with pytest.raises(ValueError, match="symlink"):
            sync_runbook_to_obsidian(conn, run_id, vault_path=vault, write=False)


def test_sync_runtime_runbook_rejects_runbook_subtree_symlink_inside_vault(tmp_path):
    from agent_runtime import db
    from agent_runtime.scribe_sync import sync_runbook_to_obsidian

    vault = tmp_path / "vault"
    redirected = vault / "Other"
    runs_parent = vault / "01 Hermes" / "Agent Runtime"
    redirected.mkdir(parents=True)
    runs_parent.mkdir(parents=True)
    (runs_parent / "Runs").symlink_to(redirected, target_is_directory=True)

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Subtree symlink", public_ref="HP-88")
        with pytest.raises(ValueError, match="symlink"):
            sync_runbook_to_obsidian(conn, run_id, vault_path=vault, write=False)


def test_sync_runtime_runbook_rejects_existing_target_symlink(tmp_path):
    from agent_runtime import db
    from agent_runtime.scribe_sync import sync_runbook_to_obsidian

    vault = tmp_path / "vault"
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Target symlink", public_ref="HP-88")
        dry = sync_runbook_to_obsidian(conn, run_id, vault_path=vault, write=False)
        target = Path(dry["path"])
        target.parent.mkdir(parents=True)
        target.symlink_to(outside)
        with pytest.raises(ValueError, match="symlink"):
            sync_runbook_to_obsidian(conn, run_id, vault_path=vault, write=True)
