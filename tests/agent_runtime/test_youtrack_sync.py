"""Tests for Runtime → YouTrack public mirror helpers."""

from __future__ import annotations

import subprocess

import pytest


@pytest.fixture(autouse=True)
def _runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))


def test_render_youtrack_comment_is_public_mirror_without_raw_job_body():
    from agent_runtime import db, youtrack_sync

    standalone_secret = "sk-" + "b" * 16
    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title="Runtime YouTrack mirror",
            objective="Mirror public progress only",
            public_ref="HP-88",
            risk_level="prod_sensitive",
            now=1_700_000_000,
        )
        job_id = db.create_job(
            conn,
            run_id=run_id,
            role="scribe",
            title="Publish safe status",
            body="Raw prompt must stay private. OPENAI_API_KEY=super-secret-value",
            now=1_700_000_010,
        )
        db.add_finding(
            conn,
            run_id=run_id,
            job_id=job_id,
            severity="high",
            category="safety",
            summary=f"Unsafe draft had token=secret-token-value and {standalone_secret}",
            recommendation="Redact public mirror",
            now=1_700_000_020,
        )
        db.record_decision(
            conn,
            run_id=run_id,
            kind="proceed",
            rationale="Only public status may be mirrored; password=another-secret-value must redact.",
            job_id=job_id,
            now=1_700_000_030,
        )

        comment = youtrack_sync.render_youtrack_comment(conn, run_id, generated_at=1_700_000_040)

    assert "Hermes Agent Runtime public mirror" in comment
    assert "HP-88" in comment
    assert "Runtime YouTrack mirror" in comment
    assert "not an execution queue" in comment
    assert "not an approval source" in comment
    assert "Raw prompt must stay private" not in comment
    assert "super-secret-value" not in comment
    assert "secret-token-value" not in comment
    assert standalone_secret not in comment
    assert "another-secret-value" not in comment
    assert "OPENAI_API_KEY" not in comment
    assert "free-form objective/result/rationale/finding text omitted" in comment


def test_sync_youtrack_dry_run_does_not_call_ytctl_or_mutate_runtime_db():
    from agent_runtime import db, youtrack_sync

    calls: list[list[str]] = []

    def fake_runner(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime YouTrack dry-run", public_ref="HP-88", now=1_700_000_000)
        before_events = len(db.list_events(conn, run_id=run_id, limit=100))

        payload = youtrack_sync.sync_run_to_youtrack(conn, run_id, write=False, runner=fake_runner, generated_at=1_700_000_010)

        after_events = len(db.list_events(conn, run_id=run_id, limit=100))

    assert payload["success"] is True
    assert payload["written"] is False
    assert payload["issue_id"] == "HP-88"
    assert payload["operations"] == ["comment"]
    assert payload["commands"][0][:3] == ["ytctl", "comment", "HP-88"]
    assert "not an execution queue" in payload["comment"]
    assert calls == []
    assert after_events == before_events


def test_sync_youtrack_write_uses_argument_lists_for_comment_and_optional_stage():
    from agent_runtime import db, youtrack_sync

    calls: list[list[str]] = []

    def fake_runner(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime YouTrack write", public_ref="HP-88", now=1_700_000_000)

        payload = youtrack_sync.sync_run_to_youtrack(
            conn,
            run_id,
            stage="Review",
            write=True,
            runner=fake_runner,
            generated_at=1_700_000_010,
        )

    assert payload["written"] is True
    assert payload["operations"] == ["comment", "stage"]
    assert calls[0][:3] == ["ytctl", "comment", "HP-88"]
    assert calls[1] == ["ytctl", "update", "HP-88", "stage", "Review"]
    assert all(isinstance(call, list) for call in calls)
    assert "Runtime YouTrack write" in calls[0][3]


def test_sync_youtrack_rejects_missing_or_invalid_issue_ref():
    from agent_runtime import db, youtrack_sync

    db.init_db()
    with db.connect() as conn:
        missing_ref_run = db.create_run(conn, title="No issue", public_ref="")
        invalid_ref_run = db.create_run(conn, title="Bad issue", public_ref="https://youtrack.local/issue/HP-88")

        with pytest.raises(ValueError, match="YouTrack issue"):
            youtrack_sync.sync_run_to_youtrack(conn, missing_ref_run, write=False)
        with pytest.raises(ValueError, match="YouTrack issue"):
            youtrack_sync.sync_run_to_youtrack(conn, invalid_ref_run, write=False)
        payload = youtrack_sync.sync_run_to_youtrack(conn, invalid_ref_run, issue_id="HP-88", write=False)

    assert payload["issue_id"] == "HP-88"


def test_sync_youtrack_rejects_control_or_secret_like_stage_values():
    from agent_runtime import db, youtrack_sync

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Stage guard", public_ref="HP-88")

        with pytest.raises(ValueError, match="Stage"):
            youtrack_sync.sync_run_to_youtrack(conn, run_id, stage="\nReview", write=False)
        with pytest.raises(ValueError, match="Stage"):
            youtrack_sync.sync_run_to_youtrack(conn, run_id, stage="Review\r", write=False)
        with pytest.raises(ValueError, match="Stage"):
            youtrack_sync.sync_run_to_youtrack(conn, run_id, stage="Review\x1b[31m", write=False)
        with pytest.raises(ValueError, match="Stage"):
            youtrack_sync.sync_run_to_youtrack(conn, run_id, stage="password=secret-value", write=False)
        with pytest.raises(ValueError, match="Stage"):
            youtrack_sync.sync_run_to_youtrack(conn, run_id, stage="OPENAI_API_KEY", write=False)


def test_sync_youtrack_explicit_issue_id_is_reflected_in_comment():
    from agent_runtime import db, youtrack_sync

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Issue override", public_ref="HP-88")
        payload = youtrack_sync.sync_run_to_youtrack(conn, run_id, issue_id="HP-89", write=False)

    assert payload["issue_id"] == "HP-89"
    assert "YouTrack issue: `HP-89`" in payload["comment"]
    assert "YouTrack issue: `HP-88`" not in payload["comment"]


def test_render_youtrack_comment_omits_freeform_private_runtime_fields():
    from agent_runtime import db, youtrack_sync

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title="Public status only",
            objective="PRIVATE semantic deployment context that is not secret-shaped",
            public_ref="HP-88",
            now=1_700_000_000,
        )
        job_id = db.create_job(
            conn,
            run_id=run_id,
            role="scribe",
            title="PRIVATE job title context",
            body="PRIVATE body context",
            now=1_700_000_010,
        )
        conn.execute("UPDATE runtime_jobs SET result_summary=? WHERE id=?", ("PRIVATE result context", job_id))
        db.add_finding(
            conn,
            run_id=run_id,
            severity="high",
            category="safety",
            summary="PRIVATE finding context",
            recommendation="PRIVATE recommendation context",
            now=1_700_000_020,
        )
        db.record_decision(
            conn,
            run_id=run_id,
            kind="proceed",
            rationale="PRIVATE decision context",
            now=1_700_000_030,
        )

        comment = youtrack_sync.render_youtrack_comment(conn, run_id)

    assert "PRIVATE semantic deployment context" not in comment
    assert "PRIVATE job title context" not in comment
    assert "PRIVATE result context" not in comment
    assert "PRIVATE finding context" not in comment
    assert "PRIVATE recommendation context" not in comment
    assert "PRIVATE decision context" not in comment
    assert "free-form objective/result/rationale/finding text omitted" in comment
