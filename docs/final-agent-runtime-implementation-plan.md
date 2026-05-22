# Final Agent Runtime Implementation Plan

> **For Hermes:** This plan documents the first implementation slice of the final Agent Runtime. Continue future work with TDD and keep Kanban as optional UI only.

**Goal:** Build a durable Orchestrator-centered Agent Runtime that replaces the old Kanban/profile fleet as machine execution truth.

**Architecture:** Runtime stores runs, jobs, dependencies, attempts, leases, events, artifacts, findings, approvals, and decisions in SQLite under Hermes home. Orchestrator uses CLI/tools to create runs/jobs and reconcile findings. Worker/Scheduler code is deliberately separated from Telegram gateway and legacy Kanban.

**Tech Stack:** Python 3.11, SQLite WAL, pytest, Hermes tool registry, Hermes CLI argparse.

---

## Completed MVP slice

### Task 1: Runtime schema and DB API

**Objective:** Create durable Runtime state independent from legacy Kanban.

**Files:**
- Create: `agent_runtime/db.py`
- Create: `agent_runtime/models.py`
- Test: `tests/agent_runtime/test_db_core.py`

**Implemented behavior:**
- idempotent DB init;
- `runtime_runs`, `runtime_jobs`, dependencies, attempts, events, artifacts, findings, approvals, decisions;
- run/job create/read;
- DAG child promotion;
- job lease claim/heartbeat/recovery;
- exact-scope approval lookup; model-callable approval recording remains disabled/fail-closed, while trusted operator writes go through `hermes runtime approve-command --write --operator-confirm APPROVE_RUNTIME_APPROVAL`.

### Task 2: Runtime policy primitives

**Objective:** Make Ops mutation approval deterministic instead of prompt-only.

**Files:**
- Create: `agent_runtime/policy.py`
- Test: `tests/agent_runtime/test_policy.py`

**Implemented behavior:**
- explicitly safe read-only kubectl/helm/terraform/docker commands allowed, while secret-bearing reads require approval;
- prod mutations/destructive cleanup require approval;
- approval packet contains exact command hashes and an unambiguous canonical JSON scope hash;
- exact command approval does not authorize different destructive commands.
- approval lookup validates exact command string + exact target/scope, recomputes command hashes from packet commands, and rejects malformed approval rows/dicts.

### Task 3: Runtime roles

**Objective:** Encode final role shape without recreating 20 Hermes profiles.

**Files:**
- Create: `agent_runtime/roles.py`

**Implemented roles:**
- orchestrator;
- explorer;
- code_worker;
- ops_worker;
- scribe;
- sentinel.

### Task 4: CLI surface

**Objective:** Add `hermes runtime ...` for doctor/status/run/job operations.

**Files:**
- Create: `hermes_cli/runtime.py`
- Modify: `hermes_cli/main.py`
- Test: `tests/hermes_cli/test_runtime_cli.py`

**Implemented commands:**
- `hermes runtime doctor [--json]`
- `hermes runtime init [--json]`
- `hermes runtime runs [--json]`
- `hermes runtime show <run_id> [--json]`
- `hermes runtime create-run ...`
- `hermes runtime create-job ...`
- `hermes runtime events ...`
- `hermes runtime approve-command RUN_ID --target ... --command ... --reason ... --blast-radius ... --rollback ... --verification ... --approved-by ... [--job-id JOB_ID] [--write --operator-confirm APPROVE_RUNTIME_APPROVAL] [--json]`

### Task 5: Orchestrator toolset

**Objective:** Expose Runtime to the main Orchestrator as tools.

**Files:**
- Create: `tools/agent_runtime_tools.py`
- Modify: `toolsets.py`
- Test: `tests/tools/test_agent_runtime_tools.py`

**Implemented tools:**
- `runtime_create_run`
- `runtime_create_job`
- `runtime_get_status`
- `runtime_record_decision`
- `runtime_add_finding`
- `runtime_check_command`
- `runtime_record_approval` exists only as a disabled fail-closed placeholder; it is not registered/model-callable.

### Task 6: Scheduler/spawner skeleton

**Objective:** Establish non-Kanban execution primitive without launching unsafe workers by default.

**Files:**
- Create: `agent_runtime/scheduler.py`
- Create: `agent_runtime/spawner.py`
- Create: `agent_runtime/worker_main.py`
- Test: `tests/agent_runtime/test_scheduler.py`

**Implemented behavior:**
- worker invocation env requires lease identity and omits run/job/role/Hermes-home/runtime-DB context from environment;
- default dispatch pass recovers expired leases and promotes ready jobs without claiming or burning attempts;
- subprocess spawn remains opt-in: the scheduler claims jobs only when `spawn=True`, `enable_spawn=True`, and a reviewed isolation backend launch plan is available.

### Task 7: Trusted worker broker and isolation preflight

**Objective:** Prepare the safe handoff path for future real worker spawn without giving workers direct Runtime DB/Hermes-home access.

**Files:**
- Create: `agent_runtime/worker_broker.py`
- Create: `agent_runtime/worker_isolation.py`
- Test: `tests/agent_runtime/test_worker_broker.py`
- Test: `tests/agent_runtime/test_worker_isolation.py`

**Implemented behavior:**
- trusted parent validates active lease owner + attempt id before materializing sanitized context;
- context JSON is private, contains no runtime DB/Hermes-home pointer, and is written into a private scratch sandbox outside `HERMES_HOME`;
- sandbox creation avoids ambient `TMPDIR` placement and rejects unsafe/symlink workspace hints;
- worker invocation validates sandbox paths before exporting `HOME`, `TMPDIR`, and XDG dirs;
- `bwrap`/`firejail` executable presence alone does not enable spawn; a backend-specific launch policy is required, and only the reviewed bubblewrap policy can currently allow spawn.

### Task 8: Backend-specific launch policy and cwd enforcement

**Objective:** Define the backend launch contract before real worker spawn is allowed.

**Files:**
- Modify: `agent_runtime/spawner.py`
- Modify: `agent_runtime/worker_isolation.py`
- Test: `tests/agent_runtime/test_scheduler.py`
- Test: `tests/agent_runtime/test_worker_isolation.py`

**Implemented behavior:**
- `build_worker_invocation()` returns a `WorkerInvocation` carrying `argv`, sanitized `env`, private `context_path`, and enforced `cwd=sandbox.workdir`;
- bubblewrap launch plans set child env through explicit allowlisted `--setenv`, reject runtime DB/Hermes-home/secret/tool config pointers, unshare networking, bind context read-only, and keep scratch HOME/TMPDIR/workdir/XDG dirs writable;
- reviewed bubblewrap launch plans now set `allows_spawn=true`, but scheduler still requires the explicit operator gate (`--spawn --enable-spawn`) and an available `bwrap` executable before claiming jobs.

### Task 9: Role-specific worker execution gate

**Objective:** Allow the isolated worker entrypoint to build role-specific `AIAgent` calls from brokered context without letting the scheduler spawn real workers yet.

**Files:**
- Create: `agent_runtime/worker_execution.py`
- Modify: `agent_runtime/worker_main.py`
- Modify: `agent_runtime/scheduler.py`
- Modify: `hermes_cli/runtime.py`
- Test: `tests/agent_runtime/test_worker_main.py`
- Test: `tests/agent_runtime/test_scheduler.py`
- Test: `tests/hermes_cli/test_runtime_cli.py`

**Implemented behavior:**
- `worker_main` refuses without lease identity, `HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION=1`, and a brokered context path;
- context job/attempt/lease identity is validated before role execution;
- non-ops roles instantiate `AIAgent` with role model, reasoning, and toolsets, with memory/context-files disabled and no writable runtime DB access;
- `ops_worker` refuses until the mandatory command guard wrapper exists;
- scheduler/CLI can launch non-ops role workers only behind the explicit spawn gate plus reviewed bubblewrap isolation; default dispatch/daemon execution remains recovery-only.

---

## Verification command

```bash
python -m pytest tests/agent_runtime tests/hermes_cli/test_runtime_cli.py tests/tools/test_agent_runtime_ops_tools.py tests/tools/test_agent_runtime_tools.py -q -o 'addopts='
python -m compileall -q agent_runtime hermes_cli/runtime.py hermes_cli/main.py tools/agent_runtime_ops_tools.py tools/agent_runtime_tools.py
hermes runtime doctor --json
```

Latest result: targeted runtime/approval/tool tests pass (`117 passed`), broad runtime/tools suite passes (`194 passed`), and compile check passes. Live-safe `approve-command` dry-run returned `written=false` with `active_approvals` unchanged (`0 -> 0`). Independent no-tools review passed after the trusted approval-channel redesign: `runtime_record_approval` is not registered/model-callable, `db.record_approval()` fails closed for importable callers, operator approval writes are CLI-only through `hermes runtime approve-command --write --operator-confirm APPROVE_RUNTIME_APPROVAL`, dry-run is non-mutating, approval packets are exact-scope, and ops execution requires an active matching approval plus unexpired worker context. Earlier verification remains valid: real subprocess CLI probes, deterministic Obsidian sync probes, deterministic YouTrack sync probes, read-only dashboard/Kanban mirror probes, health/alert probes, sandbox cleanup dry-runs, live-safe RAG retrieval smoke probe, and independent no-tools review pass for daemon/worker/context-broker/isolation-launch/role-execution/spawn-enable/bwrap-systemd/scribe-sync/youtrack-sync/mirror-observability-cleanup/parent-brokered-RAG hardening. The latest RAG review closed repeated boundary findings: Runtime worker RAG is parent-brokered only, direct `personal_kb` toolsets were removed from bounded worker roles, secret/restricted query material is refused before retrieval, service-packed context and untrusted `payload.query` are ignored, citations are normalized/escaped, and loopback/proxy/redirect controls protect the local RAG request path. The live host has `bwrap` installed (`/usr/bin/bwrap`, bubblewrap 0.9.0); an explicitly enabled empty-queue bubblewrap dispatch probe returns `rc=0` with `claimed=0`, `spawned=0`, and `errors=[]`. The recovery-only user daemon is live as `hermes-agent-runtime.service` with root linger enabled, `ActiveState=active`, `SubState=running`, `NRestarts=0`, no live `--spawn`/`--enable-spawn` flags, no child worker processes, and manual recovery-only tick output `claimed=0`, `spawned=0`, `errors=[]`. Runtime mirror live probe returned `runs=2`, `cards=6`, `mirror_only=true`, and `raw_job_bodies_returned=false`; `runtime health --json` returned `status=ok` with no alerts; sandbox cleanup dry-run returned `executed=false`, `candidates=0`. Runtime runbook mirrors were written under `/root/.hermes/obsidian/01 Hermes/Agent Runtime/Runs/` for the existing HP-88 runtime runs. Runtime sync deliberately stops at curated Obsidian markdown; the existing Hermes RAG pipeline owns Obsidian -> Postgres/Qdrant embedding/indexing.

Additional CLI skeleton:

- `hermes runtime dispatch-once [--json]` runs one scheduler tick with `spawn=False` by default.
- `hermes runtime daemon --max-ticks N [--json]` runs a bounded scheduler loop with `spawn=False` by default.
- `--spawn --enable-spawn --isolation-backend bubblewrap` is the explicit operator gate for worker launch; it claims only when `bwrap` is available and the reviewed launch plan validates.
- `hermes runtime service-unit` prints a recovery-only user systemd unit. `hermes runtime install-service` is dry-run by default and only writes the unit with `--write`; it never starts the service automatically.
- `hermes runtime approve-command RUN_ID --target ... --command ... --reason ... --blast-radius ... --rollback ... --verification ... --approved-by ... [--job-id JOB_ID] [--write --operator-confirm APPROVE_RUNTIME_APPROVAL] [--json]` previews or records a trusted operator approval packet. It is dry-run by default, validates strict trusted `approval_source` labels, enforces required audit fields, records exact target/commands/hash/scope, never registers model-callable approval creation, and keeps `db.record_approval()` fail-closed for importable callers. Job-scoped approvals are visible only to that job; run-level approvals are visible run-wide.
- `hermes runtime sync-obsidian RUN_ID [--write] [--json]` renders a deterministic documentation-only Obsidian mirror for one runtime run. It is dry-run by default, omits raw job bodies, redacts secret-like values in rendered rationale/findings/title/public-ref paths including environment-style names such as `*_API_KEY` and `*_TOKEN`, keeps note paths inside the selected vault and the fixed runbook subtree (including symlink escape/target checks), returns non-zero on write failures and corrupt DB reads, and labels the note as not an execution queue or approval source. Pure service-unit rendering and missing-DB sync probes do not create a runtime DB. It does not embed or upsert vectors; the existing Obsidian RAG ingest handles PG/Qdrant indexing after notes are written.
- `hermes runtime sync-youtrack RUN_ID [--issue-id HP-88] [--stage Review] [--write] [--json]` renders/posts a deterministic public YouTrack mirror. It is dry-run by default, reads an existing Runtime DB read-only, never reads YouTrack text as instructions, omits raw bodies/prompts/private context/approval packets/free-form objective/result/rationale/finding text, rejects invalid issue ids plus control/secret-like Stage values, uses argv-list `ytctl` calls with timeout only when `--write` is explicit, returns non-zero on `ytctl`/DB errors, and labels YouTrack as human-visible status only — not execution queue, approval source, scheduler, or worker command channel.
- `hermes runtime mirror [--json]` prints a read-only dashboard/Kanban-style Runtime snapshot. It reads an existing Runtime DB only, does not create/migrate DBs, returns compact run/job cards with lanes/progress/alerts, omits raw job bodies and approval command payloads, redacts secret-like values from display fields, marks every card `mirror_only=true`, and keeps Runtime SQLite as the execution truth.
- `hermes runtime health [--json]` probes the recovery-only user systemd service and Runtime DB read-only, emitting `ok`/`warning`/`critical` plus alert codes for inactive service, restarts, missing/corrupt DB, expired/stale job leases, and high/critical open findings. It sanitizes DB-derived alert strings and DB-read error details, and does not recover leases or mutate state.
- `hermes runtime cleanup-sandboxes [--max-age-seconds N] [--execute] [--json]` plans stale worker sandbox cleanup under the trusted temp parent. It is dry-run by default, rejects unsafe parents such as `/`, symlink components, or paths overlapping `HERMES_HOME`, only targets Runtime sandbox-prefix directories, skips symlinks/escaping paths, and deletes only with explicit `--execute`.
- Runtime worker contexts include parent-brokered RAG evidence for selected roles (`explorer`/`scribe`) only. The trusted parent calls the existing local Hermes RAG service, not a new embedding/indexing path, and materializes compact cited snippets as untrusted evidence. Bounded worker roles do not receive direct `personal_kb` toolsets; `code_worker`, `ops_worker`, and `sentinel` do not request RAG by default.
- Live rollout: `/root/.config/systemd/user/hermes-agent-runtime.service` is enabled and running under root's user manager. `loginctl enable-linger root` was applied so the user manager can survive logout. The service command is `/usr/local/lib/hermes-agent/venv/bin/python -m hermes_cli.main runtime daemon --interval 5 --lease-owner agent-runtime-daemon`.

## Independent review follow-up fixes

- `runtime_record_approval` is not registered as a model-callable tool; `db.record_approval()` remains fail-closed for importable callers, and actual approval writes are restricted to the reviewed operator CLI path `hermes runtime approve-command --write --operator-confirm APPROVE_RUNTIME_APPROVAL`.
- Unknown commands fail closed to approval-required unless explicitly allowlisted read-only.
- Secret-bearing reads (`kubectl` secrets/logs/config raw, raw API paths, structured `get -o*` output, configmaps, describe output; Helm release/chart `get`/`template`/`show`; Terraform output/state pull/show; Docker inspect/logs/ps/top) require approval instead of being treated as generic read-only discovery.
- `kubectl auth/config` and `terraform state/workspace` are subcommand-aware; mutating variants require approval.
- Approval hashes preserve exact shell-significant whitespace/newlines and use canonical JSON scope hashing to avoid newline ambiguity.
- Shell redirection/background/heredoc/process-substitution operators fail closed to approval-required.
- Compound shell commands fail closed to approval-required, including newline-separated commands checked before whitespace canonicalization.
- `rm`, `rm -fr`, and `terraform state rm` require approval.
- Cross-run job dependencies are rejected.
- Orchestrator/main-session role is rejected for runtime jobs.
- Scheduler claims only active-run jobs with attempts remaining.
- Lease expiry fails jobs whose max attempts are exhausted.
- Scheduler spawn mode is still disabled by default; with `--spawn --enable-spawn --isolation-backend bubblewrap`, it validates isolation, commits the claim/context before `Popen`, launches the worker, and records completion through the trusted parent broker.
- Worker subprocess env is allowlisted, strips secret-like keys, `PYTHONPATH`, `HOME`, `PATH`, `VIRTUAL_ENV`, `HERMES_HOME`, and writable runtime DB pointers, and rejects approval-control/worker-identity extra_env injection.
- Trusted worker broker materializes context only after active lease validation, writes private context JSON under a private scratch sandbox outside `HERMES_HOME`, and avoids ambient `TMPDIR` placement.
- Worker isolation preflight remains fail-closed for disabled/env-only/unknown/missing backends. Bubblewrap launch plans enforce cwd, allowlisted child env, writable scratch dirs, read-only context, network unsharing, and set `allows_spawn=true` only for the reviewed bubblewrap backend.
- `worker_main` can run role-specific `AIAgent` for non-ops worker roles only when the trusted launch env sets `HERMES_AGENT_RUNTIME_ENABLE_WORKER_EXECUTION=1` and context identity matches job/attempt/lease; `ops_worker` refuses until the mandatory command guard exists.
- Agent Runtime tools are not in the default core platform toolset; they require the explicit `agent_runtime` toolset.
- Current `worker_main` is DB-isolated: it refuses without the trusted execution gate, never opens the writable runtime DB, validates brokered context identity before role execution, and leaves result recording to the trusted parent broker/reaper.
- Exact approval lookup is job-scope aware: callers with a concrete `job_id` can use run-level approvals plus approvals scoped to that job only; callers without `job_id` see run-level approvals only.
- Ops command guard refuses missing/expired brokered context `expires_at` before command classification, approval matching, or runner execution, and it ignores inactive approval snapshots.
- Worker completion/failure and heartbeat are bound to active lease owner + attempt identity; stale workers cannot extend or complete recovered jobs.
- Completion with stale lease identity after recovery is rejected, and planned dependency children cannot be completed before promotion.
- Expired-lease recovery uses predicate-bound updates and skips rows whose lease changed concurrently.
- Parent reaper can record killed/timeout worker failure after lease expiry only when the same lease owner + attempt id still match; stale success completion after expiry remains rejected.
- `hermes runtime dispatch-once` and bounded `daemon` return non-zero when scheduler errors such as spawn refusal occur, including through the real CLI subprocess path.
- Explicit scheduler spawn path now has regressions for claim commit before `Popen`, private context materialization, successful parent-broker result recording, `max_claims`, immediate retry cleanup when `Popen` fails, post-spawn bookkeeping failure kill/retry, active-lease-predicated PID recording, reaper-exception fail/retry cleanup, and worker timeout clamping below the lease TTL.
- Runtime user-systemd packaging is recovery-only by default: the generated `hermes-agent-runtime.service` omits `--spawn`/`--enable-spawn`, quotes `HERMES_HOME`/lease-owner values to avoid directive/argument injection, rejects control-character lease-owner values, bounds `daemon-reload` with a timeout, reports `daemon-reload` failures non-zero, and leaves service start/enable as an explicit operator action.
- Runtime Obsidian sync is deterministic and documentation-only: it writes under `01 Hermes/Agent Runtime/Runs/`, stays dry-run unless `--write` is passed, keeps paths inside the selected vault and runbook directory including symlink escape/target checks, omits raw job bodies, redacts secret-like text from decisions/findings/title/public-ref paths and malformed run-id path suffixes including environment-style `*_API_KEY`/`*_TOKEN` labels and standalone provider-token shapes, returns non-zero on write failures and corrupt DB reads, avoids creating a runtime DB for missing-DB sync probes, does not write embeddings/vectors/Postgres/Qdrant directly, and explicitly says the note is not an execution queue or approval source.
- Runtime YouTrack sync is deterministic and public-mirror-only: it posts only with `--write`, omits arbitrary private/free-form runtime text, validates target issue/stage inputs, uses bounded argv-list external commands, avoids Runtime DB mutation, avoids creating a DB on missing-DB probes, and explicitly states YouTrack is not an execution queue, approval source, scheduler, or worker command channel.
- Runtime dashboard/Kanban mirror is read-only: `hermes runtime mirror` opens an existing DB in read-only mode, emits compact run/job cards with lanes/progress/alerts, omits raw job bodies plus approval command payloads, redacts secret-like values from display fields including prefixed env-style assignment names, marks cards `mirror_only=true`, and does not create or mutate Runtime DB state.
- Runtime health/alerting is read-only: `hermes runtime health` probes systemd with fixed argv, summarizes service active/substate/restarts plus DB counts and stale/expired leases, sanitizes DB-derived alert strings and DB-read error details, returns non-zero for warning/critical states, and never performs recovery or lease mutation.
- Worker sandbox cleanup is dry-run by default: `hermes runtime cleanup-sandboxes` only plans/deletes directories with the Runtime sandbox prefix under a validated temp parent, rejects `/`, symlinks/symlink components, and `HERMES_HOME` overlap, and requires explicit `--execute` before deletion.
- Runtime RAG is parent-brokered only for bounded workers: `explorer` and `scribe` receive compact cited evidence in brokered context; other worker roles do not request RAG by default; no bounded worker receives direct `personal_kb` toolsets. The broker uses only the existing local Hermes RAG service, forces safe source types/history mode, rejects secret/TEZ/Finance/HUMO/MCC/contact/business query or result indicators before worker exposure, ignores service-packed context and untrusted `payload.query`, normalizes/escapes citation metadata, uses loopback-only/proxy-disabled/no-redirect HTTP, and never writes vectors/embeddings/PG/Qdrant.
- A queued-job integration regression now runs one non-empty Runtime job through the claim/context/materialization/reaper path using a deterministic real child process, proving spawn plumbing before any live LLM worker workload is enabled.

## Next implementation slice

1. Prepare an owner-approved non-empty live LLM worker rollout after the approval channel and final security review pass.
2. Consider defense-in-depth follow-ups from the approval-channel review: approval/run-id matching in `ops_guard`, stricter packet type validation, and timeout/output clamps for guarded ops terminal.
