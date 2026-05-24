"""Deterministic gateway -> Agent Runtime router for durable L2/L3 work.

The router is intentionally opt-in via ``config.yaml``.  It creates Runtime
runs/jobs before the normal main-agent loop so long-running work is visible in
Subagent Monitor and executed by the runtime daemon instead of being silently
handled only by the gateway AIAgent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Iterable

from agent_runtime import db


_ACK_RE = re.compile(r"^(да|ок|окей|yes|yep|approve|апрув|спасибо|thanks)[\s!.]*$", re.IGNORECASE)

_RESEARCH_HINTS = (
    "проверь", "проверим", "исслед", "найди", "сравни", "проанализ", "разберись",
    "research", "investigate", "compare", "analy", "find", "audit",
)
_PROJECT_HINTS = (
    "настрой", "сделай", "реализ", "внедри", "почини", "исправ", "добавь",
    "deploy", "implement", "build", "configure", "fix", "refactor", "rewrite", "rollback",
)
_MULTI_STEP_HINTS = (
    "тест", "tests", "verify", "проверь", "dashboard", "gateway", "runtime", "router",
    "youtrack", "obsidian", "docker", "systemd", "api", "frontend", "backend", "monitor",
)
_CODE_HINTS = (
    "код", "test", "tests", "pytest", "frontend", "backend", "gateway", "runtime", "router",
    "api", "repo", "branch", "commit", "fix", "implement", "refactor", "typescript", "python",
)
_OPS_HINTS = (
    "kubernetes", "kubectl", "helm", "terraform", "systemd", "nginx", "docker compose",
    "production", "prod", "deploy", "server", "service", "restart", "перезапусти",
)
_L4_MUTATION_HINTS = (
    "kubectl apply", "kubectl delete", "helm upgrade", "terraform apply", "terraform destroy",
    "production", "prod", "прод", "restart", "перезапусти", "delete", "удали", "apply",
)
_DEFAULT_WORKSPACE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("subagent monitor", "monitor", "dashboard", "frontend", "backend"), "/opt/subagent-monitor-v2"),
    (("hermes", "runtime", "gateway", "router", "subagent", "сабагент"), "/usr/local/lib/hermes-agent"),
    (("obsidian", "обсидиан", "документац", "docs", "runbook", "note", "заметка"), "/root/.hermes/obsidian"),
)
_DEFAULT_SCRIBE_WORKSPACE = "/root/.hermes/obsidian"


@dataclass(frozen=True)
class RouteResult:
    run_id: str
    level: str
    roles: list[str]
    response_text: str
    reason: str

    def agent_result(self, *, history_len: int, user_text: str) -> dict[str, Any]:
        """Return a synthetic AIAgent-like result for gateway transcript handling."""
        return {
            "final_response": self.response_text,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": self.response_text},
            ],
            "api_calls": 0,
            "failed": False,
            "partial": False,
            "completed": True,
            "interrupted": False,
            "tools": [],
            "history_offset": history_len,
            "last_prompt_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "agent-runtime-router",
            "context_length": 0,
        }


@dataclass(frozen=True)
class WorkspaceResolution:
    """Deterministic workspace routing decision stored as job body audit text.

    Runtime DB intentionally keeps only ``workspace_kind`` and ``workspace_path``.
    The richer resolver metadata lives in the job body so old Runtime schemas and
    dashboards remain compatible while workers can still understand why a repo
    was mounted.
    """

    workspace_kind: str = "scratch"
    path: str = ""
    workspace_id: str = "scratch"
    matched_by: str = "none"
    confidence: str = "none"
    mode: str = "none"
    reason: str = ""
    role: str = ""

    def to_job_workspace(self) -> tuple[str, str]:
        return self.workspace_kind, self.path


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    low = text.lower()
    return any(needle.lower() in low for needle in needles)


def _source_allowed(cfg: dict[str, Any], *, platform_key: str, chat_id: str, thread_id: str) -> bool:
    platforms = [str(item).strip().lower() for item in _as_list(cfg.get("platforms")) if str(item).strip()]
    if platforms and platform_key.lower() not in platforms:
        return False

    allowed = _as_list(cfg.get("allowed_sources"))
    if not allowed:
        return True

    for item in allowed:
        if not isinstance(item, dict):
            continue
        item_platform = str(item.get("platform") or "").strip().lower()
        item_chat = str(item.get("chat_id") or "").strip()
        item_thread = str(item.get("thread_id") or "").strip()
        if item_platform and item_platform != platform_key.lower():
            continue
        if item_chat and item_chat != str(chat_id or ""):
            continue
        if item_thread and item_thread != str(thread_id or ""):
            continue
        return True
    return False


def classify_level(message_text: str) -> tuple[str, str]:
    """Heuristic L-level classifier for the deterministic pre-agent router.

    It is deliberately conservative: simple questions and acknowledgements stay
    in the main session; production mutations stay L4 and are not auto-routed by
    the L2/L3 router.
    """
    text = (message_text or "").strip()
    low = text.lower()
    if not text or text.startswith("/") or _ACK_RE.match(text):
        return "L0", "command_or_ack"

    if _contains_any(low, _L4_MUTATION_HINTS) and _contains_any(low, _OPS_HINTS):
        return "L4", "production_mutation_hint"

    has_project = _contains_any(low, _PROJECT_HINTS)
    has_research = _contains_any(low, _RESEARCH_HINTS)
    multi_markers = sum(1 for needle in _MULTI_STEP_HINTS if needle.lower() in low)
    code_markers = sum(1 for needle in _CODE_HINTS if needle.lower() in low)
    ops_markers = sum(1 for needle in _OPS_HINTS if needle.lower() in low)
    long_enough = len(text) >= 90

    if has_project and (multi_markers >= 2 or code_markers >= 2 or ops_markers >= 1 or long_enough):
        return "L3", "project_complexity"
    if has_research and (long_enough or multi_markers >= 1):
        return "L2", "durable_research"
    return "L0", "simple_or_conversational"


_BROAD_WORKSPACE_IDS = {"all", "home", "root", "tmp"}
_SECRET_PATH_MARKERS = (".env", ".git-credentials", ".gnupg", ".kube", ".ssh", "auth.json", "credential", "secret", "token")
_SECRET_CHILD_EXACT = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    ".git-credentials",
    "auth.json",
    "credentials",
    "credentials.json",
    "secret",
    "secrets",
    "token.json",
}
_BROAD_PATHS = {
    Path("/"),
    Path("/home"),
    Path("/opt"),
    Path("/root"),
    Path("/tmp"),
    Path("/usr"),
    Path("/usr/local"),
    Path("/var"),
}
_WORKER_WRITE_ROLES = {"code_worker", "ops_worker", "scribe"}


def _workspace_mode(role: str) -> str:
    return "writable_intent" if role in _WORKER_WRITE_ROLES else "read_only_intent"


def _existing_dir(raw_path: Any) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    return str(path) if path.is_absolute() and path.is_dir() and not path.is_symlink() else ""


def _path_guardrail_reason(path: Path, *, workspace_id: str = "", aliases: Iterable[str] = ()) -> str:
    ids = {str(workspace_id or "").strip().lower()}
    ids.update(str(alias or "").strip().lower() for alias in aliases)
    ids.discard("")
    if ids.intersection(_BROAD_WORKSPACE_IDS):
        return "broad_workspace_identifier"

    resolved = path.expanduser()
    if resolved in _BROAD_PATHS:
        return "broad_workspace_path"
    try:
        if resolved == Path.home().expanduser():
            return "broad_workspace_path"
    except RuntimeError:
        pass

    lowered_parts = [part.lower() for part in resolved.parts]
    if any(any(marker in part for marker in _SECRET_PATH_MARKERS) for part in lowered_parts):
        return "secret_path_guardrail"
    try:
        for child in resolved.iterdir():
            child_name = child.name.lower()
            if child_name in _SECRET_CHILD_EXACT:
                return "secret_child_guardrail"
    except OSError as exc:
        return f"workspace_child_scan_failed:{type(exc).__name__}"
    return ""


def _validated_workspace_path(raw_path: Any, *, workspace_id: str = "", aliases: Iterable[str] = ()) -> tuple[str, str]:
    text = str(raw_path or "").strip()
    if not text:
        return "", "missing_workspace_path"
    path = Path(text).expanduser()
    if not path.is_absolute():
        return "", "workspace_path_not_absolute"
    if path.is_symlink():
        return "", "workspace_path_symlink"
    if not path.is_dir():
        return "", "workspace_path_not_existing_dir"
    guardrail = _path_guardrail_reason(path, workspace_id=workspace_id, aliases=aliases)
    if guardrail:
        return "", guardrail
    return str(path), ""


def _keyword_matches(text: str, keyword: str) -> bool:
    needle = str(keyword or "").strip().lower()
    if not needle:
        return False
    if re.fullmatch(r"[\wа-яё-]+", needle, flags=re.IGNORECASE) and len(needle) <= 3:
        return bool(re.search(rf"(?<![0-9a-zа-яё_]){re.escape(needle)}(?![0-9a-zа-яё_])", text, flags=re.IGNORECASE))
    return needle in text


def _workspace_resolution(
    *,
    role: str,
    workspace_id: str,
    raw_path: Any,
    aliases: Iterable[str],
    matched_by: str,
    confidence: str,
) -> WorkspaceResolution:
    path, reason = _validated_workspace_path(raw_path, workspace_id=workspace_id, aliases=aliases)
    if not path:
        return WorkspaceResolution(
            workspace_kind="scratch",
            workspace_id=workspace_id or "scratch",
            matched_by=matched_by,
            confidence="none",
            mode="none",
            reason=reason,
            role=role,
        )
    return WorkspaceResolution(
        workspace_kind="dir",
        path=path,
        workspace_id=workspace_id or Path(path).name,
        matched_by=matched_by,
        confidence=confidence,
        mode=_workspace_mode(role),
        reason="matched",
        role=role,
    )


def _structured_workspace_matches(item: dict[str, Any], *, role: str, low_text: str) -> tuple[bool, str, list[str]]:
    workspace_id = str(item.get("workspace_id") or item.get("id") or item.get("name") or "").strip()
    aliases = [
        str(value).strip().lower()
        for value in _as_list(item.get("aliases") or item.get("match") or item.get("matches") or item.get("keywords"))
        if str(value).strip()
    ]
    if workspace_id:
        aliases.append(workspace_id.lower())

    roles = [str(value).strip() for value in _as_list(item.get("default_roles") or item.get("roles")) if str(value).strip()]
    if roles and role not in roles:
        return False, "", aliases

    for alias in aliases:
        if _keyword_matches(low_text, alias):
            return True, f"alias:{alias}", aliases
    return False, "", aliases


def resolve_workspace(*, cfg: dict[str, Any], role: str, message_text: str) -> WorkspaceResolution:
    """Resolve a Runtime job workspace without changing the Runtime DB schema.

    Precedence is deterministic:
    1. ``scribe`` always uses the docs/Obsidian workspace when configured.
    2. Structured ``subagent_router.workspaces`` entries by alias/id and role.
    3. Backward-compatible ``workspace_rules`` entries.
    4. Built-in Hermes/monitor/Obsidian fallback rules.
    5. Scratch with explicit no-match metadata.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    role = str(role or "").strip()
    low = (message_text or "").lower()

    if role == "scribe":
        raw_scribe = cfg.get("scribe_workspace") or _DEFAULT_SCRIBE_WORKSPACE
        resolution = _workspace_resolution(
            role=role,
            workspace_id="obsidian",
            raw_path=raw_scribe,
            aliases=["obsidian", "docs", "documentation"],
            matched_by="role:scribe",
            confidence="high",
        )
        if resolution.workspace_kind == "dir":
            return resolution

    for item in _as_list(cfg.get("workspaces")):
        if not isinstance(item, dict):
            continue
        matched, matched_by, aliases = _structured_workspace_matches(item, role=role, low_text=low)
        if not matched:
            continue
        workspace_id = str(item.get("workspace_id") or item.get("id") or item.get("name") or "").strip()
        resolution = _workspace_resolution(
            role=role,
            workspace_id=workspace_id or (aliases[0] if aliases else "workspace"),
            raw_path=item.get("path") or item.get("workspace_path"),
            aliases=aliases,
            matched_by=matched_by,
            confidence="high",
        )
        return resolution

    for item in _as_list(cfg.get("workspace_rules")):
        if not isinstance(item, dict):
            continue
        needles = [
            str(value).strip().lower()
            for value in _as_list(item.get("match") or item.get("matches") or item.get("keywords"))
            if str(value).strip()
        ]
        for needle in needles:
            if _keyword_matches(low, needle):
                return _workspace_resolution(
                    role=role,
                    workspace_id=str(item.get("workspace_id") or item.get("id") or Path(str(item.get("path") or item.get("workspace_path") or "workspace")).name),
                    raw_path=item.get("path") or item.get("workspace_path"),
                    aliases=needles,
                    matched_by=f"workspace_rules:{needle}",
                    confidence="high",
                )

    for needles, path in _DEFAULT_WORKSPACE_RULES:
        for needle in needles:
            if _keyword_matches(low, needle):
                return _workspace_resolution(
                    role=role,
                    workspace_id=Path(path).name,
                    raw_path=path,
                    aliases=needles,
                    matched_by=f"default:{needle}",
                    confidence="medium",
                )

    return WorkspaceResolution(reason="no_workspace_match", role=role)


def _workspace_for_role(*, cfg: dict[str, Any], role: str, message_text: str) -> tuple[str, str]:
    return resolve_workspace(cfg=cfg, role=role, message_text=message_text).to_job_workspace()


def _job_plan(level: str, message_text: str) -> list[tuple[str, str, list[int]]]:
    low = message_text.lower()
    if level == "L2":
        return [
            ("explorer", "Research / discovery", []),
            ("scribe", "Owner-facing synthesis", [0]),
        ]

    worker_role = "ops_worker" if _contains_any(low, _OPS_HINTS) and not _contains_any(low, _CODE_HINTS) else "code_worker"
    return [
        ("explorer", "Discovery and constraints", []),
        (worker_role, "Implementation / execution", []),
        ("sentinel", "Independent verification", [0, 1]),
        ("scribe", "Owner-facing synthesis", [2]),
    ]


def _compact_title(message_text: str, *, prefix: str) -> str:
    title = re.sub(r"\s+", " ", (message_text or "").strip())
    if len(title) > 90:
        title = title[:87].rstrip() + "..."
    return f"{prefix}: {title or 'Untitled task'}"


def _workspace_audit_lines(resolution: WorkspaceResolution | None) -> list[str]:
    if resolution is None:
        return []
    return [
        "",
        "workspace_resolution:",
        f"  workspace_id: {resolution.workspace_id}",
        f"  workspace_kind: {resolution.workspace_kind}",
        f"  workspace_path: {resolution.path}",
        f"  matched_by: {resolution.matched_by}",
        f"  confidence: {resolution.confidence}",
        f"  mode: {resolution.mode}",
        f"  reason: {resolution.reason}",
    ]


def _job_body(
    *,
    level: str,
    role: str,
    title: str,
    message_text: str,
    owner_source: str,
    workspace_resolution: WorkspaceResolution | None = None,
) -> str:
    lines = [
        f"level: {level}",
        f"role: {role}",
        f"owner_source: {owner_source}",
        "goal:",
        message_text.strip(),
        "",
        "constraints:",
        "- Keep owner-facing output concise and semantic.",
        "- Do not expose internal runtime IDs unless explicitly asked.",
        "- Do not perform production mutations without an exact approval packet.",
        "- Return summary,evidence,artifacts,risks,next.",
        "",
        f"task: {title}",
    ]
    lines.extend(_workspace_audit_lines(workspace_resolution))
    return "\n".join(lines)


def create_runtime_route(
    *,
    level: str,
    title: str,
    objective: str,
    owner_source: str,
    orchestrator_session_id: str = "",
    public_ref: str = "auto-router",
    router_config: dict[str, Any] | None = None,
) -> RouteResult:
    """Create a Runtime run/job graph after the main agent decides L2/L3 routing."""
    level = str(level or "").strip().upper()
    if level not in {"L2", "L3"}:
        raise ValueError("runtime smart router only supports L2/L3 auto routes")
    message_text = (objective or title or "").strip()
    run_title = _compact_title(title or message_text, prefix=level)
    cfg = router_config if isinstance(router_config, dict) else {}

    db.init_db()
    roles: list[str] = []
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title=run_title,
            objective=message_text,
            owner_source=owner_source,
            public_ref=public_ref or "auto-router",
            risk_level="medium" if level == "L2" else "high",
            orchestrator_session_id=orchestrator_session_id,
        )
        job_ids: list[str] = []
        base_now = int(time.time())
        for index, (role, job_title, parent_indexes) in enumerate(_job_plan(level, message_text)):
            deps = [job_ids[parent] for parent in parent_indexes]
            workspace_resolution = resolve_workspace(cfg=cfg, role=role, message_text=message_text)
            workspace_kind, workspace_path = workspace_resolution.to_job_workspace()
            job_id = db.create_job(
                conn,
                run_id=run_id,
                role=role,
                title=f"{job_title}: {run_title}",
                body=_job_body(
                    level=level,
                    role=role,
                    title=job_title,
                    message_text=message_text,
                    owner_source=owner_source,
                    workspace_resolution=workspace_resolution,
                ),
                depends_on=deps,
                priority=max(0, 100 - index),
                workspace_kind=workspace_kind,
                workspace_path=workspace_path,
                idempotency_key=f"smart-router:{role}:{index}",
                now=base_now + index,
            )
            job_ids.append(job_id)
            roles.append(role)
        conn.commit()

    response = (
        f"Принял. Главный агент классифицировал задачу как {level} и запустил "
        f"Runtime subagents: {', '.join(roles)}.\n"
        "Runtime daemon должен подхватить jobs, а dashboard покажет executing subagents/processes."
    )
    return RouteResult(run_id=run_id, level=level, roles=roles, response_text=response, reason="main_agent_decision")


def maybe_create_runtime_route(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str,
    chat_id: str,
    thread_id: str,
    session_id: str,
    message_text: str,
    owner_source: str,
) -> RouteResult | None:
    """Create an Agent Runtime run/jobs for enabled L2/L3 gateway messages."""
    cfg = ((user_config or {}).get("subagent_router") or {})
    if not isinstance(cfg, dict) or not _truthy(cfg.get("enabled")):
        return None
    if not _source_allowed(cfg, platform_key=platform_key, chat_id=str(chat_id or ""), thread_id=str(thread_id or "")):
        return None

    level, reason = classify_level(message_text)
    allowed_levels = [str(item).strip().upper() for item in _as_list(cfg.get("levels") or ["L2", "L3"])]
    if level not in allowed_levels:
        return None

    return create_runtime_route(
        level=level,
        title=message_text,
        objective=message_text,
        owner_source=owner_source,
        orchestrator_session_id=session_id,
        public_ref=str(cfg.get("public_ref") or "auto-router"),
        router_config=cfg,
    )
