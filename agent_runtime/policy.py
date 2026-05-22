"""Policy and approval primitives for Hermes Agent Runtime.

This module deliberately stays deterministic.  LLM Sentinel passes can add
additional findings, but production-sensitive mutations must first pass this
small exact-scope policy layer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import shlex
import time
from typing import Any, Iterable


@dataclass(frozen=True)
class CommandVerdict:
    command: str
    category: str
    allowed_without_approval: bool
    requires_approval: bool
    reason: str
    matched: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["matched"] = list(self.matched)
        return data


def canonical_command(command: str) -> str:
    return " ".join((command or "").strip().split())


def command_hash(command: str) -> str:
    # Approval packets are exact-scope. Preserve shell-significant whitespace,
    # quotes, and newlines instead of using canonical_command(), which is only
    # for display/classification.
    return hashlib.sha256((command or "").encode("utf-8")).hexdigest()


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return canonical_command(command).split()


def _has_any(command_lc: str, needles: Iterable[str]) -> tuple[str, ...]:
    return tuple(n for n in needles if n in command_lc)


def _shell_control_matches(canonical: str) -> tuple[str, ...]:
    """Return shell control operators that make command policy fail-closed.

    The runtime policy is intentionally conservative.  A single string may
    contain a read-only command followed by a mutating command (`kubectl get ...
    && kubectl scale ...`).  Until the runtime has a proper shell AST, any
    sequence/pipeline/substitution operator requires an approval packet.
    """
    matches: list[str] = []
    for needle in ("&&", "||", ">>", "2>", "<<", "<(", ";", "|", "`", "$(", ">", "<", "&", "\n"):
        if needle in canonical:
            matches.append(needle)
    return tuple(matches)


_DESTRUCTIVE_NEEDLES = (
    "rm -rf",
    " rmdir ",
    " delete ",
    " destroy",
    " prune",
    " drop database",
    " truncate ",
    " wipe",
    " purge",
)

_KUBECTL_MUTATING = {
    "apply", "delete", "patch", "scale", "replace", "create", "edit",
    "label", "annotate", "cordon", "uncordon", "drain", "taint", "rollout",
}
_KUBECTL_READONLY = {"get", "describe", "logs", "top", "version", "api-resources", "api-versions", "explain"}
_HELM_MUTATING = {"upgrade", "install", "uninstall", "delete", "rollback", "repo", "plugin"}
_HELM_READONLY = {"list", "status", "get", "template", "history", "show", "version", "env", "lint"}
_TERRAFORM_MUTATING = {"apply", "destroy", "import"}
_TERRAFORM_READONLY = {"plan", "show", "output", "validate", "fmt", "version", "providers"}
_DOCKER_MUTATING = {"rm", "rmi", "restart", "kill", "stop", "start", "compose", "system"}
_DOCKER_READONLY = {"ps", "logs", "inspect", "images", "info", "version", "stats", "top"}
_KUBECTL_SENSITIVE_GET_RESOURCES = {"secret", "secrets", "configmap", "configmaps", "cm"}
_KUBECTL_SENSITIVE_OUTPUT_FLAGS = {"-o", "--output"}


def _first_positional_after_binary(tokens: list[str], binary: str) -> str:
    """Return the command verb after flags such as `kubectl -n prod get`."""
    try:
        idx = tokens.index(binary)
    except ValueError:
        return ""
    i = idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("-"):
            # Skip common flag values for flags that take a following argument.
            if token in {"-n", "--namespace", "-f", "--filename", "-l", "--selector", "--context", "--kubeconfig"}:
                i += 2
            else:
                i += 1
            continue
        return token.lower()
    return ""


def _token_after(tokens: list[str], marker: str) -> str:
    try:
        idx = tokens.index(marker)
    except ValueError:
        return ""
    return tokens[idx + 1].lower() if idx + 1 < len(tokens) else ""


def _kubectl_get_resource(tokens: list[str]) -> str:
    try:
        idx = tokens.index("get")
    except ValueError:
        return ""
    i = idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("-"):
            if token in {"-n", "--namespace", "-l", "--selector", "--context", "--kubeconfig", "-o", "--output"}:
                i += 2
            else:
                i += 1
            continue
        return token.split("/", 1)[0].lower()
    return ""


def _has_structured_output_flag(tokens: list[str]) -> bool:
    if any(t in _KUBECTL_SENSITIVE_OUTPUT_FLAGS for t in tokens):
        return True
    return any(
        t.startswith("-o=")
        or (t.startswith("-o") and len(t) > 2)
        or t.startswith("--output=")
        or t == "--template"
        or t.startswith("--template=")
        for t in tokens
    )


def _has_kubectl_raw_flag(tokens: list[str]) -> bool:
    return any(t == "--raw" or t.startswith("--raw=") for t in tokens)


def classify_command(command: str) -> CommandVerdict:
    raw_command = command or ""
    shell_controls = _shell_control_matches(raw_command)
    canonical = canonical_command(raw_command)
    if shell_controls:
        return CommandVerdict(
            command=canonical,
            category="compound_shell",
            allowed_without_approval=False,
            requires_approval=True,
            reason="Compound shell commands require an approval packet so read-only and mutating segments cannot be mixed.",
            matched=shell_controls,
        )
    lc = f" {canonical.lower()} "
    tokens = [t.lower() for t in _tokens(canonical)]
    executable = tokens[0] if tokens else ""
    matched = _has_any(lc, _DESTRUCTIVE_NEEDLES)
    if matched:
        return CommandVerdict(
            command=canonical,
            category="destructive",
            allowed_without_approval=False,
            requires_approval=True,
            reason="Destructive command requires an approval packet.",
            matched=matched,
        )

    if "rm" in tokens:
        return CommandVerdict(canonical, "destructive", False, True, "rm deletes filesystem state and requires an approval packet.", ("rm",))

    if executable == "kubectl":
        verb = _first_positional_after_binary(tokens, "kubectl")
        if verb == "logs":
            return CommandVerdict(canonical, "sensitive_read", False, True, "kubectl logs can disclose sensitive application data and require an approval packet.", ("logs",))
        if verb in {"get", "describe"} and any("secret" in t for t in tokens):
            return CommandVerdict(canonical, "sensitive_read", False, True, "kubectl secret reads require an approval packet.", (verb, "secret"))
        if verb == "describe":
            return CommandVerdict(canonical, "sensitive_read", False, True, "kubectl describe can disclose sensitive runtime configuration and requires an approval packet.", ("describe",))
        if verb == "get":
            if _has_kubectl_raw_flag(tokens):
                return CommandVerdict(canonical, "sensitive_read", False, True, "kubectl raw API reads can disclose secret-bearing resources and require an approval packet.", ("get", "--raw"))
            resource = _kubectl_get_resource(tokens)
            if resource in _KUBECTL_SENSITIVE_GET_RESOURCES or _has_structured_output_flag(tokens):
                return CommandVerdict(canonical, "sensitive_read", False, True, "kubectl get of secret-bearing resources or structured specs requires an approval packet.", ("get", resource or "structured-output"))
        if verb == "auth":
            sub = _token_after(tokens, "auth")
            if sub in {"can-i", "whoami"}:
                return CommandVerdict(canonical, "read_only", True, False, "kubectl auth read-only discovery command.", (f"auth {sub}",))
            return CommandVerdict(canonical, "prod_mutation", False, True, "kubectl auth mutation requires an approval packet.", (f"auth {sub or '*'}",))
        if verb == "config":
            sub = _token_after(tokens, "config")
            if sub == "view" and _has_kubectl_raw_flag(tokens):
                return CommandVerdict(canonical, "sensitive_read", False, True, "kubectl config view --raw can disclose credentials and requires an approval packet.", ("config view --raw",))
            if sub in {"view", "current-context", "get-contexts"}:
                return CommandVerdict(canonical, "read_only", True, False, "kubectl config read-only discovery command.", (f"config {sub}",))
            return CommandVerdict(canonical, "prod_mutation", False, True, "kubectl config mutation requires an approval packet.", (f"config {sub or '*'}",))
        if verb in _KUBECTL_MUTATING:
            return CommandVerdict(canonical, "prod_mutation", False, True, "kubectl mutation requires an approval packet.", (verb,))
        if verb in _KUBECTL_READONLY:
            return CommandVerdict(canonical, "read_only", True, False, "kubectl read-only discovery command.", (verb,))
        return CommandVerdict(canonical, "unknown_ops", False, True, "Unknown kubectl verb requires an approval packet.", (verb or "kubectl",))

    if executable == "helm":
        verb = _first_positional_after_binary(tokens, "helm")
        if verb in _HELM_MUTATING:
            return CommandVerdict(canonical, "prod_mutation", False, True, "helm mutation requires an approval packet.", (verb,))
        if verb in {"get", "template", "show"}:
            sub = _token_after(tokens, "get")
            matched = f"{verb} {sub}" if sub else verb
            return CommandVerdict(canonical, "sensitive_read", False, True, "helm rendered output or chart/release values can disclose secrets and require an approval packet.", (matched,))
        if verb in _HELM_READONLY:
            return CommandVerdict(canonical, "read_only", True, False, "helm read-only discovery command.", (verb,))
        return CommandVerdict(canonical, "unknown_ops", False, True, "Unknown helm verb requires an approval packet.", (verb or "helm",))

    if executable == "terraform":
        verb = _first_positional_after_binary(tokens, "terraform")
        if verb == "state":
            sub = _token_after(tokens, "state")
            if sub in {"show", "pull"}:
                return CommandVerdict(canonical, "sensitive_read", False, True, "terraform state reads can disclose secrets and require an approval packet.", (f"state {sub}",))
            if sub in {"list"}:
                return CommandVerdict(canonical, "read_only", True, False, "terraform state read-only command.", (f"state {sub}",))
            return CommandVerdict(canonical, "prod_mutation", False, True, "terraform state mutation requires an approval packet.", (f"state {sub or '*'}",))
        if verb in {"output", "show"}:
            return CommandVerdict(canonical, "sensitive_read", False, True, "terraform output/show can disclose secrets and require an approval packet.", (verb,))
        if verb == "workspace":
            sub = _token_after(tokens, "workspace")
            if sub in {"list", "show"}:
                return CommandVerdict(canonical, "read_only", True, False, "terraform workspace read-only command.", (f"workspace {sub}",))
            return CommandVerdict(canonical, "prod_mutation", False, True, "terraform workspace mutation requires an approval packet.", (f"workspace {sub or '*'}",))
        if verb in _TERRAFORM_MUTATING:
            return CommandVerdict(canonical, "prod_mutation", False, True, "terraform mutation requires an approval packet.", (verb,))
        if verb in _TERRAFORM_READONLY:
            return CommandVerdict(canonical, "read_only", True, False, "terraform read-only command.", (verb,))
        return CommandVerdict(canonical, "unknown_ops", False, True, "Unknown terraform verb requires an approval packet.", (verb or "terraform",))

    if executable == "docker":
        verb = _first_positional_after_binary(tokens, "docker")
        if verb == "system" and any(t == "prune" for t in tokens):
            return CommandVerdict(canonical, "destructive", False, True, "docker system prune requires an approval packet.", ("system prune",))
        if verb in {"inspect", "logs", "ps", "top"}:
            return CommandVerdict(canonical, "sensitive_read", False, True, "docker process/container inspection can disclose secrets and requires an approval packet.", (verb,))
        if verb in _DOCKER_MUTATING:
            return CommandVerdict(canonical, "mutation", False, True, "docker mutation requires an approval packet.", (verb,))
        if verb in _DOCKER_READONLY:
            return CommandVerdict(canonical, "read_only", True, False, "docker read-only command.", (verb,))

    return CommandVerdict(canonical, "unknown_command", False, True, "Unknown command requires an approval packet unless explicitly allowlisted as read-only.", ())


def approval_scope_hash(target: str, commands: list[str]) -> str:
    payload = {
        "target": target or "",
        "commands": [command or "" for command in commands],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_approval_packet(
    *,
    target: str,
    commands: list[str],
    reason: str,
    blast_radius: str,
    rollback: str,
    verification: list[str],
    approved_by: str,
    approval_source: str = "operator",
    expires_at: int | None = None,
) -> dict[str, Any]:
    exact_commands = [c or "" for c in commands]
    return {
        "target": target,
        "commands": exact_commands,
        "command_hashes": [command_hash(c) for c in exact_commands],
        "reason": reason,
        "blast_radius": blast_radius,
        "rollback": rollback,
        "verification": verification,
        "approved_by": approved_by,
        "approval_source": approval_source,
        "approved_at": int(time.time()),
        "expires_at": expires_at,
        "scope_hash": approval_scope_hash(target, exact_commands),
    }


def approval_allows_command(approval: dict[str, Any], command: str, *, target: str, now: int | None = None) -> bool:
    if not approval:
        return False
    if approval.get("target") != target:
        return False
    expires_at = approval.get("expires_at")
    if expires_at is not None and int(expires_at) < int(now or time.time()):
        return False
    commands = [str(c) for c in (approval.get("commands") or [])]
    if command not in commands:
        return False
    expected_hashes = [command_hash(c) for c in commands]
    if list(approval.get("command_hashes") or []) != expected_hashes:
        return False
    if approval.get("scope_hash") != approval_scope_hash(target, commands):
        return False
    return command_hash(command) in set(expected_hashes)
