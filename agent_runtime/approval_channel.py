"""Trusted operator approval channel helpers for Agent Runtime.

This module intentionally does not register any model-callable tool and does not
expose an approval-writer capability.  It only builds and validates exact-scope
approval packet payloads for the reviewed operator CLI path.
"""

from __future__ import annotations

import time
from typing import Any

from . import policy

OPERATOR_CONFIRM_PHRASE = "APPROVE_RUNTIME_APPROVAL"
TRUSTED_APPROVAL_SOURCES = {
    "operator",
    "operator-cli",
    "runtime-operator-cli",
    "runtime-verify-cli",
    "telegram-owner",
}
MIRROR_APPROVAL_SOURCE_LABELS = ("youtrack", "obsidian", "rag", "mirror", "dashboard", "kanban")


def validate_approval_source(approval_source: str) -> str:
    source = str(approval_source or "").strip()
    if not source:
        raise ValueError("approval packet approval_source is required")
    lower = source.lower()
    if any(label in lower for label in MIRROR_APPROVAL_SOURCE_LABELS):
        raise ValueError("approval_source must be a trusted operator channel, not a mirror/status surface")
    if source not in TRUSTED_APPROVAL_SOURCES:
        raise ValueError("approval_source must be one of the trusted operator channel labels")
    return source


def build_operator_approval_packet(
    *,
    target: str,
    commands: list[str],
    reason: str,
    blast_radius: str,
    rollback: str,
    verification: list[str],
    approved_by: str,
    approval_source: str = "operator-cli",
    expires_in_seconds: int | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Build an exact-scope approval packet for operator review/write paths."""
    ts = int(time.time() if now is None else now)
    expires_at = None
    if expires_in_seconds is not None:
        ttl = int(expires_in_seconds)
        if ttl <= 0:
            raise ValueError("expires_in_seconds must be positive")
        expires_at = ts + ttl
    if not str(reason or "").strip():
        raise ValueError("approval reason is required")
    if not str(blast_radius or "").strip():
        raise ValueError("approval blast_radius is required")
    if not str(rollback or "").strip():
        raise ValueError("approval rollback is required")
    verification_items = [str(item) for item in (verification or [])]
    if not verification_items or any(not item.strip() for item in verification_items):
        raise ValueError("approval verification is required")
    packet = policy.build_approval_packet(
        target=str(target or ""),
        commands=[str(command) for command in (commands or [])],
        reason=str(reason or ""),
        blast_radius=str(blast_radius or ""),
        rollback=str(rollback or ""),
        verification=verification_items,
        approved_by=str(approved_by or ""),
        approval_source=validate_approval_source(approval_source or "operator-cli"),
        expires_at=expires_at,
    )
    packet["approved_at"] = ts
    return packet
