"""ApprovalManager — same-process human-in-the-loop tool gate (§3.6).

The wire_driver atom, on a ``ToolCallEvent`` for a tool the operator
flagged write-class, calls :meth:`ApprovalManager.request`. That renders
an ``approval_request`` outbound card (two buttons, Approve / Deny) and
awaits an :class:`asyncio.Future`. The button click round-trips as an
``inbound`` envelope carrying ``button_value`` (``"<approval_id>:approve"``);
the gateway routes it to :meth:`ApprovalManager.resolve`, which resolves
the future.

There is no cross-process plumbing: no MessageBus, no worker, no
``root_session_key`` rewriting. The future lives in this process and the
session loop awaits it directly.

Identity check: only the original requester's ``sender_id`` may resolve
their approval; a click from anyone else is silently dropped (§3.6).
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

_APPROVE = "approve"
_APPROVE_TOOL = "approve_tool"
_APPROVE_SESSION = "approve_session"
_DENY = "deny"
_VALUE_SEP = ":"  # "<approval_id>:<approve|approve_tool|approve_session|deny>"

APPROVAL_TIMEOUT_S: float = 300.0

# The sink takes a single ``outbound`` body dict (§2.5) and ships it.
OutboundSink = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class _PendingApproval:
    future: asyncio.Future[bool]
    requester_sender_id: str
    session_key: str
    tool_name: str
    tool_args: dict[str, Any]
    tool_call_id: str


class ApprovalManager:
    """Per-tool-call future map; renders cards, resolves on click."""

    def __init__(
        self,
        outbound_sink: OutboundSink,
        *,
        require_approval: frozenset[str] = frozenset(),
        always_block: frozenset[str] = frozenset(),
        timeout_seconds: float = APPROVAL_TIMEOUT_S,
    ) -> None:
        self._sink = outbound_sink
        self._require = require_approval
        self._block = always_block
        self._timeout = timeout_seconds
        # approval_id -> pending approval data.
        self._pending: dict[str, _PendingApproval] = {}
        # Session-scoped grants created by terminal approval choices.
        self._session_approved_all: set[str] = set()
        self._session_approved_patterns: set[tuple[str, str]] = set()
        # Session-scoped rules created by the terminal /permissions UI.
        self._session_rules: dict[str, dict[str, set[str]]] = {}

    # -- policy -------------------------------------------------------

    def requires(
        self,
        tool_name: str,
        *,
        session_key: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> bool:
        """True if a button press is needed before ``tool_name`` runs.

        ``always_block`` tools also "require" approval — but
        :meth:`request` short-circuits them to a denial without a card.
        ``"*"`` in ``require_approval`` means every tool not allowed
        elsewhere needs approval. Session-approved tools skip.
        """
        if tool_name in self._block:
            return True
        rule = self._session_rule_decision(
            session_key=session_key or "",
            tool_name=tool_name,
            tool_args=tool_args or {},
        )
        if rule == "deny":
            return True
        if rule == "ask":
            return True
        if rule == "allow":
            return False
        if self._is_session_approved(
            session_key=session_key or "",
            tool_name=tool_name,
            tool_args=tool_args or {},
        ):
            return False
        return tool_name in self._require or "*" in self._require

    # -- request / resolve -------------------------------------------

    async def request(
        self,
        *,
        session_key: str,
        sender_id: str,
        channel: str,
        chat_id: str,
        thread_id: str | None,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str = "",
    ) -> bool:
        """Block until the user approves or denies ``tool_name``.

        Returns ``True`` to proceed, ``False`` to deny (explicit deny,
        timeout, or always-block policy).
        """
        if tool_name in self._block:
            return False
        rule = self._session_rule_decision(
            session_key=session_key,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        if rule == "deny":
            return False
        if rule == "allow":
            return True
        if self._is_session_approved(
            session_key=session_key,
            tool_name=tool_name,
            tool_args=tool_args,
        ):
            return True
        approval_id = f"appr-{uuid.uuid4().hex[:12]}"
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = _PendingApproval(
            future=future,
            requester_sender_id=sender_id,
            session_key=session_key,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
        )
        await self._sink(
            self._render_card(
                approval_id=approval_id,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                thread_id=thread_id,
                sender_id=sender_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
            )
        )
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except TimeoutError:
            self._pending.pop(approval_id, None)
            await self._sink(
                self._render_resolution(
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    tool_name=tool_name,
                    decision=_DENY,
                    by_user="timeout",
                    approval_id=approval_id,
                )
            )
            return False

    def resolve(self, button_value: str, clicker_sender_id: str) -> bool:
        """Resolve a pending approval from a button click.

        Returns ``True`` if a pending future was resolved, ``False`` for
        a stale / already-resolved id or an identity mismatch (silently
        dropped per §3.6).
        """
        if _VALUE_SEP not in button_value:
            return False
        approval_id, _, decision = button_value.partition(_VALUE_SEP)
        if decision not in (_APPROVE, _APPROVE_TOOL, _APPROVE_SESSION, _DENY):
            return False
        entry = self._pending.get(approval_id)
        if entry is None:
            return False  # stale, already resolved or timed out
        if clicker_sender_id != entry.requester_sender_id:
            logger.info(
                f"approval {approval_id} click by {clicker_sender_id} ignored "
                f"(expected {entry.requester_sender_id})"
            )
            return False  # identity mismatch, silently drop
        self._pending.pop(approval_id, None)
        approved = decision in (_APPROVE, _APPROVE_TOOL, _APPROVE_SESSION)
        if approved and decision == _APPROVE_TOOL:
            self._session_approved_patterns.add(
                (
                    entry.session_key,
                    _permission_pattern(entry.tool_name, entry.tool_args),
                )
            )
        if approved and decision == _APPROVE_SESSION:
            self._session_approved_all.add(entry.session_key)
        if not entry.future.done():
            entry.future.set_result(approved)
        return True

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add_session_rule(self, session_key: str, kind: str, pattern: str) -> bool:
        """Record a manual permission rule for one terminal session."""
        session_key = str(session_key or "")
        kind = str(kind or "").strip().lower()
        pattern = str(pattern or "").strip()
        if not session_key or not pattern or kind not in {"allow", "ask", "deny"}:
            return False
        rules = self._session_rules.setdefault(
            session_key,
            {"allow": set(), "ask": set(), "deny": set()},
        )
        rules[kind].add(pattern)
        return True

    def pending_for_session(self, session_key: str) -> list[str]:
        """List still-pending interaction ids for a session key.

        ``ApprovalManager`` is gateway-local and all approvals route through
        the same process, so this can be projected into session snapshots
        without asking core for new APIs.
        """
        return [
            approval_id
            for approval_id, pending in self._pending.items()
            if pending.session_key == session_key
        ]

    def _is_session_approved(
        self,
        *,
        session_key: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> bool:
        if not session_key:
            return False
        if session_key in self._session_approved_all:
            return True
        pattern = _permission_pattern(tool_name, tool_args)
        return (session_key, pattern) in self._session_approved_patterns

    def _session_rule_decision(
        self,
        *,
        session_key: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str | None:
        rules = self._session_rules.get(session_key)
        if not rules:
            return None
        for kind in ("deny", "ask", "allow"):
            if any(
                _permission_rule_matches(rule, tool_name, tool_args)
                for rule in rules.get(kind, set())
            ):
                return kind
        return None

    # -- rendering ----------------------------------------------------

    def _render_card(
        self,
        *,
        approval_id: str,
        session_key: str,
        channel: str,
        chat_id: str,
        thread_id: str | None,
        sender_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        try:
            args = json.dumps(
                tool_args, indent=2, sort_keys=True, ensure_ascii=False
            )
        except (TypeError, ValueError):
            args = repr(tool_args)
        if len(args) > 1500:
            args = args[:1500] + "\n…"
        body: dict[str, Any] = {
            "channel": channel,
            "chat_id": chat_id,
            "content": (
                f"Approve `{tool_name}`? (requested by {sender_id})\n"
                f"```json\n{args}\n```"
            ),
            "buttons": [
                {
                    "label": "Approve",
                    "value": f"{approval_id}{_VALUE_SEP}{_APPROVE}",
                    "style": "primary",
                },
                {
                    "label": "Deny",
                    "value": f"{approval_id}{_VALUE_SEP}{_DENY}",
                    "style": "danger",
                },
            ],
            "metadata": {
                "kind": "approval_request",
                "approval_id": approval_id,
                "name": tool_name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "args": tool_args,
                "tool_call_id": tool_call_id or approval_id,
                "requested_by": sender_id,
            },
            "_session_key": session_key,
        }
        if thread_id is not None:
            body["thread_id"] = thread_id
        return body

    def _render_resolution(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        thread_id: str | None,
        tool_name: str,
        decision: str,
        by_user: str,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        icon = "✅" if decision == _APPROVE else "🛑"
        verb = "approved" if decision in (_APPROVE, _APPROVE_TOOL, _APPROVE_SESSION) else "denied"
        body: dict[str, Any] = {
            "channel": channel,
            "chat_id": chat_id,
            "content": f"{icon} `{tool_name}` {verb} by {by_user}",
            "metadata": {
                "kind": "approval_resolved",
                "decision": decision,
                **({"approval_id": approval_id} if approval_id is not None else {}),
            },
            "_session_key": session_key,
        }
        if thread_id is not None:
            body["thread_id"] = thread_id
        return body


def _permission_pattern(tool_name: str, tool_args: dict[str, Any]) -> str:
    if tool_name in {"bash", "shell"}:
        cmd = str(tool_args.get("cmd") or tool_args.get("command") or "")
        fields = cmd.split()
        if fields:
            return f"{tool_name}:cmd={fields[0]}*"
    return tool_name


def _permission_rule_matches(
    rule: str, tool_name: str, tool_args: dict[str, Any]
) -> bool:
    rule = rule.strip()
    if not rule:
        return False
    if ":cmd=" in rule:
        rule_tool, _, command_pattern = rule.partition(":cmd=")
        return _tool_name_matches(rule_tool, tool_name) and _command_matches(
            command_pattern,
            tool_args,
        )

    parsed_tool, parsed_command_pattern = _parse_claude_permission_rule(rule)
    if parsed_command_pattern is None:
        return _tool_name_matches(parsed_tool, tool_name)
    return _tool_name_matches(parsed_tool, tool_name) and _command_matches(
        parsed_command_pattern,
        tool_args,
    )


def _parse_claude_permission_rule(rule: str) -> tuple[str, str | None]:
    if "(" not in rule or not rule.endswith(")"):
        return rule, None
    tool, _, rest = rule.partition("(")
    return tool, rest[:-1]


def _tool_name_matches(rule_tool: str, tool_name: str) -> bool:
    rule_tool = rule_tool.strip().lower()
    tool_name = tool_name.strip().lower()
    if rule_tool in {"bash", "shell"}:
        return tool_name in {"bash", "shell"}
    return rule_tool == tool_name


def _command_matches(pattern: str, tool_args: dict[str, Any]) -> bool:
    pattern = pattern.strip()
    command = str(tool_args.get("cmd") or tool_args.get("command") or "").strip()
    if not pattern or not command:
        return False
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatchcase(command, pattern)
    return command == pattern


__all__ = ["APPROVAL_TIMEOUT_S", "ApprovalManager", "OutboundSink"]
