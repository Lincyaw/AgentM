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
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger("agentm.gateway.approval")

_APPROVE = "approve"
_DENY = "deny"
_VALUE_SEP = ":"  # "<approval_id>:<approve|deny>" — internal encoding.

APPROVAL_TIMEOUT_S: float = 300.0

# The sink takes a single ``outbound`` body dict (§2.5) and ships it.
OutboundSink = Callable[[dict[str, Any]], Awaitable[None]]


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
        # approval_id -> (future, requester_sender_id)
        self._pending: dict[str, tuple[asyncio.Future[bool], str]] = {}

    # -- policy -------------------------------------------------------

    def requires(self, tool_name: str) -> bool:
        """True if a button press is needed before ``tool_name`` runs.

        ``always_block`` tools also "require" approval — but
        :meth:`request` short-circuits them to a denial without a card.
        ``"*"`` in ``require_approval`` means every tool not allowed
        elsewhere needs approval.
        """
        if tool_name in self._block:
            return True
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
    ) -> bool:
        """Block until the user approves or denies ``tool_name``.

        Returns ``True`` to proceed, ``False`` to deny (explicit deny,
        timeout, or always-block policy).
        """
        if tool_name in self._block:
            return False
        approval_id = f"appr-{uuid.uuid4().hex[:12]}"
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = (future, sender_id)
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
            )
        )
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
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
        if decision not in (_APPROVE, _DENY):
            return False
        entry = self._pending.get(approval_id)
        if entry is None:
            return False  # stale, already resolved or timed out
        future, requester_sender_id = entry
        if clicker_sender_id != requester_sender_id:
            log.info(
                "approval %s click by %s ignored (expected %s)",
                approval_id,
                clicker_sender_id,
                requester_sender_id,
            )
            return False  # identity mismatch, silently drop
        self._pending.pop(approval_id, None)
        if not future.done():
            future.set_result(decision == _APPROVE)
        return True

    @property
    def pending_count(self) -> int:
        return len(self._pending)

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
                "tool_name": tool_name,
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
    ) -> dict[str, Any]:
        icon = "✅" if decision == _APPROVE else "🛑"
        verb = "approved" if decision == _APPROVE else "denied"
        body: dict[str, Any] = {
            "channel": channel,
            "chat_id": chat_id,
            "content": f"{icon} `{tool_name}` {verb} by {by_user}",
            "metadata": {"kind": "approval_resolved", "decision": decision},
            "_session_key": session_key,
        }
        if thread_id is not None:
            body["thread_id"] = thread_id
        return body


__all__ = ["APPROVAL_TIMEOUT_S", "ApprovalManager", "OutboundSink"]
