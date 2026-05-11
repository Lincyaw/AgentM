"""Channel-agnostic approval bridge.

Hooks ``ToolCallEvent`` and, for tools the operator has flagged as
write-class, posts an :class:`OutboundMessage` with two
:class:`Button`s (``Approve`` / ``Deny``) and awaits the user's reply.
The handler is an async coroutine; the kernel's
:class:`agentm.core.abi.EventBus` awaits awaitable handlers, so the
agent's loop pauses at the gate without polling.

How replies come back:

- The channel converts each :class:`Button` into its native UI (Feishu
  interactive card, Slack action block, Telegram inline keyboard,
  plaintext "reply with ``approve``" as last resort) and round-trips
  ``button.value`` into :attr:`InboundMessage.button_value` when the
  user clicks.
- The gateway hands every inbound with a ``button_value`` to
  :meth:`ApprovalBridge.try_resolve_inbound`; the bridge owns the
  encoding (``"<approval_id>:approve|deny"``) and tells the gateway
  whether the click was claimed.

This means **no Feishu-specific code lives here** — every channel
participates the same way.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import ToolCallEvent

from .bus import Button, InboundMessage, MessageBus, OutboundMessage


logger = logging.getLogger(__name__)


_APPROVE = "approve"
_DENY = "deny"
_VALUE_SEP = ":"  # "<approval_id>:<approve|deny>" — internal to this module.


@dataclass(frozen=True, slots=True)
class ApprovalContext:
    """Per-turn routing context, set by the gateway before each prompt."""

    channel: str
    chat_id: str
    sender_id: str


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    always_allow: frozenset[str] = frozenset()
    always_block: frozenset[str] = frozenset()
    require_approval: frozenset[str] = frozenset()
    """Tools that require a button press. ``"*"`` means *every tool not
    in always_allow needs approval*."""
    timeout_seconds: float = 300.0


_PendingResult = tuple[str, str]  # (decision, by_user)


class ApprovalBridge:
    def __init__(
        self,
        bus: MessageBus,
        policy: ApprovalPolicy,
        *,
        get_context: Callable[[], ApprovalContext | None],
        index: "dict[str, ApprovalBridge] | None" = None,
    ) -> None:
        self._bus = bus
        self._policy = policy
        self._get_context = get_context
        self._pending: dict[str, asyncio.Future[_PendingResult]] = {}
        self._lock = asyncio.Lock()
        # Gateway-owned ``approval_id → bridge`` index for O(1) routing
        # of button clicks. Optional so the bridge stays usable in unit
        # tests that drive ``resolve()`` directly. When provided, the
        # bridge keeps it in sync: register on request publish, pop on
        # any terminal outcome (resolve / timeout). Misses fall back to
        # the broadcast path in the gateway.
        self._index = index

    def _decide(self, tool_name: str) -> str:
        if tool_name in self._policy.always_block:
            return "deny"
        if tool_name in self._policy.always_allow:
            return "allow"
        if (
            tool_name in self._policy.require_approval
            or "*" in self._policy.require_approval
        ):
            return "ask"
        return "allow"

    async def handle_tool_call(
        self, event: ToolCallEvent
    ) -> dict[str, Any] | None:
        decision = self._decide(event.tool_name)
        if decision == "allow":
            return None
        if decision == "deny":
            return {
                "block": True,
                "reason": f"tool '{event.tool_name}' is denied by gateway policy",
            }

        ctx = self._get_context()
        if ctx is None:
            return {
                "block": True,
                "reason": (
                    f"tool '{event.tool_name}' requires approval, but no "
                    "originating chat is in scope"
                ),
            }

        future: asyncio.Future[_PendingResult] = (
            asyncio.get_running_loop().create_future()
        )
        approval_id = f"approval-{id(future):x}"
        async with self._lock:
            self._pending[approval_id] = future
        if self._index is not None:
            self._index[approval_id] = self

        body = self._format_request(event, requested_by=ctx.sender_id)
        # We encode the approval_id into each button's typed ``value``
        # so the click round-trips cleanly even when several approval
        # cards are open in the same chat. The encoding is private to
        # this module; the gateway uses :meth:`try_resolve_inbound`
        # rather than parsing it itself.
        msg = OutboundMessage(
            channel=ctx.channel,
            chat_id=ctx.chat_id,
            content=body,
            buttons=[
                Button(
                    label="Approve",
                    value=f"{approval_id}{_VALUE_SEP}{_APPROVE}",
                    style="primary",
                ),
                Button(
                    label="Deny",
                    value=f"{approval_id}{_VALUE_SEP}{_DENY}",
                    style="danger",
                ),
            ],
            metadata={
                "kind": "approval_request",
                "approval_id": approval_id,
                "tool_name": event.tool_name,
                "requested_by": ctx.sender_id,
            },
        )
        await self._bus.publish_outbound(msg)

        try:
            decision_text, by_user = await asyncio.wait_for(
                future, timeout=self._policy.timeout_seconds
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending.pop(approval_id, None)
            if self._index is not None:
                self._index.pop(approval_id, None)
            await self._post_resolution(
                ctx, event.tool_name, _DENY, "timeout"
            )
            return {
                "block": True,
                "reason": (
                    f"tool '{event.tool_name}' approval timed out after "
                    f"{self._policy.timeout_seconds:.0f}s"
                ),
            }

        await self._post_resolution(ctx, event.tool_name, decision_text, by_user)
        if decision_text == _APPROVE:
            return None
        return {
            "block": True,
            "reason": f"tool '{event.tool_name}' was denied by {by_user}",
        }

    async def try_resolve_inbound(self, msg: InboundMessage) -> bool:
        """Claim ``msg`` if it is the round-trip of one of our buttons.

        Returns True when the inbound carried a valid approval click and
        was consumed — the caller should drop it rather than forwarding
        to the agent. Returns False otherwise (no ``button_value``,
        unrecognized prefix, malformed decision, or a stale id whose
        future already expired).

        The encoding (``"<approval_id>:<decision>"``) is intentionally
        opaque outside this module — gateways must not parse
        ``button_value`` themselves.
        """
        value = msg.button_value
        if not value or _VALUE_SEP not in value:
            return False
        approval_id, _, decision = value.partition(_VALUE_SEP)
        if not approval_id or decision not in {_APPROVE, _DENY}:
            return False
        return await self._resolve(
            approval_id, decision=decision, sender_id=msg.sender_id
        )

    async def resolve(self, approval_id: str, *, value: str, sender_id: str) -> bool:
        """Direct resolve path for tests and embedders that already
        decoded the value. Production code paths through
        :meth:`try_resolve_inbound`.
        """
        if not value.startswith(approval_id + _VALUE_SEP):
            return False
        decision = value.split(_VALUE_SEP, 1)[1]
        if decision not in {_APPROVE, _DENY}:
            return False
        return await self._resolve(
            approval_id, decision=decision, sender_id=sender_id
        )

    async def _resolve(
        self, approval_id: str, *, decision: str, sender_id: str
    ) -> bool:
        async with self._lock:
            future = self._pending.pop(approval_id, None)
        if future is None or future.done():
            return False

        ctx = self._get_context()
        if ctx is not None and sender_id != ctx.sender_id:
            # Re-register so the rightful user can still act.
            async with self._lock:
                self._pending[approval_id] = future
            logger.info(
                "ignored approval click on %s by %s (expected %s)",
                approval_id,
                sender_id,
                ctx.sender_id,
            )
            return False

        if self._index is not None:
            self._index.pop(approval_id, None)
        future.set_result((decision, sender_id))
        return True

    @staticmethod
    def _format_request(event: ToolCallEvent, *, requested_by: str) -> str:
        try:
            args = json.dumps(event.args, indent=2, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args = repr(event.args)
        if len(args) > 1500:
            args = args[:1500] + "\n…"
        return (
            f"Approve `{event.tool_name}`? (requested by {requested_by})\n"
            f"```json\n{args}\n```"
        )

    async def _post_resolution(
        self, ctx: ApprovalContext, tool_name: str, decision: str, by_user: str
    ) -> None:
        icon = "✅" if decision == _APPROVE else "🛑"
        verb = "approved" if decision == _APPROVE else "denied"
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=ctx.channel,
                chat_id=ctx.chat_id,
                content=f"{icon} `{tool_name}` {verb} by {by_user}",
                metadata={"kind": "approval_resolved", "decision": decision},
            )
        )
