"""Approval bridge: ToolCall gate ↔ interactive cards.

Hooks ``ToolCallEvent`` and, for tools the operator has flagged as
write-class, posts an interactive card to the originating chat and
awaits the requester's button click. The handler is an async coroutine;
the kernel's :class:`agentm.core.abi.EventBus` awaits awaitable handler
return values (see ``EventBus.emit``), so the agent's loop pauses at
the gate without polling.

Decision flow
=============

1. ``tool_call`` fires. Bridge consults the policy:

   - tool name in ``always_allow`` → return ``None`` (no gate).
   - tool name in ``always_block`` → return block dict immediately.
   - otherwise → post approval card, register a Future, await it.

2. The user clicks **Approve** or **Deny** in Feishu. The gateway's
   ``card_actions`` consumer routes the action back into
   :meth:`ApprovalBridge.resolve`, which sets the Future.

3. The bridge updates the card to a resolved-state body so it stops
   showing live buttons, then returns ``None`` (allow) or
   ``{"block": True, "reason": ...}`` (deny / timeout).

Identity check
==============

Only the user who triggered the tool call may approve it — anyone else
clicking the card is ignored. The requester is derived from the
inbound message that started the turn; the gateway threads it into
:attr:`ApprovalContext`. This is *defense-in-depth*; group-chat
permission errors should still be caught by Feishu's own ACLs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import ToolCallEvent

from .cards import approval_card, resolved_card
from .chat_source import CardActionEvent, ChatSource


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ApprovalContext:
    """Per-turn routing context, set by the gateway before each prompt."""

    chat_id: str
    user_id: str
    thread_id: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """Which tool calls require human approval.

    ``always_allow`` and ``always_block`` are checked first; tools not
    matched fall to ``require_approval`` (and tools matching nothing
    pass through ungated). Wildcard ``"*"`` in ``require_approval``
    means *every tool not explicitly allowed needs approval*.
    """

    always_allow: frozenset[str] = frozenset()
    always_block: frozenset[str] = frozenset()
    require_approval: frozenset[str] = frozenset()
    timeout_seconds: float = 300.0


# Internal future-registry payload — the click metadata the gateway
# routes back from a CardActionEvent.
_PendingResult = tuple[str, str]  # (decision, by_user)


class ApprovalBridge:
    """Owns the gate handler and the pending-future registry."""

    def __init__(
        self,
        source: ChatSource,
        policy: ApprovalPolicy,
        *,
        get_context: Callable[[], ApprovalContext | None],
    ) -> None:
        self._source = source
        self._policy = policy
        self._get_context = get_context
        self._pending: dict[str, asyncio.Future[_PendingResult]] = {}
        self._lock = asyncio.Lock()
        # Card_id → tool_name, used when rendering the resolved-state card.
        self._card_tool: dict[str, str] = {}

    def _decision(self, tool_name: str) -> str:
        if tool_name in self._policy.always_block:
            return "deny"
        if tool_name in self._policy.always_allow:
            return "allow"
        if tool_name in self._policy.require_approval or "*" in self._policy.require_approval:
            return "ask"
        return "allow"

    async def handle_tool_call(self, event: ToolCallEvent) -> dict[str, Any] | None:
        """``tool_call`` handler. Returns block dict or ``None``."""
        decision = self._decision(event.tool_name)
        if decision == "allow":
            return None
        if decision == "deny":
            return {
                "block": True,
                "reason": f"tool '{event.tool_name}' is denied by gateway policy",
            }

        ctx = self._get_context()
        if ctx is None:
            # No live chat context — refuse rather than send a card to no-one.
            return {
                "block": True,
                "reason": (
                    f"tool '{event.tool_name}' requires approval, but no "
                    "originating chat is in scope"
                ),
            }

        summary = self._summarize(event)
        future: asyncio.Future[_PendingResult] = asyncio.get_running_loop().create_future()
        # We need the card_id to wire the future, but send_card is what gives
        # it to us. Allocate a temporary key, send the card with placeholder
        # content, then re-key once we know the real id. To avoid that
        # round-trip the cards module accepts a card_id we choose; here we
        # mint one from the future's identity.
        card_id = f"approval-{id(future):x}"
        self._card_tool[card_id] = event.tool_name
        async with self._lock:
            self._pending[card_id] = future

        body = approval_card(
            card_id=card_id,
            tool_name=event.tool_name,
            summary=summary,
            requested_by=ctx.user_id,
        )
        try:
            await self._source.send_card(ctx.chat_id, body, thread_id=ctx.thread_id)
        except Exception as exc:
            logger.exception("approval card send failed: %s", exc)
            self._pending.pop(card_id, None)
            return {
                "block": True,
                "reason": f"failed to deliver approval card: {exc}",
            }

        try:
            decision_text, by_user = await asyncio.wait_for(
                future, timeout=self._policy.timeout_seconds
            )
        except asyncio.TimeoutError:
            self._pending.pop(card_id, None)
            await self._safe_update(
                card_id,
                resolved_card(
                    tool_name=event.tool_name, decision="deny", by_user="timeout"
                ),
            )
            return {
                "block": True,
                "reason": (
                    f"tool '{event.tool_name}' approval timed out after "
                    f"{self._policy.timeout_seconds:.0f}s"
                ),
            }
        finally:
            self._card_tool.pop(card_id, None)

        await self._safe_update(
            card_id,
            resolved_card(
                tool_name=event.tool_name, decision=decision_text, by_user=by_user
            ),
        )

        if decision_text == "approve":
            return None
        return {
            "block": True,
            "reason": f"tool '{event.tool_name}' was denied by {by_user}",
        }

    async def resolve(self, action: CardActionEvent) -> bool:
        """Route a ``cardAction`` event back into the pending future.

        Returns True if the action matched a pending approval, else False
        (the gateway can use this to know whether to forward to other
        handlers).
        """
        async with self._lock:
            future = self._pending.pop(action.card_id, None)
        if future is None or future.done():
            return False

        ctx = self._get_context()
        # Ignore clicks from someone other than the original requester —
        # but only when we have a live context. If the bridge was created
        # without identity scoping, accept any click (useful in tests).
        if ctx is not None and action.user_id != ctx.user_id:
            logger.info(
                "ignored card action from %s (expected %s)",
                action.user_id,
                ctx.user_id,
            )
            # Re-register so the rightful user can still click later.
            async with self._lock:
                self._pending[action.card_id] = future
            return False

        decision = action.action if action.action in {"approve", "deny"} else "deny"
        future.set_result((decision, action.user_id))
        return True

    async def _safe_update(self, card_id: str, body: dict[str, Any]) -> None:
        try:
            await self._source.update_card(card_id, body)
        except Exception:
            logger.exception("update_card failed for %s", card_id)

    @staticmethod
    def _summarize(event: ToolCallEvent) -> str:
        try:
            args = json.dumps(event.args, indent=2, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args = repr(event.args)
        if len(args) > 1500:
            args = args[:1500] + "\n…"
        return f"**{event.tool_name}**\n```json\n{args}\n```"
