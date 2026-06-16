"""``/status`` — report session id and pending approvals for this chat."""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol import (
    CommandKind,
    CommandContext,
    CommandInvocation,
    CommandResult,
)


@dataclass(slots=True)
class StatusCommand:
    name: str = "status"
    namespace: str | None = None
    summary: str = "Show session id, turn count, pending approvals"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        stats = ctx.get_route_stats()
        session_id = stats.get("session_id") or "(not started)"
        turn_count = stats.get("turn_count", 0)
        pending = stats.get("pending_approvals", 0)
        body = (
            f"**Status**\n"
            f"- chat: `{ctx.channel}:{ctx.chat_id}`\n"
            f"- session: `{session_id}`\n"
            f"- turns this session: {turn_count}\n"
            f"- pending approvals: {pending}"
        )
        return CommandResult(outbound=[ctx.notice(body)])


HANDLER = StatusCommand()
