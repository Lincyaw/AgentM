"""``/new`` — start a fresh session for this chat (§3.5).

Shuts down the current in-memory ``AgentSession`` AND clears the
persistent ``ChatSessionMap`` entry so the next inbound message creates
a brand-new session with no prior history.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol import (
    CommandKind,
    CommandContext,
    CommandInvocation,
    CommandResult,
)


@dataclass(slots=True)
class NewCommand:
    name: str = "new"
    namespace: str | None = None
    summary: str = "Start a fresh session (clears history)"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        await ctx.end_session()
        await ctx.forget_chat_mapping()
        return CommandResult(
            outbound=[
                ctx.notice("🌱 New session started. History cleared.")
            ]
        )


HANDLER = NewCommand()
