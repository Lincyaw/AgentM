"""``/new`` — discard the current AgentSession for this chat.

Next inbound mints a fresh session. Useful when the conversation has
gone off the rails or the user wants to start over without losing the
chat itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...bus import OutboundMessage
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
    summary: str = "Start a fresh session for this chat"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        await ctx.drop_route()
        return CommandResult(
            outbound=[
                OutboundMessage(
                    channel=ctx.channel,
                    chat_id=ctx.chat_id,
                    content=(
                        "🌱 Session reset. The next message starts a fresh "
                        "conversation."
                    ),
                )
            ]
        )


HANDLER = NewCommand()
