"""``/end`` — shut down the session and forget the chat mapping.

Differs from ``/new``: ``/new`` keeps the chat alive (next message
starts a new session); ``/end`` *also* clears the persistent
``ChatSessionMap`` entry, so even after a gateway restart the chat
starts cold.
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
class EndCommand:
    name: str = "end"
    namespace: str | None = None
    summary: str = "Close this chat's session and forget it"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        # §3.5: shut down the in-memory session AND clear the persistent
        # ChatSessionMap entry, so the next message starts a fresh,
        # cold session rather than resuming the transcript.
        await ctx.end_session()
        await ctx.forget_chat_mapping()
        return CommandResult(
            outbound=[ctx.reply("👋 Session closed. This chat is now cold.")]
        )


HANDLER = EndCommand()
