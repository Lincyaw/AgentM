"""``/new`` — shut down the current AgentSession for this chat (§3.5).

Keeps the persistent :class:`ChatSessionMap` entry intact, so the next
inbound re-resumes the same transcript from a fresh in-memory session.
Contrast ``/end``, which also clears the map for a cold start.
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
    summary: str = "Restart this chat's session (keeps transcript)"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        # §3.5: shut down the in-memory session; leave ChatSessionMap so
        # the next message resumes from transcript.
        await ctx.end_session()
        return CommandResult(
            outbound=[
                ctx.reply(
                    "🌱 Session restarted. The next message resumes this "
                    "chat's transcript in a fresh session."
                )
            ]
        )


HANDLER = NewCommand()
