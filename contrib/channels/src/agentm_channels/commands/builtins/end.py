"""``/end`` — shut down the session and forget the chat mapping.

Differs from ``/new``: ``/new`` keeps the chat alive (next message
starts a new session); ``/end`` *also* clears the persistent
``ChatSessionMap`` entry, so even after a gateway restart the chat
starts cold.
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
class EndCommand:
    name: str = "end"
    namespace: str | None = None
    summary: str = "Close this chat's session and forget it"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        # The route's ``drop_route`` callback is the gateway-side
        # cleanup; ``forget_chat_mapping`` is exposed through stats
        # (not a separate facade method) to keep CommandContext
        # surface area small. The gateway implementation flips the
        # ``forget`` flag based on the command name; cleaner than
        # giving the handler raw access to ChatSessionMap.
        stats = ctx.get_route_stats()
        # Implementation note: the gateway honours
        # ``stats["_supports_forget"]`` (always True for the current
        # gateway). Listed for future-proofing — if a different
        # gateway impl ever lacks this, we degrade to /new behavior.
        await ctx.drop_route()
        if stats.get("_supports_forget"):
            forget = stats.get("_forget_chat_mapping")
            if callable(forget):
                await forget()
        return CommandResult(
            outbound=[
                OutboundMessage(
                    channel=ctx.channel,
                    chat_id=ctx.chat_id,
                    content="👋 Session closed. This chat is now cold.",
                )
            ]
        )


HANDLER = EndCommand()
