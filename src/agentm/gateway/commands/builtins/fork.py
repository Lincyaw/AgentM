"""``/fork`` — branch the current session into a new one.

Forks the current chat's session: a new session is created seeded with the
current transcript (optionally truncated to the first ``N`` messages), and the
chat switches to it. The original session is untouched and remains resumable via
``/resume <id>``. The fork takes effect on the next message, mirroring
``/resume``'s deferred-switch model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol import (
    CommandContext,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


@dataclass(slots=True)
class ForkCommand:
    name: str = "fork"
    namespace: str | None = None
    summary: str = "Branch this session into a new one (usage: /fork [up_to])"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        up_to: int | None = None
        arg = inv.args.strip()
        if arg:
            try:
                up_to = int(arg)
            except ValueError:
                return CommandResult(
                    outbound=[
                        ctx.reply(
                            f"Invalid message count: `{arg}`. "
                            "Usage: `/fork [up_to]`",
                            kind="diagnostic_error",
                        )
                    ]
                )
            if up_to < 0:
                return CommandResult(
                    outbound=[
                        ctx.reply(
                            "Message count must be non-negative.",
                            kind="diagnostic_error",
                        )
                    ]
                )

        source = await ctx.fork_session(up_to)
        if source is None:
            return CommandResult(
                outbound=[
                    ctx.notice("No active session to fork. Send a message first.")
                ]
            )
        scope = f" (first {up_to} messages)" if up_to is not None else ""
        return CommandResult(
            outbound=[
                ctx.notice(
                    f"\U0001f33f Forked from `{source[:12]}…`{scope}. "
                    "Your next message continues on the new branch; "
                    f"`/resume {source}` returns to the original."
                )
            ]
        )


HANDLER = ForkCommand()
