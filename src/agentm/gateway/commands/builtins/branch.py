"""``/branch`` — Claude-style named conversation branch command."""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol import (
    CommandContext,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


@dataclass(slots=True)
class BranchCommand:
    name: str = "branch"
    namespace: str | None = None
    summary: str = "Create a branch of the current conversation at this point"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        branch_name = inv.args.strip() or None

        source = await ctx.fork_session(None)
        if source is None:
            return CommandResult(
                outbound=[
                    ctx.notice(
                        "Failed to branch conversation: No conversation to branch",
                        display="notice",
                    )
                ]
            )

        if branch_name:
            summary = f'Branched conversation "{branch_name}". '
        else:
            summary = "Branched conversation. "
        return CommandResult(
            outbound=[
                ctx.notice(
                    summary +
                    "Your next message continues on the new branch. "
                    f"Use /resume {source} to return to the original.",
                    display="notice",
                )
            ]
        )


HANDLER = BranchCommand()
