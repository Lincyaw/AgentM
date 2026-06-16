"""``/resume <session_id>`` — switch this chat to a different session.

Shuts down the current in-memory session and points the persistent
:class:`ChatSessionMap` entry at ``session_id`` so the next inbound
message resumes from that session's transcript. A ``control`` command:
the typed text never reaches the LLM.
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
class ResumeCommand:
    name: str = "resume"
    namespace: str | None = None
    summary: str = "Resume a previous session by ID"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        target = inv.args.strip()

        if not target:
            stats = ctx.get_route_stats()
            sid = stats.get("session_id") or "(none)"
            return CommandResult(
                outbound=[
                    ctx.notice(
                        f"Current session: `{sid}`\nUsage: `/resume <session_id>`"
                    )
                ]
            )

        if not target.isalnum() or not (16 <= len(target) <= 64):
            return CommandResult(
                outbound=[
                    ctx.reply(
                        f"Invalid session ID: `{target}`",
                        kind="diagnostic_error",
                    )
                ]
            )

        from agentm.core.observability.otel_export import resolve_observability_dir

        trace_path = resolve_observability_dir(ctx.cwd) / f"{target}.jsonl"
        if not trace_path.exists():
            return CommandResult(
                outbound=[
                    ctx.reply(
                        f"Session not found: `{target}`",
                        kind="diagnostic_error",
                    )
                ]
            )

        await ctx.resume_session(target)
        return CommandResult(
            outbound=[
                ctx.notice(
                    f"\U0001f504 Resumed session `{target[:12]}…`. "
                    "Next message continues from that session's transcript."
                )
            ]
        )


HANDLER = ResumeCommand()
