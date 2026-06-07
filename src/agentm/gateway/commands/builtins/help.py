"""``/help`` — list every discovered command, grouped by source."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from ..protocol import (
    CommandKind,
    CommandContext,
    CommandHandler,
    CommandInvocation,
    CommandResult,
)


@dataclass(slots=True)
class HelpCommand:
    name: str = "help"
    namespace: str | None = None
    summary: str = "List available commands"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        del inv
        body = _format_help(ctx.list_commands(), ctx.list_session_commands())
        return CommandResult(outbound=[ctx.reply(body)])


HANDLER = HelpCommand()


def _format_help(
    handlers: list[CommandHandler], session_commands: list[str]
) -> str:
    """Group by namespace; render compactly enough that Feishu's card
    body doesn't overflow on a 20-command list.

    ``session_commands`` are bare names registered inside the session (e.g.
    ``compact``), dispatched by the in-session ``slash_commands`` atom — they
    never appear among ``handlers`` (the gateway registry), so they get their
    own "session" group. Names already owned by a gateway handler are dropped
    to avoid double-listing."""
    grouped: OrderedDict[str, list[CommandHandler]] = OrderedDict()
    grouped["control"] = []
    grouped["prompt"] = []
    gateway_names: set[str] = set()
    for h in handlers:
        if h.namespace is None:
            gateway_names.add(h.name)
            bucket = "control" if h.kind == "control" else "prompt"
            grouped.setdefault(bucket, []).append(h)
        else:
            grouped.setdefault(h.namespace, []).append(h)

    session_only = [n for n in session_commands if n not in gateway_names]

    lines: list[str] = ["**Available commands**"]
    total = sum(len(v) for v in grouped.values()) + len(session_only)
    lines.append(f"_({total} total)_")
    lines.append("")
    for source, items in grouped.items():
        if not items:
            continue
        lines.append(f"**{source}** ({len(items)})")
        for h in items:
            prefix = (
                f"/{h.name}"
                if h.namespace is None
                else f"/{h.namespace}:{h.name}"
            )
            lines.append(f"  `{prefix}` — {h.summary}")
        lines.append("")
    if session_only:
        lines.append(f"**session** ({len(session_only)})")
        for name in session_only:
            lines.append(f"  `/{name}`")
        lines.append("")
    return "\n".join(lines).rstrip()
