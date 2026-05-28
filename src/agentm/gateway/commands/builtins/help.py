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
        body = _format_help(ctx.list_commands())
        return CommandResult(outbound=[ctx.reply(body)])


HANDLER = HelpCommand()


def _format_help(handlers: list[CommandHandler]) -> str:
    """Group by namespace; render compactly enough that Feishu's card
    body doesn't overflow on a 20-command list."""
    grouped: OrderedDict[str, list[CommandHandler]] = OrderedDict()
    grouped["control"] = []
    grouped["prompt"] = []
    for h in handlers:
        if h.namespace is None:
            bucket = "control" if h.kind == "control" else "prompt"
            grouped.setdefault(bucket, []).append(h)
        else:
            grouped.setdefault(h.namespace, []).append(h)

    lines: list[str] = ["**Available commands**"]
    total = sum(len(v) for v in grouped.values())
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
    return "\n".join(lines).rstrip()
