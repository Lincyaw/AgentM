"""``/scenario`` — show or switch the active scenario.

``/scenario`` lists discoverable scenarios. ``/scenario <name>`` switches this
chat to that scenario and starts a fresh chat session using it.
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
class ScenarioCommand:
    name: str = "scenario"
    namespace: str | None = None
    summary: str = "Show or switch the active scenario"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        current, available = ctx.list_scenarios()
        arg = inv.args.strip()

        if not arg:
            return CommandResult(outbound=[ctx.notice(_render_list(current, available))])

        ok, message = await ctx.switch_scenario(arg)
        if ok:
            return CommandResult(
                outbound=[
                    ctx.notice(
                        f"Switched to `{message}`. Fresh session started."
                    )
                ]
            )
        hint = f"\n{_render_list(current, available)}" if available else ""
        return CommandResult(
            outbound=[
                ctx.reply(
                    f"Could not switch to `{arg}`: {message}.{hint}",
                    kind="diagnostic_error",
                )
            ]
        )


def _render_list(current: str, available: list[dict[str, str]]) -> str:
    active = current or "default"
    if not available:
        return f"**Scenario**: `{active}`\nNo discoverable scenarios."
    lines = []
    for item in available:
        name = item.get("name", "")
        if not name:
            continue
        suffix = "  <- active" if name == current else ""
        description = _compact_text(item.get("description", ""), 96)
        if description:
            lines.append(f"- `{name}`{suffix} - {description}")
        else:
            lines.append(f"- `{name}`{suffix}")
    return (
        f"**Scenarios** (active: `{active}`)\n"
        + "\n".join(lines)
        + "\n\nSwitch with `/scenario <name>`."
    )


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


HANDLER = ScenarioCommand()
