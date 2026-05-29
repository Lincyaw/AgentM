"""``/model`` — show or switch the active model.

``/model`` lists the named profiles from ``config.toml`` and marks the active
one. ``/model <name>`` switches to that profile and restarts this chat's
session, keeping the transcript (the next message resumes on the new model) —
the same teardown ``/new`` uses. Contrast ``/end``, which clears context.

A ``control`` command: the typed text never reaches the LLM.
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
class ModelCommand:
    name: str = "model"
    namespace: str | None = None
    summary: str = "Show or switch the active model (keeps transcript)"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        current, available = ctx.list_models()
        arg = inv.args.strip()

        if not arg:
            return CommandResult(outbound=[ctx.reply(_render_list(current, available))])

        ok, message = await ctx.switch_model(arg)
        if ok:
            return CommandResult(
                outbound=[
                    ctx.reply(
                        f"🔀 Switched to `{message}`. The next message uses the "
                        "new model (transcript kept)."
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


def _render_list(current: str, available: list[str]) -> str:
    active = current or "default"
    if not available:
        return (
            f"**Model**: `{active}`\n"
            "No named profiles in `config.toml` ([models.<name>] tables)."
        )
    lines = "\n".join(
        f"- `{m}`" + ("  ← active" if m == current else "") for m in available
    )
    return (
        f"**Models** (active: `{active}`)\n{lines}\n\n"
        "Switch with `/model <name>` — keeps the transcript."
    )


HANDLER = ModelCommand()
