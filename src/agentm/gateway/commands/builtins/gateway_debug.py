"""``/gateway_debug`` — introspect gateway runtime state for a session.

This is intentionally a control command that does not affect conversation
history; it exposes non-semantic, operational state that helps diagnose why a
turn is stuck, why approvals are not progressing, or whether routing/session
projection is still healthy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..protocol import (
    CommandContext,
    CommandInvocation,
    CommandKind,
    CommandResult,
)


@dataclass(slots=True)
class GatewayDebugCommand:
    name: str = "gateway_debug"
    namespace: str | None = None
    summary: str = "Show gateway runtime debug diagnostics"
    kind: CommandKind = "control"

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult:
        state = ctx.get_gateway_debug_state()

        args = inv.args.strip().lower()
        mode = "session"
        if args in {"all", "sessions"}:
            mode = "all"

        if mode == "all":
            body = _format_all(state)
        else:
            body = _format_session(state)

        return CommandResult(outbound=[ctx.notice(body)])


def _format_session(state: dict) -> str:
    session = state.get("session", {})
    globals_ = state.get("global", {})
    payload = {
        "session_key": state.get("session_key"),
        "session": session,
        "global": globals_,
    }
    return (
        "**Gateway Debug**\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )


def _format_all(state: dict) -> str:
    lines: list[str] = ["**Gateway Debug**", "", "**sessions**"]
    sessions = state.get("sessions") or {}
    for key in sorted(sessions.keys()):
        lines.append(f"- `{key}`")
        snap = sessions[key]
        sid = snap.get("session_id")
        phase = (snap.get("snapshot") or {}).get("phase")
        route = snap.get("route") or {}
        if sid:
            lines.append(f"  - session_id: `{sid}`")
        if phase:
            lines.append(f"  - phase: {phase}")
        if route:
            lines.append(
                "  - route: "
                f"{route.get('channel')}/{route.get('chat_id')}"
                + (f"#{route.get('thread_id')}" if route.get("thread_id") else "")
            )
        pending = (snap.get("snapshot") or {}).get("pending_interactions", [])
        lines.append(f"  - pending_interactions: {len(pending)}")
        if pending:
            lines.append(f"    - {', '.join(pending)}")
    if not sessions:
        lines.append("- no tracked sessions")

    lines.append("")
    lines.append("**globals**")
    globals_ = state.get("global", {})
    for key, value in sorted(globals_.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


HANDLER = GatewayDebugCommand()
