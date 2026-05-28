"""Builtin ``wire_driver`` atom — AgentSession events -> wire envelopes (§4).

Installed by the single-process gateway's :class:`SessionManager` onto
every chat session so that the session's events fan out as ``outbound``
wire-envelope *bodies* to the originating chat client.

This is the only substantial new code the channels-v2 rewrite adds. It
replaces what would have been an entire worker package: pure translation
glue, fully expressible inside the §11 single-file atom contract — no
``core.runtime.*`` import, no atom-to-atom import, communicating with the
gateway only through services it gets by name.

Service contract (set by SessionManager before install):

* ``wire_outbound``    -> ``async (body_dict) -> None`` outbound sink.
* ``session_key``      -> ``str`` (echoed back so the gateway routes the
  outbound to the right chat client).
* ``turn_context``     -> mutable dict with ``channel`` / ``chat_id`` /
  ``thread_id`` / ``sender_id`` for the current turn (used to address
  outbound bodies and approval cards).
* ``approval_manager`` -> optional object exposing ``requires(name)`` and
  ``request(...)`` (the gateway's ApprovalManager). When present and a
  tool needs approval, the atom blocks the tool until the user decides.

The atom builds plain ``dict`` bodies (not the gateway's ``Envelope`` /
``OutboundBody`` types) precisely so it never imports a gateway module —
the gateway wraps the dict into an envelope on the way out.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import AssistantMessage, TextContent
from agentm.core.abi.events import (
    DiagnosticEvent,
    ToolCallEvent,
    TurnEndEvent,
)
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="wire_driver",
    description=(
        "Translate AgentSession events into wire outbound envelope bodies "
        "for the single-process gateway. Installed per chat session by the "
        "gateway's SessionManager; reads wire_outbound / session_key / "
        "turn_context / approval_manager services. Emits assistant text on "
        "turn_end and diagnostics on warning/error; gates write-class tools "
        "through the approval_manager when policy requires it."
    ),
    registers=(
        "event:turn_end",
        "event:tool_call",
        "event:diagnostic",
    ),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
)


def _assistant_text(message: AssistantMessage) -> str:
    return "\n".join(
        block.text
        for block in message.content
        if isinstance(block, TextContent) and block.text
    )


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:  # noqa: ARG001
    outbound_sink = api.get_service("wire_outbound")
    session_key = api.get_service("session_key")
    if outbound_sink is None or session_key is None:
        # Mounting wire_driver outside the gateway has no effect; fail at
        # install so the misconfiguration surfaces immediately rather than
        # silently swallowing every session event.
        raise RuntimeError(
            "wire_driver requires 'wire_outbound' and 'session_key' services; "
            "this atom only works inside the agentm gateway process."
        )
    turn_context: dict[str, Any] | None = api.get_service("turn_context")
    approval_mgr = api.get_service("approval_manager")

    def _addr() -> dict[str, Any]:
        ctx = turn_context or {}
        return {
            "channel": str(ctx.get("channel") or ""),
            "chat_id": str(ctx.get("chat_id") or ""),
            "thread_id": ctx.get("thread_id"),
        }

    async def emit(body_kind: str, content: str) -> None:
        addr = _addr()
        body: dict[str, Any] = {
            "channel": addr["channel"],
            "chat_id": addr["chat_id"],
            "content": content,
            "metadata": {"kind": body_kind},
        }
        if addr["thread_id"] is not None:
            body["thread_id"] = addr["thread_id"]
        await outbound_sink(body)

    async def on_turn_end(ev: TurnEndEvent) -> None:
        text = _assistant_text(ev.message)
        if text.strip():
            await emit("assistant_text", text)

    async def on_tool_call(ev: ToolCallEvent) -> dict[str, Any] | None:
        if approval_mgr is None or not approval_mgr.requires(ev.tool_name):
            return None
        ctx = turn_context or {}
        ok = await approval_mgr.request(
            session_key=session_key,
            sender_id=str(ctx.get("sender_id") or ""),
            channel=str(ctx.get("channel") or ""),
            chat_id=str(ctx.get("chat_id") or ""),
            thread_id=ctx.get("thread_id"),
            tool_name=ev.tool_name,
            tool_args=dict(ev.args),
        )
        if not ok:
            return {
                "block": True,
                "reason": f"tool '{ev.tool_name}' was denied via the chat approval gate",
            }
        return None

    async def on_diagnostic(ev: DiagnosticEvent) -> None:
        severity = getattr(ev, "level", "info")
        message = getattr(ev, "message", "")
        if not message:
            return
        if severity == "warning":
            await emit("diagnostic_warning", message)
        elif severity == "error":
            await emit("diagnostic_error", message)

    api.on(TurnEndEvent.CHANNEL, on_turn_end)
    api.on(ToolCallEvent.CHANNEL, on_tool_call)
    api.on(DiagnosticEvent.CHANNEL, on_diagnostic)
