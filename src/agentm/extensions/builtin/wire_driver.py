"""Builtin ``wire_driver`` atom — AgentSession events -> wire envelopes (§4).

Installed by the single-process gateway's :class:`SessionManager` onto every
chat session so the session's events fan out as ``outbound`` wire-envelope
*bodies* to the originating chat client. It is the bus→wire bridge: the TUI is
a separate process and cannot subscribe to the session bus, so whatever it
renders must first be projected onto the wire here (see
``.claude/designs/textual-tui.md``).

Design — a **declarative projector table**. ``_PROJECTORS`` maps a bus channel
to a pure function that turns the event into a JSON-safe outbound body (or
``None`` to skip a particular event). The *set of surfaced events and their
wire shape lives in one reviewable place*; the table omitting a channel is the
explicit allow-list (kernel-internal / persistence / veto / rewrite hooks are
deliberately absent). This mirrors the per-event ``to_otel`` projection pattern
in ``core/abi/events.py``.

Async vs sync dispatch. Conversation channels (``stream_delta``, ``turn_*``,
``tool_*``, ``child_session_*``, ``diagnostic``, ``agent_end``) are dispatched
via async ``bus.emit``; their handlers are coroutines that ``await`` the sink,
preserving order and backpressure on the streaming hot path. The runtime
control channels (``extension_*``, ``api_register``, ``api_send_user_message``,
...) are dispatched via ``bus.emit_sync`` — an **async handler there is
silently skipped**, which would drop exactly the self-modification events the
TUI exists to surface — so those use a *sync* handler that schedules the async
sink via ``loop.create_task``. A sync handler is valid under both emit paths.

Delivery class is the gateway sink's concern (``_DURABLE_OUTBOUND_KINDS`` in
``agentm.gateway.cli``); this atom only stamps ``metadata.kind`` and stays
transport-agnostic, communicating with the gateway only through services it
gets by name (§11: no ``core.runtime.*`` import, no atom-to-atom import).

Service contract (set by SessionManager before install):

* ``wire_outbound``    -> ``async (body_dict) -> None`` outbound sink.
* ``session_key``      -> ``str`` (echoed back so the gateway routes the
  outbound to the right chat client).
* ``turn_context``     -> mutable dict with ``channel`` / ``chat_id`` /
  ``thread_id`` / ``sender_id`` for the current turn.
* ``approval_manager`` -> optional object exposing ``requires(name)`` and
  ``request(...)``; when present, write-class tools are gated through it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import AssistantMessage, TextContent
from agentm.core.abi.events import (
    AfterCompactEvent,
    AgentEndEvent,
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    CommandDispatchedEvent,
    CostBudgetExceededEvent,
    DiagnosticEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    ExtensionUnloadEvent,
    PlanSubmittedEvent,
    ResourceWriteEvent,
    SessionReadyEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.stream import TextDelta, ThinkingDelta
from agentm.core.lib import to_jsonable
from agentm.extensions import ExtensionManifest

# A projector turns one event into an outbound body, a list of bodies, or
# nothing. Each body MUST carry ``kind``; an optional ``content`` is the
# primary human text, every other key rides in ``metadata``.
ProjectorResult = dict[str, Any] | list[dict[str, Any]] | None
Projector = Callable[[Any], ProjectorResult]

_PREVIEW_LIMIT = 4000

class WireDriverConfig(BaseModel):
    model_config = {"extra": "allow"}


MANIFEST = ExtensionManifest(
    name="wire_driver",
    description=(
        "Translate AgentSession bus events into wire outbound envelope bodies "
        "for the single-process gateway, via a declarative projector table. "
        "Installed per chat session by the gateway's SessionManager; reads "
        "wire_outbound / session_key / turn_context / approval_manager "
        "services. Forwards the full surfaceable event taxonomy — conversation "
        "(stream text/thinking, tool call/result, turn, usage, child sessions) "
        "and runtime control/observability (extension install/reload/unload, "
        "api_register, injected user messages, resource writes, compaction, "
        "budget, session_ready) — discriminated by metadata.kind. Gates "
        "write-class tools through the approval_manager when policy requires."
    ),
    registers=(
        "event:turn_start",
        "event:turn_end",
        "event:stream_delta",
        "event:tool_call",
        "event:tool_result",
        "event:child_session_start",
        "event:child_session_end",
        "event:diagnostic",
        "event:agent_end",
        "event:extension_install",
        "event:extension_reload",
        "event:extension_unload",
        "event:api_register",
        "event:api_send_user_message",
        "event:resource_write",
        "event:plan_submitted",
        "event:after_compact",
        "event:cost_budget_exceeded",
        "event:session_ready",
        "event:command_dispatched",
    ),
    config_schema=WireDriverConfig,
)


def _assistant_text(message: AssistantMessage) -> str:
    return "\n".join(
        block.text
        for block in message.content
        if isinstance(block, TextContent) and block.text
    )


def _content_text(blocks: Any, limit: int = _PREVIEW_LIMIT) -> str:
    """Best-effort text view of a content-block list (tool results, injected
    messages). Duck-typed on ``.text`` so no extra ABI import is needed."""
    if isinstance(blocks, str):
        return blocks[:limit]
    parts: list[str] = []
    try:
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    except TypeError:
        return str(blocks)[:limit]
    return "\n".join(parts)[:limit]


# --- projectors (pure: event -> JSON-safe body) ----------------------------


def _p_turn_start(ev: TurnStartEvent) -> ProjectorResult:
    return {"kind": "turn_start", "turn_id": ev.turn_id, "turn_index": ev.turn_index}


def _p_stream_delta(ev: StreamDeltaEvent) -> ProjectorResult:
    delta = ev.delta
    if isinstance(delta, TextDelta):
        return (
            {"kind": "stream_text", "content": delta.text, "turn_id": ev.turn_id}
            if delta.text
            else None
        )
    if isinstance(delta, ThinkingDelta):
        return (
            {"kind": "stream_thinking", "content": delta.text, "turn_id": ev.turn_id}
            if delta.text
            else None
        )
    # ToolCallStart / ArgsDelta / End / MessageEnd are surfaced via the
    # tool_call / tool_result / turn_end channels instead.
    return None


def _p_tool_call(ev: ToolCallEvent) -> ProjectorResult:
    return {
        "kind": "tool_call",
        "tool_call_id": ev.tool_call_id,
        "name": ev.tool_name,
        "args": to_jsonable(ev.args),
    }


def _p_tool_result(ev: ToolResultEvent) -> ProjectorResult:
    # Covers both success and error: an errored tool still flows through
    # ToolResultEvent with ``result.is_error=True`` (the loop emits
    # ToolErrorEvent only to let an atom fill the result content, then emits
    # this with the same instance), so there is no separate tool_error frame.
    return {
        "kind": "tool_result",
        "tool_call_id": ev.tool_call_id,
        "name": ev.tool_name,
        "ok": not ev.result.is_error,
        "content": _content_text(ev.result.content),
    }


def _p_turn_end(ev: TurnEndEvent) -> ProjectorResult:
    bodies: list[dict[str, Any]] = []
    text = _assistant_text(ev.message)
    if text.strip():
        bodies.append({"kind": "assistant_text", "content": text})
    usage = ev.message.usage
    if usage is not None:
        bodies.append(
            {
                "kind": "usage",
                "turn_id": ev.turn_id,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read": usage.cache_read,
                "cache_write": usage.cache_write,
            }
        )
    return bodies or None


def _p_child_start(ev: ChildSessionStartEvent) -> ProjectorResult:
    return {
        "kind": "child_start",
        "child_id": ev.child_session_id,
        "purpose": ev.purpose,
    }


def _p_child_end(ev: ChildSessionEndEvent) -> ProjectorResult:
    return {
        "kind": "child_end",
        "child_id": ev.child_session_id,
        "final_message_count": ev.final_message_count,
        "error": ev.error,
    }


def _p_diagnostic(ev: DiagnosticEvent) -> ProjectorResult:
    if ev.level == "warning":
        return {"kind": "diagnostic_warning", "content": ev.message, "source": ev.source}
    if ev.level == "error":
        return {"kind": "diagnostic_error", "content": ev.message, "source": ev.source}
    return None  # info is not surfaced


def _p_agent_end(ev: AgentEndEvent) -> ProjectorResult:
    return {"kind": "agent_end", "cause": type(ev.cause).__name__}


def _p_extension_install(ev: ExtensionInstallEvent) -> ProjectorResult:
    return {
        "kind": "extension_install",
        "module_path": ev.module_path,
        "phase": ev.phase,
        "trigger": ev.trigger,
        "error": ev.error,
    }


def _p_extension_reload(ev: ExtensionReloadEvent) -> ProjectorResult:
    return {
        "kind": "extension_reload",
        "name": ev.name,
        "is_self_modify": ev.is_self_modify,
        "trigger": ev.trigger,
        "error": ev.error,
    }


def _p_extension_unload(ev: ExtensionUnloadEvent) -> ProjectorResult:
    return {
        "kind": "extension_unload",
        "name": ev.name,
        "trigger": ev.trigger,
        "error": ev.error,
    }


def _p_api_register(ev: ApiRegisterEvent) -> ProjectorResult:
    # ``reg_kind`` (not ``kind``) so it does not collide with the wire
    # metadata.kind discriminator.
    return {
        "kind": "api_register",
        "reg_kind": ev.kind,
        "name": ev.name,
        "extension": ev.extension,
    }


def _p_api_send_user_message(ev: ApiSendUserMessageEvent) -> ProjectorResult:
    return {
        "kind": "api_send_user_message",
        "extension": ev.extension,
        "content": _content_text(ev.content),
    }


def _p_resource_write(ev: ResourceWriteEvent) -> ProjectorResult:
    return {
        "kind": "resource_write",
        "path": ev.path,
        "author": ev.author,
        "rationale": ev.rationale,
        "post_sha": ev.post_sha,
    }


def _p_plan_submitted(ev: PlanSubmittedEvent) -> ProjectorResult:
    return {"kind": "plan_submitted", "plan_id": ev.plan_id, "content": ev.plan_text}


def _p_after_compact(ev: AfterCompactEvent) -> ProjectorResult:
    return {
        "kind": "after_compact",
        "kept": ev.kept_message_count,
        "discarded": ev.discarded_message_count,
        "content": ev.summary,
    }


def _p_cost_budget(ev: CostBudgetExceededEvent) -> ProjectorResult:
    return {
        "kind": "cost_budget_exceeded",
        "used": ev.used,
        "limit": ev.limit,
        "currency": ev.currency,
    }


def _p_session_ready(ev: SessionReadyEvent) -> ProjectorResult:
    return {
        "kind": "session_ready",
        "tool_names": list(ev.tool_names),
        "command_names": list(ev.command_names),
        "model": ev.model.id if ev.model is not None else None,
    }


def _p_command_dispatched(ev: CommandDispatchedEvent) -> ProjectorResult:
    return {
        "kind": "command_dispatched",
        "name": ev.name,
        "args": ev.args,
        "owner": ev.owner,
    }


# Channels dispatched via async ``bus.emit`` — async handlers preserve order
# and backpressure on the streaming hot path.
_ASYNC_PROJECTORS: tuple[tuple[str, Projector], ...] = (
    (TurnStartEvent.CHANNEL, _p_turn_start),
    (StreamDeltaEvent.CHANNEL, _p_stream_delta),
    (ToolCallEvent.CHANNEL, _p_tool_call),
    (ToolResultEvent.CHANNEL, _p_tool_result),
    (TurnEndEvent.CHANNEL, _p_turn_end),
    (ChildSessionStartEvent.CHANNEL, _p_child_start),
    (ChildSessionEndEvent.CHANNEL, _p_child_end),
    (AgentEndEvent.CHANNEL, _p_agent_end),
)

# Runtime control/observability channels dispatched via ``bus.emit_sync`` (or
# whose ordering is not load-bearing). A SYNC handler is required here — an
# async handler is silently skipped by ``emit_sync`` — so these schedule the
# async sink on the running loop.
_SYNC_PROJECTORS: tuple[tuple[str, Projector], ...] = (
    # ``diagnostic`` can be dispatched via emit_sync (e.g. recovery-floor
    # diagnostics), where an async handler is silently skipped — so it lives
    # here. A sync handler is valid under both emit and emit_sync.
    (DiagnosticEvent.CHANNEL, _p_diagnostic),
    (ExtensionInstallEvent.CHANNEL, _p_extension_install),
    (ExtensionReloadEvent.CHANNEL, _p_extension_reload),
    (ExtensionUnloadEvent.CHANNEL, _p_extension_unload),
    (ApiRegisterEvent.CHANNEL, _p_api_register),
    (ApiSendUserMessageEvent.CHANNEL, _p_api_send_user_message),
    (ResourceWriteEvent.CHANNEL, _p_resource_write),
    (PlanSubmittedEvent.CHANNEL, _p_plan_submitted),
    (AfterCompactEvent.CHANNEL, _p_after_compact),
    (CostBudgetExceededEvent.CHANNEL, _p_cost_budget),
    (SessionReadyEvent.CHANNEL, _p_session_ready),
    (CommandDispatchedEvent.CHANNEL, _p_command_dispatched),
)


def install(api: ExtensionAPI, config: WireDriverConfig) -> None:  # noqa: ARG001
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
    # Holds scheduled control-frame tasks so the loop does not GC them mid-flight.
    bg_tasks: set[asyncio.Task[Any]] = set()

    def _addr() -> dict[str, Any]:
        ctx = turn_context or {}
        return {
            "channel": str(ctx.get("channel") or ""),
            "chat_id": str(ctx.get("chat_id") or ""),
            "thread_id": ctx.get("thread_id"),
        }

    async def _ship(proj: dict[str, Any]) -> None:
        kind = proj.pop("kind")
        content = proj.pop("content", "")
        addr = _addr()
        body: dict[str, Any] = {
            "channel": addr["channel"],
            "chat_id": addr["chat_id"],
            "content": content,
            "metadata": {"kind": kind, **proj},
            # Echoed onto the envelope by the gateway sink so a multi-surface
            # client can attribute this outbound to its conversation (§2.5).
            "_session_key": session_key,
        }
        if addr["thread_id"] is not None:
            body["thread_id"] = addr["thread_id"]
        await outbound_sink(body)

    def _bodies(result: ProjectorResult) -> list[dict[str, Any]]:
        if result is None:
            return []
        return [dict(result)] if isinstance(result, dict) else [dict(b) for b in result]

    def _make_async(projector: Projector) -> Callable[[Any], Any]:
        async def handler(ev: Any) -> None:
            for body in _bodies(projector(ev)):
                await _ship(body)

        return handler

    def _make_sync(projector: Projector) -> Callable[[Any], Any]:
        def handler(ev: Any) -> None:
            bodies = _bodies(projector(ev))
            if not bodies:
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no loop — emitted outside the gateway; nothing to do
            for body in bodies:
                task = loop.create_task(_ship(body))
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)

        return handler

    async def _approval_gate(ev: ToolCallEvent) -> dict[str, Any] | None:
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
                "reason": (
                    f"tool '{ev.tool_name}' was denied via the chat approval gate"
                ),
            }
        return None

    for channel, projector in _ASYNC_PROJECTORS:
        api.on(channel, _make_async(projector))
    for channel, projector in _SYNC_PROJECTORS:
        api.on(channel, _make_sync(projector))
    # Approval gate runs alongside the tool_call projector; it returns a block
    # decision the loop acts on, independent of the outbound forwarding.
    api.on(ToolCallEvent.CHANNEL, _approval_gate)
