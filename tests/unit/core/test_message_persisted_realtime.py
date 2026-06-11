"""Fail-stop coverage for real-time message persistence.

The kernel now emits :class:`MessagePersistedEvent` from inside
``AgentLoop.run`` at every durable addition (assistant turn, tool_result,
extension-injected message), and the session subscribes once so each event
flows through ``SessionManager.append_message`` immediately. This guards the
property that a mid-loop kill still leaves every completed turn on disk —
the regression-of-record was a 186-turn session whose trajectory was lost
because the post-loop diff never ran.

Each test below asserts on the count / source of
``MessageAppendedEvent`` records: those are the on-disk
``agentm.message.appended`` rows once observability subscribes, so the bus
count is a faithful proxy without forcing the OTel batch processor to
flush.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    DecideTurnActionEvent,
    Inject,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
    ToolResult,
    UserMessage,
)
from agentm.core.abi import MessageAppendedEvent
from agentm.core.abi import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_text(text: str) -> Any:
    async def stream_fn(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[Any]:
        del messages, model, tools, system, signal, thinking
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=text)],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    return stream_fn


def _stream_then_tool_call() -> Any:
    """First call: assistant emits one tool_call. Second call: plain text."""
    turn = [0]

    async def stream_fn(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[Any]:
        del messages, model, tools, system, signal, thinking
        idx = turn[0]
        turn[0] += 1
        if idx == 0:
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            type="tool_call",
                            id="call-1",
                            name="echo",
                            arguments={"x": 1},
                        )
                    ],
                    timestamp=0.0,
                    stop_reason="tool_use",
                )
            )
        else:
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="done")],
                    timestamp=0.0,
                    stop_reason="end_turn",
                )
            )

    return stream_fn


def _stream_then_cancel() -> Any:
    """First call: full assistant turn. Second call: raise CancelledError."""
    turn = [0]

    async def stream_fn(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[Any]:
        del messages, model, tools, system, signal, thinking
        idx = turn[0]
        turn[0] += 1
        if idx == 0:
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="turn-one")],
                    timestamp=0.0,
                    stop_reason="end_turn",
                )
            )
            return
        # Second turn: nuke the task.
        import asyncio as _asyncio

        raise _asyncio.CancelledError()
        yield  # pragma: no cover  (keeps function an async generator)

    return stream_fn


class _EchoTool:
    name = "echo"
    description = "echo"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(
        self, args: dict[str, Any], *, signal: Any = None
    ) -> ToolResult:
        del args, signal
        return ToolResult(
            content=[TextContent(type="text", text="echo-ok")],
            is_error=False,
        )


def _register_provider(
    module_name: str, stream_fn: Any, *, register_echo: bool = False
) -> types.ModuleType:
    module = types.ModuleType(module_name)

    def install(api: Any, config: dict[str, Any]) -> None:
        del config
        api.register_provider(
            "stub",
            ProviderConfig(
                stream_fn=stream_fn,
                model=Model(
                    id="stub-model",
                    provider="stub",
                    context_window=1024,
                    max_output_tokens=64,
                ),
                name="stub",
            ),
        )
        if register_echo:
            api.register_tool(_EchoTool())

    setattr(module, "install", install)
    sys.modules[module_name] = module
    return module


async def _make_session(
    tmp_path: Path, module_name: str, stream_fn: Any, *, register_echo: bool = False
) -> AgentSession:
    _register_provider(module_name, stream_fn, register_echo=register_echo)
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(module_name, {}),
            extensions=[],
        )
    )


def _role(record: dict[str, Any]) -> str:
    """Pull the message role from a serialized MessageAppendedEvent record."""
    payload = record.get("payload")
    if isinstance(payload, dict):
        role = payload.get("role")
        if isinstance(role, str):
            return role
    return ""


def _collect_appended(session: AgentSession) -> list[dict[str, Any]]:
    """Subscribe to ``MessageAppendedEvent`` and return the live list."""
    records: list[dict[str, Any]] = []
    session.bus.on(
        MessageAppendedEvent.CHANNEL, lambda e: records.append(e.record)
    )
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------








@pytest.mark.asyncio
async def test_kill_midloop_preserves_completed_turns(tmp_path: Path) -> None:
    """If the loop is force-cancelled on its second turn, the first turn's
    assistant message must already be persisted. Pre-change this would lose
    everything because persistence happened post-loop.
    """
    module_name = f"tests.unit._persist_cancel_{id(tmp_path)}"

    # Make the loop step a second time by returning Inject from
    # decide_turn_action — the kernel default Stop(ModelEndTurn) would
    # otherwise end the run cleanly after turn 0.
    def _force_second_turn(event: DecideTurnActionEvent) -> Any:
        if event.observation.turn_index != 0:
            return None
        return Inject(
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text="continue")],
                    timestamp=0.0,
                )
            ]
        )

    try:
        session = await _make_session(
            tmp_path, module_name, _stream_then_cancel()
        )
        records = _collect_appended(session)
        session.bus.on(DecideTurnActionEvent.CHANNEL, _force_second_turn)

        import asyncio as _asyncio

        with pytest.raises(_asyncio.CancelledError):
            await session.prompt("go")

        roles_seen = [_role(r) for r in records]
        # First completed turn's assistant message must already be on the
        # log — even though the second turn cancelled the task.
        assert "user" in roles_seen
        assert roles_seen.count("assistant") >= 1
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)
