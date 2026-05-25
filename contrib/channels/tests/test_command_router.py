"""Command parsing + routing + dispatch interception.

These are fail-stop tests for the command layer: they assert the
interception contract (unknown command does not reach LLM; control
command does not reach LLM; prompt command rewrites the inbound and
*does* reach LLM).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import AssistantMessage, EventBus, TextContent
from agentm.core.abi.events import TurnEndEvent

from agentm_channels.bus import InboundMessage, MessageBus
from agentm_channels.commands import (
    CommandRegistry,
)
from agentm_channels.commands.protocol import CommandContext
from agentm_channels.commands.registry import discover_commands
from agentm_channels.gateway import Gateway, GatewayConfig
from agentm_channels.manager import ChannelManager


# --- parse_invocation -------------------------------------------------


def _inbound(content: str) -> InboundMessage:
    return InboundMessage(
        channel="stub", sender_id="u", chat_id="c", content=content
    )














# --- router dispatch --------------------------------------------------












# --- gateway-level interception ---------------------------------------


class _RecordingSession:
    """Records every prompt() — lets us assert what reached the LLM."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self.prompts: list[str] = []
        self.session_manager = type("SM", (), {"get_session_id": lambda self: "fake-1"})()

    async def prompt(self, text: str) -> None:
        self.prompts.append(text)
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=f"got: {text}")],
            timestamp=time.time(),
        )
        await self._bus.emit(
            TurnEndEvent.CHANNEL,
            TurnEndEvent(turn_index=0, message=msg, messages=()),
        )

    async def shutdown(self) -> None:
        pass


_recorded_sessions: list[_RecordingSession] = []


async def _recording_factory(_cwd: str, bus: EventBus, _resume: str | None) -> Any:
    s = _RecordingSession(bus)
    _recorded_sessions.append(s)
    return s


async def _wait_until(pred, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timeout")


@pytest.mark.asyncio
async def test_unknown_slash_command_does_not_reach_session(tmp_path: Path) -> None:
    _recorded_sessions.clear()
    bus = MessageBus()
    mgr = ChannelManager({"stub": {"enabled": True, "allow_from": ["*"]}}, bus)
    gw = Gateway(
        bus=bus,
        config=GatewayConfig(
            cwd=str(tmp_path),
            state_dir=tmp_path / "state",
            command_registry=discover_commands(
                cwd=tmp_path, include_default_skill_paths=False
            ),
        ),
        session_factory=_recording_factory,
    )
    await mgr.start()
    await gw.start()
    try:
        stub = mgr.channels["stub"]
        await stub.push(sender_id="u1", chat_id="c1", content="/nope")  # type: ignore[attr-defined]
        await _wait_until(
            lambda: any("Unknown command" in o.content for o in stub.outbox)  # type: ignore[attr-defined]
        )
        # Critical assertion: no session was constructed because no
        # non-command message ever reached the route lookup.
        assert _recorded_sessions == []
    finally:
        await gw.stop()
        await mgr.stop()






# --- helpers ----------------------------------------------------------


def _ctx(*, registry: CommandRegistry | None = None) -> CommandContext:
    async def _drop() -> None: ...
    def _stats() -> dict[str, Any]:
        return {"session_id": None, "turn_count": 0, "pending_approvals": 0}
    reg = registry or CommandRegistry()
    return CommandContext(
        route_key="stub:c",
        channel="stub",
        chat_id="c",
        sender_id="u",
        drop_route=_drop,
        get_route_stats=_stats,
        list_commands=reg.all,
        approval_bridge=None,
    )
