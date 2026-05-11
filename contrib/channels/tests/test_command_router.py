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
    CommandRouter,
    parse_invocation,
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


def test_parse_plain_command() -> None:
    inv = parse_invocation(_inbound("/new"))
    assert inv is not None
    assert inv.name == "new"
    assert inv.namespace is None
    assert inv.args == ""


def test_parse_command_with_args_preserves_whitespace() -> None:
    inv = parse_invocation(_inbound("/skill:feishu-cli   list members"))
    assert inv is not None
    assert inv.namespace == "skill"
    assert inv.name == "feishu-cli"
    # First whitespace run is consumed as the args separator; everything
    # else is preserved verbatim.
    assert inv.args == "list members"


def test_parse_namespace_lowercased() -> None:
    inv = parse_invocation(_inbound("/SKILL:Foo bar"))
    assert inv is not None
    assert inv.namespace == "skill"
    assert inv.name == "foo"
    assert inv.args == "bar"


def test_parse_bare_slash_yields_empty_name() -> None:
    inv = parse_invocation(_inbound("/"))
    assert inv is not None
    assert inv.name == ""


def test_parse_double_slash_is_not_a_command() -> None:
    assert parse_invocation(_inbound("//etc/passwd")) is None


def test_parse_non_command_returns_none() -> None:
    assert parse_invocation(_inbound("hello")) is None


# --- router dispatch --------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_command_publishes_user_visible_error() -> None:
    registry = CommandRegistry()
    router = CommandRouter(registry=registry)
    ctx = _ctx()
    result = await router.try_dispatch(_inbound("/does-not-exist"), ctx)
    assert result is not None
    assert result.expanded_prompt is None  # never reaches the LLM
    assert len(result.outbound) == 1
    assert "Unknown command" in result.outbound[0].content


@pytest.mark.asyncio
async def test_empty_command_publishes_hint() -> None:
    router = CommandRouter(registry=CommandRegistry())
    result = await router.try_dispatch(_inbound("/"), _ctx())
    assert result is not None
    assert "/help" in result.outbound[0].content


@pytest.mark.asyncio
async def test_builtin_help_lists_registered_handlers() -> None:
    registry = discover_commands(cwd=Path("/nonexistent-cwd"))
    router = CommandRouter(registry=registry)
    result = await router.try_dispatch(_inbound("/help"), _ctx(registry=registry))
    assert result is not None
    text = result.outbound[0].content
    assert "/help" in text
    assert "/new" in text
    assert "/end" in text
    assert "/status" in text


@pytest.mark.asyncio
async def test_markdown_command_expands_arguments(tmp_path: Path) -> None:
    cmd_dir = tmp_path / ".agentm" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "echo.md").write_text(
        "---\nname: echo\nsummary: Echo args\n---\n"
        "Repeat back: $ARGUMENTS"
    )
    registry = discover_commands(cwd=tmp_path, include_default_skill_paths=False)
    router = CommandRouter(registry=registry)
    result = await router.try_dispatch(
        _inbound("/echo hello world"), _ctx(registry=registry)
    )
    assert result is not None
    assert result.expanded_prompt == "Repeat back: hello world"
    assert result.outbound == []


@pytest.mark.asyncio
async def test_skill_command_injects_body_into_prompt(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".claude" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A demo\n---\n"
        "Demo body content here."
    )
    registry = discover_commands(cwd=tmp_path, include_default_skill_paths=True)
    router = CommandRouter(registry=registry)
    result = await router.try_dispatch(
        _inbound("/skill:demo-skill do thing"), _ctx(registry=registry)
    )
    assert result is not None
    assert result.expanded_prompt is not None
    assert "Demo body content here." in result.expanded_prompt
    assert "do thing" in result.expanded_prompt
    assert "demo-skill" in result.expanded_prompt


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


@pytest.mark.asyncio
async def test_control_command_drops_route_and_does_not_prompt(
    tmp_path: Path,
) -> None:
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
        # First a real message to create a route…
        await stub.push(sender_id="u1", chat_id="c1", content="hello")  # type: ignore[attr-defined]
        await _wait_until(
            lambda: any("got: hello" in o.content for o in stub.outbox)  # type: ignore[attr-defined]
        )
        assert len(_recorded_sessions) == 1
        # …then /new — should reset the route without invoking prompt.
        await stub.push(sender_id="u1", chat_id="c1", content="/new")  # type: ignore[attr-defined]
        await _wait_until(
            lambda: any("Session reset" in o.content for o in stub.outbox)  # type: ignore[attr-defined]
        )
        # Subsequent real message mints a fresh session.
        await stub.push(sender_id="u1", chat_id="c1", content="again")  # type: ignore[attr-defined]
        await _wait_until(lambda: len(_recorded_sessions) >= 2)
        # The first session received only "hello", not "/new".
        assert _recorded_sessions[0].prompts == ["hello"]
        assert _recorded_sessions[1].prompts == ["again"]
    finally:
        await gw.stop()
        await mgr.stop()


@pytest.mark.asyncio
async def test_prompt_command_rewrites_content_into_session(
    tmp_path: Path,
) -> None:
    _recorded_sessions.clear()
    cmd_dir = tmp_path / ".agentm" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "echo.md").write_text(
        "---\nname: echo\nsummary: Echo args\n---\n"
        "Repeat back: $ARGUMENTS"
    )
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
        await stub.push(  # type: ignore[attr-defined]
            sender_id="u1", chat_id="c1", content="/echo hi there"
        )
        await _wait_until(lambda: len(_recorded_sessions) >= 1)
        await _wait_until(
            lambda: _recorded_sessions[0].prompts
            == ["Repeat back: hi there"]
        )
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
