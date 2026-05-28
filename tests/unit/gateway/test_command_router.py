"""Fail-stop: CommandRouter handler classes (§3.5).

Slash commands are intercepted before the LLM sees them. A leak (unknown
command forwarded to the model, /end not clearing the map, a prompt
command not falling through) either exposes prompts or breaks the
session lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentm.gateway.commands import (
    CommandContext,
    CommandInbound,
    CommandRegistry,
    CommandRouter,
)
from agentm.gateway.commands.builtins.end import EndCommand
from agentm.gateway.commands.builtins.help import HelpCommand
from agentm.gateway.commands.builtins.new import NewCommand
from agentm.gateway.commands.markdown_command import MarkdownPromptCommand


@dataclass
class _Calls:
    ended: int = 0
    forgot: int = 0


def _ctx(calls: _Calls, registry: CommandRegistry) -> CommandContext:
    async def end_session() -> None:
        calls.ended += 1

    async def forget() -> None:
        calls.forgot += 1

    return CommandContext(
        session_key="terminal:t1",
        channel="terminal",
        chat_id="t1",
        sender_id="u1",
        thread_id=None,
        end_session=end_session,
        forget_chat_mapping=forget,
        get_route_stats=lambda: {"session_id": "sid-1", "turn_count": 2, "pending_approvals": 0},
        list_commands=registry.all,
    )


def _inbound(content: str) -> CommandInbound:
    return CommandInbound(
        session_key="terminal:t1",
        channel="terminal",
        chat_id="t1",
        sender_id="u1",
        content=content,
    )


def _registry(*handlers: Any) -> CommandRegistry:
    reg = CommandRegistry()
    for h in handlers:
        reg.register(h)
    return reg


@pytest.mark.asyncio
async def test_non_command_returns_none() -> None:
    reg = _registry(HelpCommand())
    router = CommandRouter(registry=reg)
    result = await router.try_dispatch(_inbound("just chatting"), _ctx(_Calls(), reg))
    assert result is None


@pytest.mark.asyncio
async def test_help_replies_locally() -> None:
    reg = _registry(HelpCommand())
    router = CommandRouter(registry=reg)
    result = await router.try_dispatch(_inbound("/help"), _ctx(_Calls(), reg))
    assert result is not None
    assert result.expanded_prompt is None
    assert len(result.outbound) == 1
    assert "commands" in result.outbound[0].content.lower()


@pytest.mark.asyncio
async def test_unknown_command_rejected_not_forwarded() -> None:
    reg = _registry(HelpCommand())
    router = CommandRouter(registry=reg)
    result = await router.try_dispatch(_inbound("/nope"), _ctx(_Calls(), reg))
    assert result is not None
    assert result.expanded_prompt is None  # never reaches the LLM
    assert result.outbound[0].metadata.get("kind") == "diagnostic_error"


@pytest.mark.asyncio
async def test_new_ends_session_keeps_map() -> None:
    reg = _registry(NewCommand())
    router = CommandRouter(registry=reg)
    calls = _Calls()
    result = await router.try_dispatch(_inbound("/new"), _ctx(calls, reg))
    assert result is not None
    if result.side_effect is None:
        # NewCommand calls end_session in handle(); confirm it fired.
        assert calls.ended == 1
    assert calls.forgot == 0  # /new must not clear the map


@pytest.mark.asyncio
async def test_end_ends_session_and_clears_map() -> None:
    reg = _registry(EndCommand())
    router = CommandRouter(registry=reg)
    calls = _Calls()
    await router.try_dispatch(_inbound("/end"), _ctx(calls, reg))
    assert calls.ended == 1
    assert calls.forgot == 1  # /end clears the map for a cold start


@pytest.mark.asyncio
async def test_markdown_command_expands_and_falls_through(tmp_path: Any) -> None:
    md = tmp_path / "standup.md"
    md.write_text("Summarise $ARGUMENTS in one line.", encoding="utf-8")
    handler = MarkdownPromptCommand.from_path(md)
    assert handler is not None
    reg = _registry(handler)
    router = CommandRouter(registry=reg)
    result = await router.try_dispatch(
        _inbound("/standup yesterday's commits"), _ctx(_Calls(), reg)
    )
    assert result is not None
    # Prompt command: expanded text falls through to the session.
    assert result.expanded_prompt == "Summarise yesterday's commits in one line."
