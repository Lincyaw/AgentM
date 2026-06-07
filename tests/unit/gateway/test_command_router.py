"""Fail-stop: CommandRouter handler classes (§3.5).

Slash commands are intercepted before the LLM sees them. A leak (unknown
command forwarded to the model, /new not clearing the map, a prompt
command not falling through) either exposes prompts or breaks the
session lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentm.gateway.commands import (
    CommandContext,
    CommandInbound,
    CommandRegistry,
    CommandRouter,
    parse_invocation,
)
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
async def test_unknown_gateway_command_returns_none_for_forwarding() -> None:
    # A name the gateway registry doesn't own now returns None (not a
    # diagnostic): the gateway forwards such a command to the session so the
    # in-session slash_commands atom can dispatch session-registered commands
    # like /compact. Unknown-to-both feedback moved into the gateway, which
    # holds the per-session known-command set.
    reg = _registry(HelpCommand())
    router = CommandRouter(registry=reg)
    result = await router.try_dispatch(_inbound("/nope"), _ctx(_Calls(), reg))
    assert result is None


@pytest.mark.asyncio
async def test_bare_slash_still_hints_at_gateway() -> None:
    # A bare "/" must NOT return None (which would forward it to the session);
    # the gateway answers it with a "type /help" hint.
    reg = _registry(HelpCommand())
    router = CommandRouter(registry=reg)
    result = await router.try_dispatch(_inbound("/"), _ctx(_Calls(), reg))
    assert result is not None
    assert result.expanded_prompt is None
    assert "help" in result.outbound[0].content.lower()


@pytest.mark.asyncio
async def test_new_ends_session_and_clears_map() -> None:
    reg = _registry(NewCommand())
    router = CommandRouter(registry=reg)
    calls = _Calls()
    result = await router.try_dispatch(_inbound("/new"), _ctx(calls, reg))
    assert result is not None
    assert calls.ended == 1
    assert calls.forgot == 1  # /new must clear the map for a fresh start


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


def _model_ctx(
    *,
    list_models: Any = None,
    switch_model: Any = None,
) -> CommandContext:
    async def _noop() -> None:
        return None

    async def _no_switch(_name: str) -> tuple[bool, str]:
        return (False, "unsupported")

    return CommandContext(
        session_key="terminal:t1",
        channel="terminal",
        chat_id="t1",
        sender_id="u1",
        thread_id=None,
        end_session=_noop,
        forget_chat_mapping=_noop,
        get_route_stats=lambda: {},
        list_commands=lambda: [],
        list_models=list_models or (lambda: ("", [])),
        switch_model=switch_model or _no_switch,
    )


def _inv(content: str):
    """Build a CommandInvocation the way the router does (handlers read .args)."""
    invocation = parse_invocation(_inbound(content))
    assert invocation is not None
    return invocation


@pytest.mark.asyncio
async def test_model_command_is_discovered() -> None:
    from agentm.gateway.commands import discover_commands

    reg = discover_commands(".")
    assert "model" in {h.name for h in reg.all()}


@pytest.mark.asyncio
async def test_model_no_arg_lists_and_marks_active() -> None:
    from agentm.gateway.commands.builtins.model import ModelCommand

    ctx = _model_ctx(list_models=lambda: ("doubao", ["doubao", "deepseek"]))
    res = await ModelCommand().handle(_inv("/model"), ctx)
    body = res.outbound[0].content
    assert "doubao" in body and "deepseek" in body
    assert "active" in body.lower()


@pytest.mark.asyncio
async def test_model_switch_invokes_capability_and_confirms() -> None:
    from agentm.gateway.commands.builtins.model import ModelCommand

    switched: list[str] = []

    async def _switch(name: str) -> tuple[bool, str]:
        switched.append(name)
        return (True, name)

    ctx = _model_ctx(switch_model=_switch)
    res = await ModelCommand().handle(_inv("/model deepseek"), ctx)
    assert switched == ["deepseek"]
    assert "deepseek" in res.outbound[0].content


@pytest.mark.asyncio
async def test_model_switch_failure_is_a_diagnostic() -> None:
    from agentm.gateway.commands.builtins.model import ModelCommand

    async def _switch(name: str) -> tuple[bool, str]:
        return (False, f"unknown model '{name}'")

    ctx = _model_ctx(switch_model=_switch, list_models=lambda: ("doubao", ["doubao"]))
    res = await ModelCommand().handle(_inv("/model nope"), ctx)
    assert res.outbound[0].metadata.get("kind") == "diagnostic_error"
    assert "nope" in res.outbound[0].content


def test_merge_gateway_commands_folds_builtins(tmp_path: Any) -> None:
    # session_ready frames must gain the gateway's top-level command names so
    # chat clients surface /model, /new, /status in autocomplete.
    from agentm.gateway.chat_session_map import ChatSessionMap
    from agentm.gateway.cli import _GatewayRuntime
    from agentm.gateway.commands import discover_commands
    from agentm.gateway.outbox import SqliteOutbox

    outbox = SqliteOutbox(str(tmp_path / "o.sqlite"))
    try:
        runtime = _GatewayRuntime(
            cwd=".",
            scenario="local",
            outbox=outbox,
            chat_map=ChatSessionMap(tmp_path / "m.json"),
            session_factory=lambda *a: None,  # unused by this method
            command_router=CommandRouter(registry=discover_commands(".")),
            approval_policy=(frozenset(), frozenset(), 300.0),
            model_name="doubao",
            make_factory=lambda _n: (lambda *a: None),
        )
        meta: dict[str, Any] = {"kind": "session_ready", "command_names": ["mytool"]}
        runtime._merge_gateway_commands(meta)
        names = meta["command_names"]
        assert "mytool" in names  # session-provided names preserved
        assert {"model", "new", "status"} <= set(names)
        assert len(names) == len(set(names))  # deduped
        assert not any(":" in n for n in names)  # no namespaced /atom:* entries
    finally:
        outbox.close()


def test_help_lists_session_commands() -> None:
    # /help must surface session-registered commands (e.g. /compact) so users
    # can discover them, not just the gateway builtins.
    from agentm.gateway.commands.builtins.help import HelpCommand, _format_help

    body = _format_help([HelpCommand()], ["compact"])
    assert "session" in body
    assert "/compact" in body
    # A session name that collides with a gateway builtin is not double-listed.
    body2 = _format_help([HelpCommand()], ["help"])
    assert body2.count("/help") == 1
