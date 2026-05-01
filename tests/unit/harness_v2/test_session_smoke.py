"""Phase 1 acceptance: end-to-end ``AgentSession`` smoke test.

Per ``.claude/designs/extension-as-scenario.md`` §11. Wires up:
- echo tool extension
- permission-demo extension (passive observer)
- fake provider extension (deterministic two-call stream)

then drives ``session.prompt('hello')`` and asserts the kernel loop ran the
full ReAct cycle, the session manager appended the right entries, and the
event bus saw the expected events.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from typing import Any

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    EventBus,
    TextContent,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
)
from tests.unit.harness_v2._fixtures.fake_provider import FakeStream

from agentm.harness.events import (
    ChildSessionEndEvent,
    ChildSessionStartEvent,
)
from agentm.harness.extension import (
    CommandSpec,
    UnknownCommandError,
)
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_v2_session_smoke(tmp_path: Path) -> None:
    """One ``prompt`` call drives the loop end-to-end via extensions only."""

    # Reset fixture-module global state so re-runs are independent.
    from tests.unit.harness_v2._fixtures import permission_demo

    permission_demo.CALLS_OBSERVED.clear()

    # Subscribe a sniffer to the bus before session.create so we observe the
    # full event sequence (some events fire during prompt itself).
    seen_events: list[str] = []

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            ("tests.unit.harness_v2._fixtures.echo_ext", {}),
            ("tests.unit.harness_v2._fixtures.permission_demo", {}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        # Use an in-memory resource loader to avoid touching real ~/.agentm
        # or filesystem ancestors during the test.
        resource_loader=InMemoryResourceLoader(),
    )

    session = await AgentSession.create(config)

    for channel in (
        "before_agent_start",
        "agent_start",
        "tool_call",
        "tool_result",
        "agent_end",
    ):
        session.bus.on(
            channel,
            lambda _ev, c=channel: seen_events.append(c),
        )

    final = await session.prompt("hello")

    # --- Message shape: user → assistant(tool_call) → tool_result → assistant(text) ---
    assert len(final) == 4
    assert isinstance(final[0], UserMessage)
    assert isinstance(final[1], AssistantMessage)
    assert any(isinstance(b, ToolCallBlock) for b in final[1].content)
    assert isinstance(final[2], ToolResultMessage)
    assert isinstance(final[3], AssistantMessage)
    last_text = final[3].content[0]
    assert isinstance(last_text, TextContent)
    assert last_text.text == "done"

    # --- SessionManager has corresponding entries appended ---
    branch = session.session_manager.get_active_branch()
    # Expect 4 message entries (matching the 4 messages above).
    types = [e.type for e in branch]
    assert types.count("message") == 4

    # --- EventBus sequence ---
    # Permission demo recorded the tool call.
    assert len(permission_demo.CALLS_OBSERVED) == 1
    assert permission_demo.CALLS_OBSERVED[0]["tool_name"] == "echo"
    assert permission_demo.CALLS_OBSERVED[0]["args"] == {"text": "hi"}

    # Sniffer saw all expected events at least once.
    assert "before_agent_start" in seen_events
    assert "agent_start" in seen_events
    assert "tool_call" in seen_events
    assert "tool_result" in seen_events
    assert "agent_end" in seen_events
    # before_agent_start must precede agent_start (system-prompt hook fires
    # before the kernel loop runs).
    assert seen_events.index("before_agent_start") < seen_events.index("agent_start")
    # tool_call precedes tool_result.
    assert seen_events.index("tool_call") < seen_events.index("tool_result")
    # agent_end is last among lifecycle events.
    assert seen_events.index("agent_start") < seen_events.index("agent_end")

    await session.shutdown()


@pytest.mark.asyncio
async def test_v2_session_active_provider_exposed(tmp_path: Path) -> None:
    """``session.model`` and ``session.tools`` reflect the active provider
    and the union of registered tools."""

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            ("tests.unit.harness_v2._fixtures.echo_ext", {}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )

    session = await AgentSession.create(config)

    assert session.model is not None
    assert session.model.id == "fake"
    tool_names = [t.name for t in session.tools]
    assert "echo" in tool_names

    await session.shutdown()


# --- Phase 2.0 additions ---------------------------------------------------


@pytest.mark.asyncio
async def test_child_session_lifecycle_events_fire_on_parent_bus(
    tmp_path: Path,
) -> None:
    """When ``parent_bus`` is supplied, ``child_session_start`` fires at
    ``create`` and ``child_session_end`` at ``shutdown`` — both with the
    child's session_id and the explicitly-provided parent_session_id."""

    parent_bus = EventBus()
    seen_starts: list[ChildSessionStartEvent] = []
    seen_ends: list[ChildSessionEndEvent] = []
    parent_bus.on("child_session_start", lambda e: seen_starts.append(e))
    parent_bus.on("child_session_end", lambda e: seen_ends.append(e))

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            ("tests.unit.harness_v2._fixtures.echo_ext", {}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
        parent_bus=parent_bus,
        parent_session_id="parent-id",
        purpose="subagent:demo",
    )

    child = await AgentSession.create(config)

    assert len(seen_starts) == 1
    assert seen_starts[0].child_session_id == child.session_id
    assert seen_starts[0].parent_session_id == "parent-id"
    assert seen_starts[0].purpose == "subagent:demo"

    await child.prompt("hello")
    await child.shutdown()

    assert len(seen_ends) == 1
    assert seen_ends[0].child_session_id == child.session_id
    assert seen_ends[0].parent_session_id == "parent-id"
    # We ran one full ReAct cycle => 4 messages persisted.
    assert seen_ends[0].final_message_count == 4
    assert seen_ends[0].error is None


# Fixture shared by the slash-command tests below — defined once here so the
# count of call attempts is observable across multiple ``prompt`` invocations
# within a single test.


def _install_ping_command(api: Any, calls: list[str]) -> None:
    def handler(args: str, _api: Any) -> None:
        calls.append(args)
        # ``append_entry`` lives on ReadonlySession (api.session) per
        # extension-as-scenario.md §10b.7 — not on ExtensionAPI directly.
        _api.session.append_entry("pong", {"args": args})

    api.register_command("ping", CommandSpec(description="ping", handler=handler))


@pytest.mark.asyncio
async def test_prompt_dispatches_slash_command_without_calling_stream(
    tmp_path: Path,
) -> None:
    """``/ping`` short-circuits to the registered handler. The provider's
    StreamFn must NOT be invoked."""

    calls: list[str] = []

    # Build a fixture extension on the fly via a sys.modules injection.
    import sys
    import types

    mod = types.ModuleType("tests.unit.harness_v2._fixtures._ping_ext")

    def install(api: Any, _config: dict[str, Any]) -> None:
        _install_ping_command(api, calls)

    mod.install = install  # type: ignore[attr-defined]
    sys.modules["tests.unit.harness_v2._fixtures._ping_ext"] = mod

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            ("tests.unit.harness_v2._fixtures._ping_ext", {}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    # Find the active provider's stream object via the registry to check
    # call counts. The fake provider stores calls on the FakeStream instance.
    fake_stream = cast(FakeStream, session._loop._stream_fn)  # type: ignore[attr-defined]
    pre_calls = fake_stream.calls

    result = await session.prompt("/ping arg1 arg2")

    # Handler ran with the rest-of-line argument string.
    assert calls == ["arg1 arg2"]
    # Stream was NOT invoked — no LLM call for slash commands.
    assert fake_stream.calls == pre_calls
    # No new message was appended for the slash command itself.
    assert result == session.session_manager.get_messages()

    await session.shutdown()


@pytest.mark.asyncio
async def test_prompt_unknown_slash_command_raises(tmp_path: Path) -> None:
    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    with pytest.raises(UnknownCommandError):
        await session.prompt("/no-such-command")

    # Bare "/" is also unknown.
    with pytest.raises(UnknownCommandError):
        await session.prompt("/")

    await session.shutdown()


@pytest.mark.asyncio
async def test_double_slash_escapes_to_literal_user_message(tmp_path: Path) -> None:
    """``//ping`` becomes a normal user message of ``/ping`` and goes through
    the agent loop (StreamFn IS called)."""

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            ("tests.unit.harness_v2._fixtures.echo_ext", {}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    fake_stream = cast(FakeStream, session._loop._stream_fn)  # type: ignore[attr-defined]
    pre_calls = fake_stream.calls

    final = await session.prompt("//ping")

    # StreamFn was invoked (loop ran the full ReAct cycle).
    assert fake_stream.calls > pre_calls

    # The first user message has its literal text de-escaped to "/ping".
    user_msgs = [m for m in final if isinstance(m, UserMessage)]
    assert user_msgs, "expected a UserMessage in the final list"
    first_block = user_msgs[0].content[0]
    assert isinstance(first_block, TextContent)
    assert first_block.text == "/ping"

    await session.shutdown()
