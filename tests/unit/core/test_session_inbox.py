"""Fail-stop coverage for the SessionInbox spine (step 1).

Design: ``.claude/designs/session-inbox.md`` / plan
``.claude/plans/2026-05-28-session-inbox.md``. The inbox is the single entry
point for messages reaching the loop. These tests pin the load-bearing
positions only (quality over quantity):

* inbox FIFO drain order + ``dedup_key`` replace semantics
* the originating ``prompt`` message lands AND is persisted (parity with the
  pre-inbox path)
* a ``send_user_message`` issued mid-run is seen on the next turn (per-turn
  ``context`` drain)
* ``tick``: empty inbox ⇒ ``NoPendingInput``; non-empty ⇒ runs
* keep-alive floor: the model wants ``Stop(ModelEndTurn)`` but a non-empty
  inbox keeps the loop running

The session-level tests reuse the stub-provider harness pattern from
``test_message_persisted_realtime.py`` — a real ``AgentSession`` driven by an
in-memory stub provider, asserting on bus / session-manager observable state
rather than SDK internals.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AgentEndEvent,
    AssistantMessage,
    MessageEnd,
    Model,
    NoPendingInput,
    TextContent,
)
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession
from agentm.core.runtime.session_inbox import (
    InboxItem,
    SessionInbox,
    render_item,
)


# ---------------------------------------------------------------------------
# SessionInbox unit (no session)
# ---------------------------------------------------------------------------


def test_inbox_drain_is_fifo() -> None:
    inbox = SessionInbox()
    inbox.push(InboxItem(source="user", payload="first"))
    inbox.push(InboxItem(source="user", payload="second"))
    inbox.push(InboxItem(source="user", payload="third"))

    drained = inbox.drain()

    assert [item.payload for item in drained] == ["first", "second", "third"]
    # Drain empties the inbox.
    assert inbox.is_empty()
    assert inbox.drain() == []


def test_inbox_dedup_key_replaces_in_place() -> None:
    inbox = SessionInbox()
    inbox.push(InboxItem(source="ticker", payload="running 10%", dedup_key="job-1"))
    inbox.push(InboxItem(source="user", payload="hello"))
    # Same dedup_key supersedes the earlier item in its original position —
    # no stacking of stale status lines.
    inbox.push(InboxItem(source="ticker", payload="running 90%", dedup_key="job-1"))

    drained = inbox.drain()

    assert [item.payload for item in drained] == ["running 90%", "hello"]
    # Items without a dedup_key never collapse.
    inbox.push(InboxItem(source="user", payload="a"))
    inbox.push(InboxItem(source="user", payload="a"))
    assert len(inbox.drain()) == 2


def test_render_user_item_to_user_message() -> None:
    msg = render_item(InboxItem(source="user", payload="hi there"))
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "hi there"


def test_render_background_item_to_system_reminder() -> None:
    # Step 3: background completion / ticker → a <system-reminder>-wrapped
    # user message (append-only, cache-stable; no synthetic tool_result).
    msg = render_item(InboxItem(source="background", payload="task 7 finished"))
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert "<system-reminder>" in msg.content[0].text
    assert "task 7 finished" in msg.content[0].text


def test_render_unknown_source_raises() -> None:
    # Sources not yet wired (monitor/subagent) must still fail loudly.
    with pytest.raises(NotImplementedError):
        render_item(InboxItem(source="monitor", payload="x"))


@pytest.mark.asyncio
async def test_wait_nonempty_returns_when_pushed() -> None:
    import asyncio

    inbox = SessionInbox()

    async def pusher() -> None:
        inbox.push(InboxItem(source="user", payload="wake"))

    # Already-non-empty returns immediately; an empty inbox blocks until push.
    waiter = asyncio.create_task(inbox.wait_nonempty())
    await asyncio.sleep(0)
    assert not waiter.done()
    await pusher()
    await asyncio.wait_for(waiter, timeout=1.0)


# ---------------------------------------------------------------------------
# Session-level harness (stub provider)
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


def _register_provider(module_name: str, stream_fn: Any) -> types.ModuleType:
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

    setattr(module, "install", install)
    sys.modules[module_name] = module
    return module


async def _make_session(
    tmp_path: Path, module_name: str, stream_fn: Any
) -> AgentSession:
    _register_provider(module_name, stream_fn)
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(module_name, {}),
            extensions=[],
        )
    )


def _texts(messages: list[Any], role: str) -> list[str]:
    out: list[str] = []
    for m in messages:
        if getattr(m, "role", None) != role:
            continue
        for block in getattr(m, "content", []):
            if getattr(block, "type", None) == "text":
                out.append(block.text)
    return out


# ---------------------------------------------------------------------------
# Session-level tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_message_lands_and_is_persisted(tmp_path: Path) -> None:
    """The originating prompt text must reach the LLM as a user message AND
    be persisted to the session log — parity with the pre-inbox path."""
    module_name = f"tests.unit._inbox_prompt_{id(tmp_path)}"
    try:
        session = await _make_session(tmp_path, module_name, _stream_text("ok"))
        messages = await session.prompt("hello world")

        # Returned context contains the originating user message before the
        # assistant reply.
        user_texts = _texts(messages, "user")
        assert "hello world" in user_texts

        # And it is on the durable session log (not just the returned list).
        persisted = _texts(session.session_manager.get_messages(), "user")
        assert "hello world" in persisted
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_send_user_message_seen_on_next_turn(tmp_path: Path) -> None:
    """A ``send_user_message`` pushed mid-run is drained by the per-turn
    ``context`` handler and seen on the very next turn — and the keep-alive
    floor keeps the loop alive for that turn even though the model ends."""
    module_name = f"tests.unit._inbox_midrun_{id(tmp_path)}"

    seen_on_second_turn: list[bool] = []
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
        del model, tools, system, signal, thinking
        idx = turn[0]
        turn[0] += 1
        if idx == 0:
            # Push out-of-band user content while the first turn is in flight,
            # mid-stream, exactly like an atom calling send_user_message.
            api = next(iter(session._apis.values()))  # type: ignore[attr-defined]
            api.send_user_message("mid-run note")
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="first")],
                    timestamp=0.0,
                    stop_reason="end_turn",
                )
            )
        else:
            # Second turn: the mid-run note must be visible in the context.
            seen_on_second_turn.append(
                "mid-run note" in _texts(messages, "user")
            )
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="second")],
                    timestamp=0.0,
                    stop_reason="end_turn",
                )
            )

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        messages = await session.prompt("go")

        # The loop ran a second turn (keep-alive floor fired on the non-empty
        # inbox) and that turn saw the injected note.
        assert seen_on_second_turn == [True]
        assistant_texts = _texts(messages, "assistant")
        assert "first" in assistant_texts and "second" in assistant_texts
        # The injected note is persisted to the session log.
        assert "mid-run note" in _texts(
            session.session_manager.get_messages(), "user"
        )
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_post_inbox_user_matches_send_user_message(tmp_path: Path) -> None:
    """``api.post_inbox(source="user", ...)`` round-trips to a user message on
    the next turn — identical to ``send_user_message`` (its sugar)."""
    module_name = f"tests.unit._inbox_postinbox_{id(tmp_path)}"

    seen: list[bool] = []
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
        del model, tools, system, signal, thinking
        idx = turn[0]
        turn[0] += 1
        if idx == 0:
            api = next(iter(session._apis.values()))  # type: ignore[attr-defined]
            # Generic producer entry, source="user" — the path send_user_message
            # now delegates to.
            api.post_inbox(source="user", payload="posted note")
        else:
            seen.append("posted note" in _texts(messages, "user"))
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"t{idx}")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        await session.prompt("go")
        assert seen == [True]
        assert "posted note" in _texts(
            session.session_manager.get_messages(), "user"
        )
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_tick_empty_inbox_is_no_pending_input(tmp_path: Path) -> None:
    """An empty inbox + no injector ⇒ NoPendingInput, no LLM call, unchanged
    message list (today's tick contract)."""
    module_name = f"tests.unit._inbox_tick_empty_{id(tmp_path)}"

    called = [False]

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
        called[0] = True
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="should not run")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    causes: list[Any] = []
    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        session.bus.on(
            AgentEndEvent.CHANNEL, lambda e: causes.append(e.cause)
        )
        before = len(session.session_manager.get_messages())

        await session.tick()

        assert called[0] is False
        assert any(isinstance(c, NoPendingInput) for c in causes)
        assert len(session.session_manager.get_messages()) == before
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_tick_nonempty_inbox_runs(tmp_path: Path) -> None:
    """A pre-tick inbox item is drained and drives the loop on ``tick``."""
    module_name = f"tests.unit._inbox_tick_run_{id(tmp_path)}"
    try:
        session = await _make_session(tmp_path, module_name, _stream_text("ran"))
        # Push out-of-band content the way an atom would, then resume.
        api = next(iter(session._apis.values()))  # type: ignore[attr-defined]
        api.send_user_message("resume me")

        messages = await session.tick()

        assert "resume me" in _texts(messages, "user")
        assert "ran" in _texts(messages, "assistant")
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_keep_alive_floor_overrides_model_end_turn(tmp_path: Path) -> None:
    """Model wants to stop (Stop(ModelEndTurn)) but the inbox is non-empty at
    the turn boundary ⇒ the keep-alive floor returns Step and the loop runs
    another turn. A non-final default must not short-circuit the floor."""
    module_name = f"tests.unit._inbox_floor_{id(tmp_path)}"

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
        del model, tools, system, signal, thinking
        idx = turn[0]
        turn[0] += 1
        if idx == 0:
            # Queue more work right before the model "ends" this turn. The
            # default decide action will be Stop(ModelEndTurn) (non-final);
            # the floor must turn it into another turn.
            api = next(iter(session._apis.values()))  # type: ignore[attr-defined]
            api.send_user_message("keep going")
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"turn-{idx}")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        messages = await session.prompt("start")

        assistant_texts = _texts(messages, "assistant")
        # Two assistant turns prove the floor kept the loop alive past the
        # model's voluntary end-turn.
        assert "turn-0" in assistant_texts
        assert "turn-1" in assistant_texts
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)
