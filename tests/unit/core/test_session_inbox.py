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


def test_render_monitor_item_to_system_reminder() -> None:
    # Step 4: monitor wakeups / channel fires use the same
    # <system-reminder>-wrapped UserMessage shape as background — same
    # cache-stability reasons.
    msg = render_item(InboxItem(source="monitor", payload="wakeup fired"))
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert "<system-reminder>" in msg.content[0].text
    assert "wakeup fired" in msg.content[0].text


def test_render_subagent_item_to_system_reminder() -> None:
    # Step 5b: sub_agent findings posted via api.post_inbox(source="subagent")
    # use the same <system-reminder>-wrapped UserMessage shape as background /
    # monitor — same cache-stability reasons.
    msg = render_item(
        InboxItem(source="subagent", payload="<subagent_result task_id=t1 />")
    )
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert "<system-reminder>" in msg.content[0].text
    assert "<subagent_result" in msg.content[0].text


def test_render_unknown_source_raises() -> None:
    with pytest.raises(NotImplementedError):
        render_item(InboxItem(source="totally-unknown", payload="x"))


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
async def test_post_inbox_user_seen_on_next_turn(tmp_path: Path) -> None:
    """A ``post_inbox(source="user", ...)`` push mid-run is drained by the
    per-turn ``context`` handler and seen on the very next turn. The runtime
    keep-alive floor keeps the loop alive for that turn even though the model
    ended voluntarily.

    Migrated from the Nit-1 ``session._apis``-reach pattern: the public
    producer entry is ``api.post_inbox`` (added in step 3); the test pushes
    onto the inbox directly so no SDK internals are touched.
    """
    module_name = f"tests.unit._inbox_midrun_{id(tmp_path)}"

    seen_on_second_turn: list[bool] = []
    turn = [0]
    inbox_handle: list[SessionInbox] = []

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
            # Push out-of-band user content mid-stream, exactly like an atom
            # calling api.post_inbox(source="user", ...).
            inbox_handle[0].push(
                InboxItem(source="user", payload="mid-run note")
            )
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
        # Capture the inbox handle the same way an atom would receive one
        # via ExtensionAPI scope — by mounting an atom and reading off
        # api.post_inbox. Tests assert against the inbox spine directly to
        # stay independent of the producer-entry sugar layer.
        inbox_handle.append(session._inbox)  # type: ignore[attr-defined]
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
        # Push out-of-band content directly onto the inbox spine — same
        # path ``api.post_inbox(source="user", ...)`` takes, no SDK-internal
        # reach into ``session._apis``.
        session._inbox.push(  # type: ignore[attr-defined]
            InboxItem(source="user", payload="resume me")
        )

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
    inbox_handle: list[SessionInbox] = []

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
            inbox_handle[0].push(
                InboxItem(source="user", payload="keep going")
            )
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
        inbox_handle.append(session._inbox)  # type: ignore[attr-defined]
        messages = await session.prompt("start")

        assistant_texts = _texts(messages, "assistant")
        # Two assistant turns prove the floor kept the loop alive past the
        # model's voluntary end-turn.
        assert "turn-0" in assistant_texts
        assert "turn-1" in assistant_texts
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# Step-5 fail-stop coverage: persistent driver + prompt/tick sugar + interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_driver_runs_exactly_once_per_push_burst(tmp_path: Path) -> None:
    """A single push (the prompt's user message) drives exactly one
    ``loop.run`` round, and the driver is the only thing calling it.
    Concurrent direct ``_loop.run`` calls must trip the single-ownership
    assertion."""
    module_name = f"tests.unit._driver_push_burst_{id(tmp_path)}"

    run_count = [0]

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
        run_count[0] += 1
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"r{run_count[0]}")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        await session.prompt("first")
        await session.prompt("second")
        # Two prompts → two driver rounds → two LLM stream calls.
        assert run_count[0] == 2

        # Single-ownership: a direct concurrent ``_loop.run`` while the driver
        # might pick it up MUST raise. We trip the assertion synchronously by
        # invoking ``_run_one_round`` from outside the driver while a prior
        # round is still in flight. Simpler proof: pre-flip ``_in_run`` and
        # try to call ``_run_one_round`` directly.
        session._in_run = True  # type: ignore[attr-defined]
        with pytest.raises(AssertionError):
            await session._run_one_round()  # type: ignore[attr-defined]
        session._in_run = False  # type: ignore[attr-defined]
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_driver_survives_run_exception(tmp_path: Path) -> None:
    """A round-level exception MUST NOT kill the driver: the next push still
    drives a fresh run."""
    module_name = f"tests.unit._driver_survives_{id(tmp_path)}"

    call_count = [0]

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
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("boom on first call")
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="recovered")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        # First prompt: the run raises; prompt() may not resolve cleanly
        # because the kernel never emits agent_end on a stream exception
        # (see loop.py — exception propagates from run). The driver catches,
        # logs, and continues. Park the first waiter and let it die when we
        # shutdown later; for now we just verify the SECOND prompt still
        # drives a fresh round.
        import asyncio as _aio

        first = _aio.create_task(session.prompt("first"))
        # Give the driver a turn to attempt + fail the run.
        await _aio.sleep(0.05)
        # Drain the failed first prompt's waiter by cancelling it (we don't
        # care about its return value — the point is the driver survived).
        # CancelledError is BaseException in 3.11+, so catch it explicitly.
        first.cancel()
        try:
            await first
        except (_aio.CancelledError, Exception):  # noqa: BLE001
            pass

        # Second prompt drives a fresh run; call_count[0] climbs to 2.
        messages = await session.prompt("second")
        assert call_count[0] >= 2
        assert "recovered" in _texts(messages, "assistant")
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_prompt_returns_message_list_with_originating_message(
    tmp_path: Path,
) -> None:
    """``prompt(text)`` pushes + awaits agent_end + returns the live message
    list including the originating user message AND the agent's reply (the
    sugar contract over the inbox-driver model)."""
    module_name = f"tests.unit._prompt_sugar_{id(tmp_path)}"
    try:
        session = await _make_session(tmp_path, module_name, _stream_text("hello back"))
        messages = await session.prompt("hello agent")
        user_texts = _texts(messages, "user")
        assistant_texts = _texts(messages, "assistant")
        assert "hello agent" in user_texts
        assert "hello back" in assistant_texts
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_interrupt_aborts_then_next_push_runs_with_preserved_context(
    tmp_path: Path,
) -> None:
    """``session.interrupt()`` MUST abort an in-flight run; the driver
    clears its signal afterwards and a fresh push drives a new run with the
    prior context intact (the session log is untouched)."""
    import asyncio as _aio

    module_name = f"tests.unit._interrupt_{id(tmp_path)}"

    long_tool_running = _aio.Event()
    aborts_seen: list[bool] = []

    async def stream_fn(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[Any]:
        del model, tools, system, thinking
        # First call: a long-running fake "tool" — we simulate the same
        # cooperative-abort shape an asyncio.Event-aware tool would have by
        # awaiting the signal directly INSIDE the stream coroutine. The
        # kernel's signal is threaded through stream_fn's ``signal`` arg
        # (loop.py:512 passes ``signal=signal``), so awaiting it here is the
        # idiomatic way to fake a long tool without registering one.
        joined_user = " | ".join(_texts(messages, "user"))
        if "second please" not in joined_user:
            long_tool_running.set()
            # Wait for either the signal (interrupt) or 5s timeout.
            assert signal is not None
            try:
                await _aio.wait_for(signal.wait(), timeout=5.0)
                aborts_seen.append(True)
            except _aio.TimeoutError:
                aborts_seen.append(False)
            # Yield a partial reply so the loop has SOMETHING to terminate
            # on — though the kernel's signal check at the top of the next
            # turn fires SignalAborted anyway.
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="interrupted-reply")],
                    timestamp=0.0,
                    stop_reason="end_turn",
                )
            )
            return
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="after-interrupt")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        prompt_task = _aio.create_task(session.prompt("start long task"))
        # Wait for the fake tool to start, then interrupt.
        await _aio.wait_for(long_tool_running.wait(), timeout=2.0)
        session.interrupt()
        await _aio.wait_for(prompt_task, timeout=2.0)
        assert aborts_seen == [True]

        # Context is preserved: the originating user message is on the log.
        first_log = session.session_manager.get_messages()
        assert "start long task" in _texts(first_log, "user")

        # Driver cleared the signal; a fresh prompt drives a new round
        # successfully and sees the preserved prior context.
        messages = await session.prompt("second please")
        user_texts = _texts(messages, "user")
        assert "start long task" in user_texts  # preserved
        assert "second please" in user_texts
        assert "after-interrupt" in _texts(messages, "assistant")
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)
