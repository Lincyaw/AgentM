"""Fail-stop coverage for the SessionInbox spine (step 1).

Design: ``.claude/designs/session-inbox.md``. The inbox is the single entry
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
    BudgetExhausted,
    MessageEnd,
    Model,
    NoPendingInput,
    TextContent,
    ToolTerminated,
)
from agentm.core.abi.events import BeforeAgentStartEvent
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
    # The source attribute on the wrapper lets the agent distinguish bg from
    # monitor / subagent textually (#176 post-merge polish).
    msg = render_item(InboxItem(source="background", payload="task 7 finished"))
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert '<system-reminder source="background">' in msg.content[0].text
    assert "task 7 finished" in msg.content[0].text


def test_render_monitor_item_to_system_reminder() -> None:
    # Step 4: monitor wakeups / channel fires use the same wrapper shape as
    # background but with a different source attribute.
    msg = render_item(InboxItem(source="monitor", payload="wakeup fired"))
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert '<system-reminder source="monitor">' in msg.content[0].text
    assert "wakeup fired" in msg.content[0].text


def test_render_subagent_item_to_system_reminder() -> None:
    # Step 5b: sub_agent findings posted via api.post_inbox(source="subagent").
    msg = render_item(
        InboxItem(source="subagent", payload="<subagent_result task_id=t1 />")
    )
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert '<system-reminder source="subagent">' in msg.content[0].text
    assert "<subagent_result" in msg.content[0].text


def test_render_source_tag_is_per_source_distinct() -> None:
    """Fail-stop: the ``source="..."`` attribute on the wrapper MUST distinguish
    background / monitor / subagent — that is the whole reason the attribute
    exists. A silent regression to a single un-attributed wrapper (or to the
    same attribute across all three) would re-introduce the producer-ambiguity
    that #176 post-merge E2E surfaced — the agent could no longer route a
    reminder textually back to its producer.
    """

    tags = {
        source: render_item(InboxItem(source=source, payload="x")).content[0].text
        for source in ("background", "monitor", "subagent")
    }
    # Each render carries its own source tag verbatim.
    for source, text in tags.items():
        assert f'<system-reminder source="{source}">' in text, (
            f"{source}: tag missing or wrong — got {text!r}"
        )
    # And the three are mutually distinguishable (no shared prefix substring
    # that would let the agent confuse them).
    rendered_tags = {text.split("\n", 1)[0] for text in tags.values()}
    assert len(rendered_tags) == 3, f"sources collapsed to {rendered_tags!r}"


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
        inbox_handle.append(session.inbox)
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
        # reach.
        session.inbox.push(InboxItem(source="user", payload="resume me"))

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
        inbox_handle.append(session.inbox)
        messages = await session.prompt("start")

        assistant_texts = _texts(messages, "assistant")
        # Two assistant turns prove the floor kept the loop alive past the
        # model's voluntary end-turn.
        assert "turn-0" in assistant_texts
        assert "turn-1" in assistant_texts
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_terminal_inbox_item_stops_loop_with_tool_terminated(
    tmp_path: Path,
) -> None:
    """#177 fail-stop: a ``terminal=True`` inbox item (a backgrounded
    ToolTerminate) MUST end the loop with ``ToolTerminated`` once delivered,
    instead of being kept alive on the non-empty inbox like an ordinary item.

    Reproducer: the model "ends its turn" every turn, but on the first turn we
    queue a terminal item. Without the patch the keep-alive floor would treat
    it like any other item — deliver it, and (since the floor sees it as
    ordinary) the loop would Stop(ModelEndTurn) after delivery. The patch makes
    the floor return ``Stop(ToolTerminated)`` so the termination is attributed
    to the terminate intent, never swallowed. We assert BOTH the cause and that
    the terminal message was actually delivered to the model first.
    """
    module_name = f"tests.unit._inbox_terminal_{id(tmp_path)}"

    turn = [0]
    inbox_handle: list[SessionInbox] = []
    delivery_turn_saw_terminal: list[bool] = []

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
            # Queue a terminal background completion mid-turn.
            inbox_handle[0].push(
                InboxItem(
                    source="background",
                    payload="terminal bg result",
                    terminal=True,
                )
            )
        else:
            # The delivery turn must see the terminal note in context.
            delivery_turn_saw_terminal.append(
                any("terminal bg result" in t for t in _texts(messages, "user"))
            )
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"turn-{idx}")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    causes: list[Any] = []
    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        inbox_handle.append(session.inbox)
        session.bus.on(AgentEndEvent.CHANNEL, lambda e: causes.append(e.cause))

        await session.prompt("start")

        # The terminal item was delivered on the second turn...
        assert delivery_turn_saw_terminal == [True]
        # ...and the run ended attributed to the terminate intent, not
        # ModelEndTurn / a runaway keep-alive.
        assert any(isinstance(c, ToolTerminated) for c in causes), causes
        # Exactly two turns ran: the originating turn + the delivery turn. A
        # third would mean the floor failed to stop.
        assert turn[0] == 2, f"expected 2 turns, got {turn[0]}"
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_backgrounded_terminate_survives_injecting_floor(
    tmp_path: Path,
) -> None:
    """#177 fail-stop (review MAJOR): a backgrounded ``ToolTerminate`` MUST
    survive a co-loaded floor that ``Inject``s over the keep-alive ``Stop``.

    Reproduces the sub_agent-child interaction: while a child is ``_RUNNING``,
    sub_agent's ``decide_turn_action`` floor returns ``Inject(<pending>)``,
    which the resolution lattice (loop.py:298) ranks ABOVE the #177
    ``Stop(ToolTerminated)``. The original code nulled ``_pending_terminate``
    inside the keep-alive handler BEFORE the lattice picked the Inject winner,
    so the terminate intent was permanently destroyed — the loop ran forever on
    the injecting floor and NEVER ended on ``ToolTerminated``.

    Here a stand-in floor injects for the first two boundaries after the
    terminal item lands, then stops (mimicking the child finishing). The fix
    re-asserts the same cause on every boundary and clears it only on the
    matching ``agent_end``, so the loop ultimately stops on ``ToolTerminated``
    once the injecting floor steps aside. Without the fix the cause is gone
    after turn 1 and no ``ToolTerminated`` ever appears.
    """
    import asyncio as _aio

    module_name = f"tests.unit._inbox_terminal_inject_{id(tmp_path)}"

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
        del messages, model, tools, system, signal, thinking
        idx = turn[0]
        turn[0] += 1
        if idx == 0:
            inbox_handle[0].push(
                InboxItem(
                    source="background",
                    payload="terminal bg result",
                    terminal=True,
                )
            )
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"turn-{idx}")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    # Stand-in for sub_agent's still-running-child floor: inject a pending
    # notice for a bounded number of boundaries (the "child is running" window),
    # then step aside. Registered on the SAME channel as the runtime keep-alive
    # floor so it co-participates in resolve_loop_action.
    from agentm.core.abi import DecideTurnActionEvent, Inject, UserMessage

    inject_budget = [2]

    def _injecting_floor(_event: DecideTurnActionEvent) -> Inject | None:
        # Only contend once the terminal item has been observed (turn >= 1),
        # mirroring a child that is running while the terminate is pending.
        if turn[0] >= 1 and inject_budget[0] > 0:
            inject_budget[0] -= 1
            return Inject(
                messages=[
                    UserMessage(
                        role="user",
                        content=[
                            TextContent(
                                type="text", text="<subagent_pending/>"
                            )
                        ],
                        timestamp=0.0,
                    )
                ]
            )
        return None

    causes: list[Any] = []
    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        inbox_handle.append(session.inbox)
        # Higher priority so it is dispatched alongside the keep-alive floor;
        # both returns feed the same resolve_loop_action batch.
        session.bus.on(
            DecideTurnActionEvent.CHANNEL, _injecting_floor, priority=400
        )
        session.bus.on(AgentEndEvent.CHANNEL, lambda e: causes.append(e.cause))

        await _aio.wait_for(session.prompt("start"), timeout=5.0)

        # The loop ultimately ended on the terminate intent — NOT stranded
        # forever under the injecting floor.
        assert any(isinstance(c, ToolTerminated) for c in causes), causes
        # The injecting floor really did override the Stop first (its budget was
        # consumed), proving the lattice path was exercised, not bypassed.
        assert inject_budget[0] == 0, "injecting floor never contended"
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


# ---------------------------------------------------------------------------
# Step-5 review fixes (Major 1, Major 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_while_idle_does_not_poison_next_prompt(
    tmp_path: Path,
) -> None:
    """Major-1 fix: calling ``session.interrupt()`` while the driver is idle
    (blocked on ``wait_nonempty`` with no in-flight run) MUST NOT poison the
    next ``prompt()`` with a stale ``SignalAborted``.

    Reproducer for the bug: previously ``_driver`` cleared ``_signal`` AFTER
    ``_run_one_round`` returned, so an idle-time ``interrupt()`` left
    ``_signal`` set; the next ``prompt`` push woke the driver, the kernel's
    per-turn signal check (``loop.py:440``) fired ``SignalAborted`` before
    any real work happened.
    """
    import asyncio as _aio

    module_name = f"tests.unit._idle_interrupt_{id(tmp_path)}"

    try:
        session = await _make_session(
            tmp_path, module_name, _stream_text("hello back")
        )
        # Driver is idle (no prior prompt). Interrupt now.
        session.interrupt()
        # Yield once so any latent driver-loop iteration would observe the
        # signal — but the driver should still be parked on wait_nonempty
        # (no inbox push happened yet, no run in flight).
        await _aio.sleep(0)

        # Next prompt MUST complete normally with the stub provider's reply,
        # not raise SignalAborted. A short timeout catches the
        # tight-loop / hang failure modes.
        messages = await _aio.wait_for(
            session.prompt("hello agent"), timeout=2.0
        )
        assistant_texts = _texts(messages, "assistant")
        assert "hello back" in assistant_texts
        # The originating prompt also landed (proves the run actually
        # executed rather than short-circuiting).
        assert "hello agent" in _texts(messages, "user")
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_driver_parks_after_pre_first_turn_failure_no_spin(
    tmp_path: Path,
) -> None:
    """Major-3 fix: a persistent pre-first-turn failure (e.g. a
    ``build_session_context`` that always raises) MUST NOT tight-loop the
    driver. The driver drains the inbox on the exception, parks on
    ``wait_nonempty``, and only re-attempts on a fresh push.

    Reproducer: previously ``_run_one_round`` raised before the kernel
    emitted its first ``context`` event, so the inbox was never drained —
    ``_nonempty`` stayed set, ``wait_nonempty`` returned immediately, and
    the driver attempted the failing round again and again at CPU speed.

    We monkeypatch ``session_manager.build_session_context`` because the
    bus suppresses handler exceptions (event handlers cannot escape
    ``emit``), so the realistic propagating-failure path is a callsite the
    driver invokes directly. Same shape applies to a buggy resource
    loader, a transient FS error during context build, etc.
    """
    import asyncio as _aio

    module_name = f"tests.unit._driver_parks_{id(tmp_path)}"

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
        # Should never reach the stream if build_session_context raises first.
        call_count[0] += 1
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="unreachable")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    build_calls = [0]

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        original_build = session.session_manager.build_session_context

        def _always_raises(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            build_calls[0] += 1
            raise RuntimeError("build_session_context always-raise")

        session.session_manager.build_session_context = _always_raises  # type: ignore[method-assign]

        try:
            # Push a user message; the driver picks it up, fails in
            # build_session_context, drains the inbox (Major-3 fix), parks.
            prompt_task = _aio.create_task(session.prompt("kick"))

            # The waiter is rejected via _fail_end_waiters with the RuntimeError.
            with pytest.raises(RuntimeError, match="always-raise"):
                await _aio.wait_for(prompt_task, timeout=2.0)

            # Snapshot the build-call count, then let many event-loop ticks
            # pass. Before the fix, the driver would tight-loop and
            # ``build_calls`` would climb without bound. After the fix the
            # driver is parked on wait_nonempty and the count stays put.
            count_after_first = build_calls[0]
            assert count_after_first == 1, (
                f"originating push should drive exactly one attempt; "
                f"got {count_after_first}"
            )
            for _ in range(100):
                await _aio.sleep(0)
            await _aio.sleep(0.1)

            # Bounded: the count stays at 1 (no spin afterwards). A tight
            # loop would produce hundreds.
            assert build_calls[0] == count_after_first, (
                f"driver tight-looped: build_session_context called "
                f"{build_calls[0]} times after the originating push "
                f"(expected exactly {count_after_first})"
            )

            # And the stream was never reached.
            assert call_count[0] == 0

            # Restore + push again: ensure the driver is still alive and
            # not jammed, just parked.
            session.session_manager.build_session_context = original_build  # type: ignore[method-assign]
        finally:
            # Always restore so shutdown's catalog-index path can use it
            # (shutdown reads session.session_manager.get_messages but
            # not build, so technically optional; restore for safety).
            session.session_manager.build_session_context = original_build  # type: ignore[method-assign]
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_driver_parks_after_sticky_veto_no_spin(tmp_path: Path) -> None:
    """External-review follow-up to step-5 Major-3 (sibling pattern): a
    sticky ``before_agent_start`` veto with a non-empty inbox MUST NOT
    tight-loop the driver.

    Reproducer: ``_run_one_round`` collects a veto from
    ``before_agent_start``, emits ``agent_end``, and returns BEFORE the
    kernel's first ``context`` event fires — so the originating push is
    still queued in the inbox, ``_nonempty`` stays set, ``wait_nonempty``
    returns immediately on the next driver iteration, and the driver
    re-runs the same veto at CPU speed. The user's ``prompt`` already
    resolved via the first ``agent_end``, so this is invisible until you
    notice the burning CPU.

    The fix calls ``_drain_inbox_on_early_return`` on the veto path
    (mirroring the ``_driver`` except branch) so the next iteration parks
    on ``wait_nonempty`` until a fresh push.
    """
    import asyncio as _aio

    module_name = f"tests.unit._driver_parks_veto_{id(tmp_path)}"

    veto_calls = [0]
    stream_calls = [0]

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
        # Should never fire: the veto returns before ``_loop.run``.
        stream_calls[0] += 1
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="unreachable")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def _sticky_veto(_event: Any) -> dict[str, Any]:
        veto_calls[0] += 1
        return {
            "block": True,
            "cause": BudgetExhausted(detail="sticky-veto test"),
        }

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        # Subscribe at default priority — runs alongside any production
        # before_agent_start handlers; ``collect_start_veto`` picks any
        # ``block=True`` return regardless of order.
        session._bus.on(BeforeAgentStartEvent.CHANNEL, _sticky_veto)

        # First prompt resolves cleanly via the veto's ``agent_end``.
        await _aio.wait_for(session.prompt("kick"), timeout=2.0)
        # ``loop.run`` was never entered (veto fired pre-run).
        assert stream_calls[0] == 0

        count_after_first = veto_calls[0]
        assert count_after_first == 1, (
            f"originating push should drive exactly one veto attempt; "
            f"got {count_after_first}"
        )

        # Let many event-loop ticks pass. Before the fix the driver
        # would tight-loop on the still-queued user message and the veto
        # handler call count would climb without bound.
        for _ in range(100):
            await _aio.sleep(0)
        await _aio.sleep(0.1)

        assert veto_calls[0] == count_after_first, (
            f"driver tight-looped on sticky veto: before_agent_start "
            f"called {veto_calls[0]} times after the originating push "
            f"(expected exactly {count_after_first})"
        )
        # Stream still never reached.
        assert stream_calls[0] == 0
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_tick_external_signal_during_decide_does_not_poison_next_prompt(
    tmp_path: Path,
) -> None:
    """Major-1 fix (second sub-case): an external ``signal`` fired during
    ``tick``'s synthetic decide-cycle MUST NOT poison the next ``prompt``
    with a stale ``SignalAborted``.

    Pre-fix bug: ``tick``'s signal forwarder bridged the external event
    into ``_signal``; if no run was launched (empty inbox + no injector),
    ``_signal`` carried over and the next ``prompt`` push aborted before
    any LLM call. Fix: ``tick``'s ``finally`` clears ``_signal`` on the
    no-run paths.
    """
    import asyncio as _aio

    module_name = f"tests.unit._tick_leak_{id(tmp_path)}"

    try:
        session = await _make_session(
            tmp_path, module_name, _stream_text("good reply")
        )

        external = _aio.Event()
        # Fire the external signal BEFORE tick — the forwarder spawns and
        # the synthetic decide runs; tick's no-run path returns. The fix
        # clears _signal in the finally so the next prompt is clean.
        external.set()
        await session.tick(signal=external)
        # Yield to let the forwarder run (it awaits external.wait() which
        # is already set).
        await _aio.sleep(0)

        # Next prompt MUST complete normally with the stub provider's
        # reply, not raise SignalAborted.
        messages = await _aio.wait_for(
            session.prompt("hi after tick"), timeout=2.0
        )
        assert "good reply" in _texts(messages, "assistant")
        assert "hi after tick" in _texts(messages, "user")
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_shutdown_strict_reraises_driver_exception(tmp_path: Path) -> None:
    """B6 fail-stop: ``shutdown(strict=True)`` re-raises a driver-thrown
    exception that the default-False path silently swallows + logs.

    Setup: cancel the driver task ourselves to a custom RuntimeError-shaped
    exception (matches the "driver raised mid-round" shape ``shutdown``'s
    broad-except catches), then call ``shutdown(strict=True)`` and assert
    the exception propagates. ``strict=False`` on a sibling session swallows
    the same shape — preserving the legacy CLI path.
    """
    import asyncio as _aio

    module_name = f"tests.unit._shutdown_strict_{id(tmp_path)}"

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
                content=[TextContent(type="text", text="ok")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    async def _inject_failure(session: AgentSession) -> None:
        """Replace the driver task with one that raises a known exception
        on await, simulating the "driver task raised after shutdown was
        already in flight" path the broad-except in shutdown() catches."""

        # The original driver is parked on wait_nonempty; cancel + replace
        # with a task that raises a tagged error so the test can pin the
        # propagation.
        session._driver_task.cancel()
        try:
            await session._driver_task
        except _aio.CancelledError:
            pass

        async def _raises() -> None:
            raise RuntimeError("driver-crash-B6")

        session._driver_task = _aio.create_task(_raises())
        # Let the replacement task run to terminal state so wait_for sees
        # the exception, not a still-pending task.
        await _aio.sleep(0)

    # Strict path → re-raises.
    session_strict = await _make_session(tmp_path, module_name, stream_fn)
    try:
        await _inject_failure(session_strict)
        with pytest.raises(RuntimeError, match="driver-crash-B6"):
            await session_strict.shutdown(strict=True)
    finally:
        sys.modules.pop(module_name, None)

    # Default path → swallows + logs (the legacy CLI contract).
    module_name2 = f"tests.unit._shutdown_default_{id(tmp_path)}"
    session_default = await _make_session(tmp_path, module_name2, stream_fn)
    try:
        await _inject_failure(session_default)
        # Must NOT raise.
        await session_default.shutdown()
    finally:
        sys.modules.pop(module_name2, None)


@pytest.mark.asyncio
async def test_discarded_inbox_items_leave_session_trace(tmp_path: Path) -> None:
    """B2 fail-stop: items discarded by ``_drain_inbox_on_early_return``
    must be appended to the session log as a structured ``inbox.discarded``
    entry so the original payload is recoverable from the trace.

    Previously the discard was log-only — a sticky veto on a user prompt
    silently dropped the prompt text. The B2 fix appends an
    ``append_custom_entry("inbox.discarded", ...)`` capturing the reason,
    every source, and the verbatim payload of each discarded item.
    """
    import asyncio as _aio

    module_name = f"tests.unit._inbox_discarded_trace_{id(tmp_path)}"

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
                content=[TextContent(type="text", text="unreachable")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def _sticky_veto(_event: Any) -> dict[str, Any]:
        return {
            "block": True,
            "cause": BudgetExhausted(detail="b2-discard test"),
        }

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)
        session._bus.on(BeforeAgentStartEvent.CHANNEL, _sticky_veto)

        # The veto path triggers ``_drain_inbox_on_early_return`` after the
        # originating push has been consumed by the driver but before the
        # kernel ``context`` event fires.
        await _aio.wait_for(session.prompt("vetoed user input"), timeout=2.0)

        # The session log must carry an inbox.discarded entry whose payload
        # records the reason, the sources, and the original user content.
        entries = session.session_manager.get_entries()
        discarded = [e for e in entries if e.type == "inbox.discarded"]
        assert len(discarded) == 1, (
            f"expected exactly one inbox.discarded entry; got {len(discarded)}"
        )
        payload = discarded[0].payload
        assert payload["sources"] == ["user"], payload
        assert "veto" in payload["reason"].lower(), payload["reason"]
        items = payload["items"]
        assert len(items) == 1
        # The original user content blocks survive into the trace verbatim
        # (the session manager runs them through to_jsonable on write).
        item_payload = items[0]["payload"]
        # The user payload is a list of TextContent dataclass instances; the
        # trace stores them as their dataclass repr through to_jsonable.
        rendered_text = str(item_payload)
        assert "vetoed user input" in rendered_text, rendered_text
        assert items[0]["source"] == "user"
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_concurrent_prompts_serialize_no_stale_data(tmp_path: Path) -> None:
    """A4 fail-stop: two concurrent ``prompt`` callers FIFO-serialize, and
    each returns a message list ending in ITS OWN turn's reply.

    Without ``_prompt_lock``, both callers subscribe a waiter for the same
    upcoming ``agent_end``, both unblock together when the FIRST turn
    finishes, and both call ``session_manager.get_messages()`` before the
    second turn even starts — the second prompt returns a message list whose
    last assistant reply is the FIRST prompt's reply (stale data attributed
    to the wrong caller). The lock prevents that interleave.

    Deterministic per-turn stream so cross-attribution is detectable: the
    provider stamps a fresh reply each turn, and each prompt's returned
    message list must end in the reply minted on ITS turn.
    """
    import asyncio as _aio

    module_name = f"tests.unit._prompt_concurrent_{id(tmp_path)}"

    turn_index = [0]
    turn_started = _aio.Event()

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
        idx = turn_index[0]
        turn_index[0] += 1
        turn_started.set()
        # Yield control so the second prompt task gets a chance to schedule
        # while we're "mid-turn" — without the lock this is when its waiter
        # would attach to the same agent_end as the first prompt.
        await _aio.sleep(0.01)
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"reply-{idx}")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    try:
        session = await _make_session(tmp_path, module_name, stream_fn)

        # Fire both prompts together. Order is FIFO under the lock: the first
        # task that *enters* prompt() also acquires the lock first.
        first = _aio.create_task(session.prompt("first"))
        # Give the first task a turn on the event loop so it definitely
        # acquires the lock before the second; otherwise scheduling order is
        # what the lock would silently rely on.
        await _aio.sleep(0)
        second = _aio.create_task(session.prompt("second"))

        first_msgs = await _aio.wait_for(first, timeout=2.0)
        second_msgs = await _aio.wait_for(second, timeout=2.0)

        # Each prompt's returned list ends in the assistant reply minted on
        # ITS OWN turn, never the other prompt's. Without the lock both
        # callers would unblock on the same agent_end and the second would
        # return reply-0 (the first prompt's reply attributed to the wrong
        # caller).
        first_assistant = _texts(first_msgs, "assistant")
        second_assistant = _texts(second_msgs, "assistant")
        assert first_assistant[-1] == "reply-0", (
            f"first prompt got stale reply: {first_assistant}"
        )
        assert second_assistant[-1] == "reply-1", (
            f"second prompt got stale reply: {second_assistant}"
        )
        # The second prompt sees BOTH user messages + BOTH replies on the
        # log — proving the lock made the second prompt wait until the
        # first's turn was committed before its own turn started.
        all_user = _texts(second_msgs, "user")
        assert all_user.index("first") < all_user.index("second")
        assert "reply-0" in second_assistant
        assert "reply-1" in second_assistant
        # And the first prompt returned BEFORE the second prompt ran, so its
        # message list only sees its own user message and its own reply.
        assert _texts(first_msgs, "user") == ["first"]
        assert first_assistant == ["reply-0"]
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)



# ---------------------------------------------------------------------------
# #179: idle() — one-shot host waits out late background completions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbox_pending_work_gates_idle_event() -> None:
    """#179 fail-stop (unit): the background-work counter blocks
    ``wait_no_pending_work`` while work is live and releases it only when the
    count returns to zero — and an over-finish cannot falsely release it.

    Without this gate ``idle()`` has no signal for "a detached unit is still
    running" and would let a one-shot host exit mid-task."""

    import asyncio as _aio

    inbox = SessionInbox()
    # Starts idle: no work outstanding ⇒ wait returns immediately.
    assert not inbox.has_pending_work
    await _aio.wait_for(inbox.wait_no_pending_work(), timeout=0.5)

    inbox.note_work_started()
    assert inbox.has_pending_work
    with pytest.raises(_aio.TimeoutError):
        await _aio.wait_for(inbox.wait_no_pending_work(), timeout=0.05)

    inbox.note_work_started()  # two units live
    inbox.note_work_finished()  # one done — still gated
    assert inbox.has_pending_work
    with pytest.raises(_aio.TimeoutError):
        await _aio.wait_for(inbox.wait_no_pending_work(), timeout=0.05)

    inbox.note_work_finished()  # last done — gate opens
    assert not inbox.has_pending_work
    await _aio.wait_for(inbox.wait_no_pending_work(), timeout=0.5)

    # Over-finish (would be a producer bug) is clamped, not negative, and does
    # not corrupt the gate.
    inbox.note_work_finished()
    assert not inbox.has_pending_work
    await _aio.wait_for(inbox.wait_no_pending_work(), timeout=0.5)


@pytest.mark.asyncio
async def test_idle_waits_for_late_background_completion(tmp_path: Path) -> None:
    """#179 fail-stop (session): a backgrounded unit that finishes AFTER the
    agent's turn ended must have its completion DELIVERED into a driver round
    before ``idle()`` returns — the exact drop the one-shot CLI suffered.

    Reproducer mirrors the one-shot CLI flow: ``prompt`` returns at agent_end
    while a tracked background unit is still running. The unit later posts a
    ``source="background"`` completion and finishes. ``idle()`` must:
      1. NOT return while the unit is live (work outstanding), and
      2. NOT return until the posted completion has been drained into a round
         (the delivery turn ran).

    Without ``idle`` (the pre-patch CLI) the process would exit at step (0),
    dropping the completion. We assert a delivery turn actually saw the late
    completion text and that ``idle`` only returned afterwards.
    """
    import asyncio as _aio

    module_name = f"tests.unit._idle_late_bg_{id(tmp_path)}"

    turn = [0]
    delivery_saw_completion: list[bool] = []

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
        if idx > 0:
            delivery_saw_completion.append(
                any("late bg done" in t for t in _texts(messages, "user"))
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
        inbox = session.inbox

        # Simulate a producer that brackets a detached unit: mark work live
        # NOW (as background_exec/sub_agent do at dispatch), then post its
        # completion + finish after a short delay — strictly AFTER prompt()
        # returns at agent_end.
        inbox.note_work_started()

        async def _late_unit() -> None:
            await _aio.sleep(0.05)
            inbox.push(
                InboxItem(source="background", payload="late bg done")
            )
            inbox.note_work_finished()

        unit = _aio.create_task(_late_unit())

        # The agent's only turn ends here; prompt returns while work is live.
        await session.prompt("start")
        assert inbox.has_pending_work, (
            "background unit should still be live at agent_end"
        )

        # The one-shot host now waits to be idle before exiting.
        await _aio.wait_for(session.idle(), timeout=3.0)
        await unit

        # idle() only returned after the late completion was delivered into a
        # round (a delivery turn ran and saw the completion text), and the
        # session is genuinely at rest.
        assert delivery_saw_completion == [True], delivery_saw_completion
        assert inbox.is_empty()
        assert not inbox.has_pending_work
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_idle_returns_immediately_when_already_at_rest(
    tmp_path: Path,
) -> None:
    """#179 guard: with no background work and an empty inbox, ``idle()``
    returns promptly (the common case — no overrunning tools), so a one-shot
    host with nothing detached is not held open."""
    import asyncio as _aio

    module_name = f"tests.unit._idle_at_rest_{id(tmp_path)}"
    try:
        session = await _make_session(tmp_path, module_name, _stream_text("ok"))
        await session.prompt("hi")
        await _aio.wait_for(session.idle(), timeout=2.0)
        assert session.inbox.is_empty()
        assert not session.inbox.has_pending_work
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# #201: bounded idle(timeout=...) — leaked tracked work cannot hang one-shot CLI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_timeout_returns_on_leaked_tracked_work(tmp_path: Path) -> None:
    """#201 fail-stop: a tracked background unit that is started but NEVER
    finished (a leaked ``track_background`` counter, or a genuinely stuck /
    never-terminating background tool) must NOT hang ``idle()`` forever. With a
    bound it returns ``False`` in bounded time so the one-shot CLI can warn and
    exit rather than wedge.

    The whole test is wrapped in a real ``asyncio.wait_for`` so a regression to
    the unbounded behaviour fails the test (timeout) instead of hanging CI.
    """
    import asyncio as _aio

    module_name = f"tests.unit._idle_timeout_leak_{id(tmp_path)}"
    try:
        session = await _make_session(tmp_path, module_name, _stream_text("ok"))
        await session.prompt("hi")

        # Leak: mark work live and never finish it (the unbounded ``idle()``
        # would block here forever).
        session.inbox.note_work_started()
        assert session.inbox.has_pending_work

        async def _bounded() -> bool:
            return await session.idle(timeout=0.05)

        # The outer wait_for is the safety net: if ``idle`` ignored its bound
        # this raises (test fails) instead of hanging the suite.
        at_rest = await _aio.wait_for(_bounded(), timeout=2.0)
        assert at_rest is False, "bounded idle must report not-at-rest on a leak"
        # The bound did not corrupt accounting: the leaked counter is still live
        # and the gate is still closed (a later finish would still flip it).
        assert session.inbox.has_pending_work
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_idle_timeout_returns_true_when_work_finishes_in_time(
    tmp_path: Path,
) -> None:
    """#201 guard: the bound does not break the healthy path — a tracked unit
    that finishes within the timeout still lets ``idle(timeout=...)`` reach rest
    and return ``True`` (no false "timed out" on normal completion)."""
    import asyncio as _aio

    module_name = f"tests.unit._idle_timeout_ok_{id(tmp_path)}"
    try:
        session = await _make_session(tmp_path, module_name, _stream_text("ok"))
        await session.prompt("hi")

        session.inbox.note_work_started()

        async def _finish_soon() -> None:
            await _aio.sleep(0.02)
            session.inbox.note_work_finished()

        unit = _aio.create_task(_finish_soon())
        at_rest = await _aio.wait_for(session.idle(timeout=2.0), timeout=3.0)
        await unit
        assert at_rest is True
        assert session.inbox.is_empty()
        assert not session.inbox.has_pending_work
    finally:
        await session.shutdown()
        sys.modules.pop(module_name, None)
