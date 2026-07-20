"""Integration tests for the v2 trajectory model.

Tests the full session driver loop with a mock LLM provider, verifying
turn lifecycle, tool execution, event bus dispatch, context policies,
store persistence, session graph, codec round-trips, and robustness
under failures.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    OpaqueThinkingBlock,
    TextContent,
    ToolCallBlock,
    UserMessage,
)
from agentm.core.abi.stream import MessageEnd, Model, TextDelta
from agentm.core.abi.tool import (
    FunctionTool,
    ToolContinue,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.bus import BusPriority, EventBus
from agentm.core.abi.codec import CodecRegistry, RawTrigger
from agentm.core.abi.context import (
    PolicyContext,
    build_context_sync,
    render_trigger,
)
from agentm.core.abi.events import (
    BeforeRunEvent,
    BeforeSendEvent,
    ContextEvent,
    DecideEvent,
    Inject,
    LoopAction,
    Step,
    Stop,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    TurnBeginEvent,
    TurnCommittedEvent,
)
from agentm.core.abi.termination import (
    BudgetExhausted,
    ModelEndTurn,
    ProviderRequestFailed,
    SignalAborted,
    ToolTerminated,
)
from agentm.core.abi.query import (
    ObservabilityQueryStore,
    SessionFilter,
    TrajectoryQueryStore,
)
from agentm.core.abi.resource import ResourceMutation, ResourceRef
from agentm.core.abi.roles import TRAJECTORY_QUERY_STORE_SERVICE
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import AgentSessionConfig
from agentm.core.abi.store import SessionMeta, TrajectoryCommit, TrajectoryStore
from agentm.core.abi.trajectory import Outcome, Round, TrajectoryHead, Turn, TurnMeta
from agentm.core.abi.trigger import (
    BackgroundCompletion,
    ContinueTrigger,
    Injection,
    MonitorFire,
    SubagentResult,
    UserInput,
)
from agentm.core.runtime.execution import Execution, StateError
from agentm.core.runtime.session import Session, SessionRuntimeConfig
from agentm.core.runtime.session_factory import SessionBuildConfig, create_session
from agentm.core.runtime.session_meta import ResumeIdentityError
from agentm.core.runtime.stores.query import TrajectoryStoreQueryAdapter
from agentm.storage.trajectory.memory import InMemoryTrajectoryStore
from agentm.core.runtime.trajectory import Trajectory
from agentm.storage.trajectory import JsonlTrajectoryStore
from agentm.core.runtime.trigger_queue import TriggerTerminated
from agentm.core.runtime.tree import InMemorySessionGraph
from agentm.core.runtime.trigger_queue import QueueClosed, TriggerQueue

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _memory_store(store: InMemoryTrajectoryStore) -> TrajectoryStore:
    return store


def _empty_head(session_id: str) -> TrajectoryHead:
    return TrajectoryHead(session_id=session_id, root_session_id=session_id)


class MockStreamFn:
    def __init__(self) -> None:
        self._responses: list[AssistantMessage] = []
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, *responses: AssistantMessage) -> None:
        self._responses.extend(responses)

    @property
    def call_count(self) -> int:
        return self._call_count

    async def __call__(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any,
        system: Any = None,
        signal: Any = None,
        thinking: Any = "off",
    ) -> AsyncIterator[TextDelta | MessageEnd]:
        self._call_count += 1
        self.calls.append({"messages": list(messages)})
        if not self._responses:
            raise RuntimeError("MockStreamFn: no more responses")
        resp = self._responses.pop(0)
        for block in resp.content:
            if isinstance(block, TextContent):
                yield TextDelta(text=block.text)
        yield MessageEnd(message=resp)


class FailingStreamFn:
    """A stream fn that raises for the first N calls, then delegates."""

    def __init__(self, fail_count: int, delegate: MockStreamFn) -> None:
        self._fail_count = fail_count
        self._delegate = delegate
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def __call__(
        self, *, messages: Any, model: Any, tools: Any, system: Any = None,
        signal: Any = None, thinking: Any = "off",
    ) -> AsyncIterator[TextDelta | MessageEnd]:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ConnectionError(f"stream failure #{self._call_count}")
        async for ev in self._delegate(
            messages=messages, model=model, tools=tools,
            system=system, signal=signal, thinking=thinking,
        ):
            yield ev


def text_response(text: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=time.time(),
        stop_reason="end_turn",
    )


def tool_call_response(name: str, call_id: str, args: dict[str, Any]) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[ToolCallBlock(type="tool_call", id=call_id, name=name, arguments=args)],
        timestamp=time.time(),
        stop_reason="tool_use",
    )


def empty_response() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        timestamp=time.time(),
        stop_reason="end_turn",
    )


def make_model() -> Model:
    return Model(id="mock-model", provider="mock", context_window=128000, max_output_tokens=4096)


class EventCollector:
    def __init__(self, bus: EventBus) -> None:
        self.events: dict[str, list[Any]] = {}
        channels = [
            TurnBeginEvent.CHANNEL, TurnCommittedEvent.CHANNEL,
            ContextEvent.CHANNEL, BeforeSendEvent.CHANNEL,
            StreamDeltaEvent.CHANNEL, ToolCallEvent.CHANNEL,
            ToolResultEvent.CHANNEL, DecideEvent.CHANNEL,
            BeforeRunEvent.CHANNEL, ToolErrorEvent.CHANNEL,
        ]
        for ch in channels:
            self.events[ch] = []
            bus.on(ch, self._handler(ch))

    def _handler(self, ch: str) -> Any:
        def h(event: Any) -> None:
            self.events[ch].append(event)
        return h


async def _wait_turn(session: Session, *, timeout: float = 5.0) -> None:
    """Wait for the current driver run to commit a turn and stop."""
    ev = asyncio.Event()

    def _on_commit(_: Any) -> None:
        ev.set()

    unsub = session.bus.on(TurnCommittedEvent.CHANNEL, _on_commit)
    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    finally:
        unsub()


def _make_add_tool() -> FunctionTool:
    async def add_fn(args: dict[str, Any]) -> ToolResult:
        a = args.get("a", 0)
        b = args.get("b", 0)
        return ToolResult(content=[TextContent(type="text", text=str(a + b))])
    return FunctionTool(
        name="add", description="add two numbers",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        fn=add_fn,
    )


def _make_finish_tool() -> FunctionTool:
    async def finish_fn(args: dict[str, Any]) -> ToolTerminate:
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="done")]),
            reason="test:done",
        )
    return FunctionTool(
        name="finish", description="finish the session",
        parameters={"type": "object"},
        fn=finish_fn,
    )


# ---------------------------------------------------------------------------
# GROUP 1: Core Turn Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_text() -> None:
    mock = MockStreamFn()
    mock.enqueue(text_response("hello"))
    session = Session(
        SessionRuntimeConfig(stream_fn=mock, model=make_model(), system="test")
    )
    session.start()
    await session.prompt("hi")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert turn.id
    assert turn.timestamp > 0
    assert len(turn.rounds) == 1
    assert turn.outcome.cause is not None
    assert isinstance(turn.outcome.cause, ModelEndTurn)
    assert isinstance(turn.trigger, UserInput)
    assert turn.meta.model_id == "mock-model"
    rnd = turn.rounds[0]
    assert isinstance(rnd.response.content[0], TextContent)
    assert rnd.response.content[0].text == "hello"


@pytest.mark.asyncio
async def test_multi_round_tool_call() -> None:
    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("add", "call-1", {"a": 2, "b": 3}),
        text_response("the answer is 5"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[_make_add_tool()],
    ))
    session.start()
    await session.prompt("add 2 and 3")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert len(turn.rounds) == 2
    rnd0 = turn.rounds[0]
    assert len(rnd0.tool_results) == 1
    tr = rnd0.tool_results[0]
    assert tr.call.name == "add"
    assert not tr.result.is_error
    assert any("5" in c.text for c in tr.result.content if isinstance(c, TextContent))
    rnd1 = turn.rounds[1]
    assert len(rnd1.tool_results) == 0


@pytest.mark.asyncio
async def test_tool_terminate() -> None:
    mock = MockStreamFn()
    mock.enqueue(tool_call_response("finish", "call-f", {}))
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[_make_finish_tool()],
    ))
    session.start()
    await session.prompt("finish")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert turn.outcome.cause is not None
    assert isinstance(turn.outcome.cause, ToolTerminated)
    assert turn.outcome.cause.tool_name == "finish"


@pytest.mark.asyncio
async def test_max_turns() -> None:
    mock = MockStreamFn()
    mock.enqueue(text_response("one"), text_response("two"), text_response("three"))
    session = Session(SessionRuntimeConfig(
        stream_fn=mock,
        model=make_model(),
        system="test",
        max_turns=2,
    ))
    session.start()
    await session.prompt("first")
    await _wait_turn(session)
    await session.shutdown()
    assert len(session.trajectory) <= 2


# ---------------------------------------------------------------------------
# GROUP 2: Atom Interaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_atom_services() -> None:
    counter = {"value": 0}

    async def increment_fn(args: dict[str, Any]) -> ToolResult:
        counter["value"] += 1
        return ToolResult(content=[TextContent(type="text", text=f"counter={counter['value']}")])

    async def report_fn(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text=f"report: {counter['value']}")])

    inc_tool = FunctionTool(
        name="increment", description="increment counter",
        parameters={"type": "object"}, fn=increment_fn,
    )
    rep_tool = FunctionTool(
        name="report", description="report counter",
        parameters={"type": "object"}, fn=report_fn,
    )

    mock = MockStreamFn()
    # Chain both tool calls in one turn: increment → report → text
    mock.enqueue(
        tool_call_response("increment", "c1", {}),
        tool_call_response("report", "c2", {}),
        text_response("all done"),
    )

    services = ServiceRegistry()
    services.register("counter", counter)

    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[inc_tool, rep_tool], services=services,
    ))
    session.start()
    await session.prompt("increment then report")
    await _wait_turn(session)
    await session.shutdown()

    assert counter["value"] == 1
    assert services.get("counter") is counter

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert len(turn.rounds) == 3
    # Round 1: report tool result should contain "1"
    tr_report = turn.rounds[1].tool_results[0]
    assert "1" in tr_report.result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_context_policy() -> None:
    bind_called = {"called": False, "session_id": ""}

    class MarkerPolicy:
        async def transform(
            self, messages: list[AgentMessage], turns: Sequence[Turn],
        ) -> list[AgentMessage]:
            marker = UserMessage(
                role="user",
                content=[TextContent(type="text", text="[MARKER]")],
                timestamp=0.0,
            )
            return [marker] + messages

        def bind(self, ctx: PolicyContext) -> None:
            bind_called["called"] = True
            bind_called["session_id"] = ctx.session_id

    captured_messages: list[Any] = []

    mock = MockStreamFn()
    mock.enqueue(text_response("ok"))

    policy = MarkerPolicy()
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        context_policies=[policy],  # type: ignore[list-item]
    ))

    session.bus.on(BeforeSendEvent.CHANNEL, lambda ev: captured_messages.extend(ev.messages))

    session.start()
    await session.prompt("hi")
    await _wait_turn(session)
    await session.shutdown()

    assert bind_called["called"]
    assert bind_called["session_id"] == session.id


# ---------------------------------------------------------------------------
# GROUP 3: Events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_coverage() -> None:
    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("add", "ec1", {"a": 1, "b": 2}),
        text_response("done"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[_make_add_tool()],
    ))
    collector = EventCollector(session.bus)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(collector.events[TurnBeginEvent.CHANNEL]) >= 1
    assert collector.events[TurnBeginEvent.CHANNEL][0].index == 0
    assert len(collector.events[TurnCommittedEvent.CHANNEL]) >= 1
    assert collector.events[TurnCommittedEvent.CHANNEL][0].turn is not None
    assert len(collector.events[ToolCallEvent.CHANNEL]) >= 1
    assert collector.events[ToolCallEvent.CHANNEL][0].tool_name == "add"
    assert len(collector.events[ToolResultEvent.CHANNEL]) >= 1
    assert len(collector.events[DecideEvent.CHANNEL]) >= 1
    assert len(collector.events[BeforeRunEvent.CHANNEL]) >= 1
    assert len(collector.events[StreamDeltaEvent.CHANNEL]) >= 1
    assert len(collector.events[ContextEvent.CHANNEL]) >= 1
    assert len(collector.events[BeforeSendEvent.CHANNEL]) >= 1


@pytest.mark.asyncio
async def test_bus_priority_and_error_suppression() -> None:
    bus = EventBus()
    order: list[int] = []

    bus.on("test_ch", lambda _: order.append(100), priority=BusPriority.PRE)

    def raiser(_: Any) -> None:
        order.append(500)
        raise ValueError("deliberate")

    bus.on("test_ch", raiser, priority=BusPriority.NORMAL)
    bus.on("test_ch", lambda _: order.append(900), priority=BusPriority.POST)

    results = await bus.emit("test_ch", "event")
    assert order == [100, 500, 900]
    assert results[0] is None  # PRE handler returned None
    assert results[1] is None  # raiser's slot is None (suppressed)
    assert results[2] is None  # POST handler returned None


# ---------------------------------------------------------------------------
# GROUP 4: Store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_persistence() -> None:
    store = InMemoryTrajectoryStore()
    mock = MockStreamFn()
    mock.enqueue(text_response("first"), text_response("second"))

    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=mock,
        model=make_model(),
        system="test",
        store=store,
    ))
    session.start()
    await session.prompt("one")
    await _wait_turn(session)
    await session.shutdown()

    # Create a new session for the second turn
    mock2 = MockStreamFn()
    mock2.enqueue(text_response("second"))
    session2 = await Session.resume(
        session.id,
        _memory_store(store),
        AgentSessionConfig(
            extensions=[],
            stream_fn=mock2,
            model=make_model(),
            system="test",
        ),
    )
    # Session reuses ID but store already has it, so we don't re-create
    session2.start()
    await session2.prompt("two")
    await _wait_turn(session2)
    await session2.shutdown()

    meta, turns = store.load(session.id)
    assert len(turns) == 2
    assert store.session_exists(session.id)
    sessions = store.list_sessions()
    assert any(m.id == session.id for m in sessions)

    with pytest.raises(ValueError):
        store.create_session(
            SessionMeta(id=session.id),
            head=_empty_head(session.id),
        )


@pytest.mark.asyncio
async def test_session_resume() -> None:
    store = InMemoryTrajectoryStore()
    mock = MockStreamFn()
    mock.enqueue(text_response("turn-1"))

    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=mock,
        model=make_model(),
        system="test",
        store=store,
    ))
    session.start()
    await session.prompt("first")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1

    mock2 = MockStreamFn()
    mock2.enqueue(text_response("turn-2"))
    resumed = await Session.resume(
        session.id,
        _memory_store(store),
        AgentSessionConfig(
            extensions=[],
            stream_fn=mock2,
            model=make_model(),
            system="test",
        ),
    )

    # Trajectory was loaded from store
    assert len(resumed.trajectory) == 1
    assert resumed.trajectory.turns[0].rounds[0].response.content[0].text == "turn-1"  # type: ignore[union-attr]
    assert resumed.ctx.session_id == session.id

    resumed.start()
    await resumed.prompt("second")
    await _wait_turn(resumed)
    await resumed.shutdown()
    assert len(resumed.trajectory) == 2
    assert resumed.trajectory.turns[1].index == 1


@pytest.mark.asyncio
async def test_resume_rejects_unversioned_session_metadata() -> None:
    source_store = InMemoryTrajectoryStore()
    mock = MockStreamFn()
    mock.enqueue(text_response("turn-1"))
    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=mock,
        model=make_model(),
        store=source_store,
    ))
    session.start()
    await session.prompt("first")
    await _wait_turn(session)
    await session.shutdown()

    legacy_store = InMemoryTrajectoryStore()
    legacy_store.create_session(
        SessionMeta(id="legacy-session"),
        turns=session.trajectory.turns,
        head=_empty_head("legacy-session"),
    )
    with pytest.raises(
        ResumeIdentityError,
        match="session_metadata_version",
    ):
        await Session.resume(
            "legacy-session",
            _memory_store(legacy_store),
            AgentSessionConfig(
                extensions=[],
                stream_fn=MockStreamFn(),
                model=make_model(),
            ),
        )


def test_jsonl_torn_tail_recovers_but_interior_corruption_fails(
    tmp_path: Path,
) -> None:
    store = JsonlTrajectoryStore(tmp_path)
    meta = SessionMeta(id="session")
    store.create_session(meta, head=_empty_head(meta.id))
    turn = Turn(
        index=0,
        id="turn-0",
        trigger=UserInput(content=(TextContent(type="text", text="one"),)),
        rounds=(Round(response=text_response("done")),),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=1.0,
    )
    store.commit_turn(meta.id, TrajectoryCommit(turn, (), None))
    path = store.file_path(meta.id)
    with path.open("ab") as fh:
        fh.write(b'{"index":1')

    assert store.load(meta.id)[1] == [turn]
    next_turn = Turn(
        index=1,
        id="turn-1",
        trigger=UserInput(content=(TextContent(type="text", text="two"),)),
        rounds=(Round(response=text_response("done again")),),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=2.0,
    )
    store.commit_turn(meta.id, TrajectoryCommit(next_turn, (), None))
    assert store.load(meta.id)[1] == [turn, next_turn]

    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(2, "not-json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt trajectory record 3"):
        store.load(meta.id)


def test_jsonl_create_with_turns_has_no_partial_session(tmp_path: Path) -> None:
    store = JsonlTrajectoryStore(tmp_path)
    meta = SessionMeta(id="fork")
    bad_turn = Turn(
        index=1,
        id="bad",
        trigger=ContinueTrigger(),
        rounds=(),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=0.0,
    )
    with pytest.raises(ValueError, match="expected 0"):
        store.create_session(
            meta,
            turns=[bad_turn],
            head=_empty_head(meta.id),
        )
    assert not store.session_exists(meta.id)


def test_trajectory_query_store_adapter_filters_sessions_and_turns() -> None:
    store = InMemoryTrajectoryStore()
    root_turn = Turn(
        index=0,
        id="root-turn",
        trigger=UserInput(content=(TextContent(type="text", text="root"),)),
        rounds=(Round(response=text_response("root done")),),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=1.0,
    )
    child_turn = Turn(
        index=0,
        id="child-turn",
        trigger=UserInput(content=(TextContent(type="text", text="child"),)),
        rounds=(Round(response=text_response("child done")),),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=2.0,
    )
    store.create_session(
        SessionMeta(
            id="root",
            purpose="root",
            created_at=10.0,
            config={"root_session_id": "root", "depth": 0},
        ),
        turns=[root_turn],
        head=_empty_head("root"),
    )
    store.create_session(
        SessionMeta(
            id="child",
            parent_id="root",
            purpose="worker",
            created_at=20.0,
            config={"root_session_id": "root", "depth": 1},
        ),
        turns=[child_turn],
        head=_empty_head("child"),
    )

    query = TrajectoryStoreQueryAdapter(store)

    assert [session.id for session in query.sessions()] == ["root", "child"]
    assert [turn.id for turn in query.turns("child")] == ["child-turn"]
    assert [
        session.id
        for session in query.sessions(SessionFilter(parent_session_id="root"))
    ] == ["child"]
    assert [
        session.id
        for session in query.sessions(SessionFilter(root_session_id="root"))
    ] == ["root", "child"]
    assert [
        session.id for session in query.sessions(SessionFilter(purpose="worker"))
    ] == ["child"]
    assert [
        session.id
        for session in query.sessions(SessionFilter(since=15.0, limit=1))
    ] == ["child"]
    assert isinstance(query, TrajectoryQueryStore)
    assert not isinstance(query, ObservabilityQueryStore)
    with pytest.raises(KeyError):
        list(query.turns("missing"))


@pytest.mark.asyncio
async def test_session_factory_registers_default_trajectory_query_store() -> None:
    store = InMemoryTrajectoryStore()
    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=MockStreamFn(),
        model=make_model(),
        store=store,
    ))

    query = session.services.get(
        TRAJECTORY_QUERY_STORE_SERVICE,
        TrajectoryQueryStore,
    )

    assert isinstance(query, TrajectoryQueryStore)
    assert [entry.id for entry in query.sessions()] == [session.id]


# ---------------------------------------------------------------------------
# GROUP 5: Session Graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_inheritance() -> None:
    graph = InMemorySessionGraph()
    mock = MockStreamFn()
    mock.enqueue(text_response("parent"))

    parent = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test system",
        graph=graph, tools=[_make_add_tool()],
    ))
    parent.services.register("test_svc", {"data": 42})
    parent.start()
    await parent.prompt("go")
    await _wait_turn(parent)

    child = await parent.spawn(purpose="child-task")

    assert child.ctx.parent_session_id == parent.id
    assert child.ctx.depth == 1
    assert len(child.tools) == len(parent.tools)
    assert child.system == parent.system
    assert child.services.get("test_svc") is not None
    assert len(child.trajectory) == 0
    assert child.bus is not parent.bus

    parent.services.register("new_svc", {"new": True})
    assert child.services.get("new_svc") is None

    await child.shutdown()
    await parent.shutdown()


@pytest.mark.asyncio
async def test_fork_lifecycle() -> None:
    graph = InMemorySessionGraph()
    mock = MockStreamFn()
    mock.enqueue(text_response("turn-0"), text_response("turn-1"))

    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test", graph=graph,
    ))
    session.start()
    await session.prompt("first")
    await _wait_turn(session)
    await session.shutdown()

    mock2 = MockStreamFn()
    mock2.enqueue(text_response("turn-1b"))
    session2 = Session(SessionRuntimeConfig(
        stream_fn=mock2, model=make_model(), system="test", graph=graph,
        session_id=session.id,
        trajectory=Trajectory(turns=list(session.trajectory.turns)),
    ))
    session2.start()
    await session2.prompt("second")
    await _wait_turn(session2)
    await session2.shutdown()

    assert len(session.trajectory) >= 1
    forked = await Session.fork(session, at=0, purpose="fork-test")
    assert len(forked.trajectory) == 1
    assert forked.ctx.parent_session_id == session.id
    assert [tool.name for tool in forked.tools] == [tool.name for tool in session.tools]

    edges = graph.edges(session.id)
    fork_edges = [e for e in edges if e.kind == "forked"]
    assert len(fork_edges) == 1
    assert fork_edges[0].child_id == forked.id
    await forked.shutdown()


def test_session_graph_traversals() -> None:
    graph = InMemorySessionGraph()
    graph.register("A", purpose="root")
    graph.register("B", parent_id="A", edge_kind="spawned")
    graph.register("C", parent_id="B", edge_kind="spawned")
    graph.register("D", parent_id="A", edge_kind="forked", fork_point=0)

    assert graph.ancestors("C") == ["B", "A"]
    assert graph.root("C") == "A"

    desc = graph.descendants("A")
    assert "B" in desc
    assert "C" in desc
    assert "D" in desc

    assert graph.children("A", kind="forked") == ["D"]
    assert set(graph.children("A")) == {"B", "D"}
    assert graph.children("A", kind="spawned") == ["B"]


# ---------------------------------------------------------------------------
# GROUP 6: Driver Control Flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_action_priority() -> None:
    from agentm.core.runtime.reaction import _resolve_action

    inject_msg = UserMessage(
        role="user", content=[TextContent(type="text", text="injected")], timestamp=0.0,
    )

    # Case A: Inject with non-empty messages vs Stop -> Inject wins
    result = _resolve_action(Step(), [Stop(), Inject(messages=(inject_msg,))])
    assert isinstance(result, Inject)

    # Case B: Stop vs Step -> Stop wins
    result = _resolve_action(Step(), [Step(), Stop(cause=ModelEndTurn())])
    assert isinstance(result, Stop)

    # Case C: final Stop (SignalAborted.final=True) overrides everything
    result = _resolve_action(
        Stop(cause=SignalAborted()),
        [Step(), Inject(messages=(inject_msg,))],
    )
    assert isinstance(result, Stop)
    assert isinstance(result.cause, SignalAborted)


@pytest.mark.asyncio
async def test_inject_inline() -> None:
    mock = MockStreamFn()
    inject_msg = UserMessage(
        role="user",
        content=[TextContent(type="text", text="injected")],
        timestamp=0.0,
    )

    first_decide = {"done": False}

    def decide_handler(event: DecideEvent) -> LoopAction | None:
        if not first_decide["done"]:
            first_decide["done"] = True
            return Inject(messages=(inject_msg,))
        return None

    mock.enqueue(text_response("first"), text_response("second"))
    session = Session(
        SessionRuntimeConfig(stream_fn=mock, model=make_model(), system="test")
    )
    session.bus.on(DecideEvent.CHANNEL, decide_handler)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert len(turn.rounds) == 2
    live_messages = list(mock.calls[1]["messages"])
    cold_messages = build_context_sync(session.trajectory.turns)
    assert cold_messages[:-1] == live_messages
    assert cold_messages[-1] == turn.rounds[-1].response


@pytest.mark.asyncio
async def test_signal_abort() -> None:
    mock = MockStreamFn()
    # Use a tool call to get a multi-round turn. The signal set during
    # round 1 is caught at the top of round 2.
    mock.enqueue(
        tool_call_response("add", "sa1", {"a": 1, "b": 1}),
        text_response("should not reach"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[_make_add_tool()],
    ))

    def on_result(event: ToolResultEvent) -> None:
        session.interrupt()

    session.bus.on(ToolResultEvent.CHANNEL, on_result)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    turns = session.trajectory.turns
    assert len(turns) == 1
    turn = turns[0]
    assert turn.outcome.cause is not None
    assert isinstance(turn.outcome.cause, SignalAborted)


@pytest.mark.asyncio
async def test_before_run_veto() -> None:
    mock = MockStreamFn()
    mock.enqueue(text_response("should not reach"))
    session = Session(
        SessionRuntimeConfig(stream_fn=mock, model=make_model(), system="test")
    )

    def veto_handler(event: BeforeRunEvent) -> dict[str, Any]:
        return {"veto": BudgetExhausted(detail="test")}

    session.bus.on(BeforeRunEvent.CHANNEL, veto_handler)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert isinstance(turn.outcome.cause, BudgetExhausted)
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# GROUP 7: Tool Event Hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_blocked() -> None:
    tool_executed = {"called": False}

    async def blocked_fn(args: dict[str, Any]) -> ToolResult:
        tool_executed["called"] = True
        return ToolResult(content=[TextContent(type="text", text="should not happen")])

    tool = FunctionTool(
        name="blocked_tool", description="will be blocked",
        parameters={"type": "object"}, fn=blocked_fn,
    )

    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("blocked_tool", "bc1", {}),
        text_response("ok"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[tool],
    ))

    def block_handler(event: ToolCallEvent) -> dict[str, Any]:
        return {"block": True, "reason": "denied"}

    session.bus.on(ToolCallEvent.CHANNEL, block_handler)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert not tool_executed["called"]
    turn = session.trajectory.turns[0]
    tr = turn.rounds[0].tool_results[0]
    assert tr.result.is_error
    assert "blocked" in tr.result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_tool_result_replaced() -> None:
    async def echo_fn(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="original")])

    tool = FunctionTool(
        name="echo", description="echo",
        parameters={"type": "object"}, fn=echo_fn,
    )

    replacement = ToolResult(
        content=[TextContent(type="text", text="replaced")],
        is_error=False,
    )
    observed_outcomes: list[ToolResult] = []

    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("echo", "e1", {}),
        text_response("done"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[tool],
    ))

    def replace_handler(event: ToolResultEvent) -> ToolResult:
        return replacement

    def observe_decision(event: DecideEvent) -> None:
        for _name, outcome in event.observation.tool_outcomes:
            if isinstance(outcome, (ToolContinue, ToolTerminate)):
                observed_outcomes.append(outcome.result)

    session.bus.on(ToolResultEvent.CHANNEL, replace_handler)
    session.bus.on(DecideEvent.CHANNEL, observe_decision)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    turn = session.trajectory.turns[0]
    tr = turn.rounds[0].tool_results[0]
    assert tr.result.content[0].text == "replaced"  # type: ignore[union-attr]
    assert observed_outcomes == [replacement]


# ---------------------------------------------------------------------------
# GROUP 8: Triggers & Context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_waits_for_its_receipt_while_idle_tracks_background() -> None:
    mock = MockStreamFn()
    mock.enqueue(text_response("done"))
    session = Session(
        SessionRuntimeConfig(stream_fn=mock, model=make_model(), system="test")
    )
    session.triggers.note_work_started()
    try:
        messages = await asyncio.wait_for(session.run("go"), timeout=1.0)
        assert messages[-1].content[0].text == "done"  # type: ignore[union-attr]
        assert not await session.idle(timeout=0.01)
    finally:
        session.triggers.note_work_finished()
        await session.shutdown()
    assert await session.idle(timeout=1.0)


@pytest.mark.asyncio
async def test_trigger_queue_lifecycle() -> None:
    tq = TriggerQueue()
    trigger = UserInput(content=(TextContent(type="text", text="hi"),))
    receipt = tq.push(trigger)
    got = await tq.wait()
    assert got is trigger
    tq.complete("done")
    assert await receipt.wait() == "done"

    tq.kick()
    got = await tq.wait()
    assert isinstance(got, ContinueTrigger)
    tq.complete("kicked")

    tq.close()
    with pytest.raises(QueueClosed):
        await tq.wait()


def test_render_trigger_all_types() -> None:
    user = UserInput(content=(TextContent(type="text", text="hello"),))
    msgs = render_trigger(user)
    assert len(msgs) == 1
    assert isinstance(msgs[0], UserMessage)
    assert msgs[0].content[0].text == "hello"  # type: ignore[union-attr]

    cont = ContinueTrigger()
    msgs = render_trigger(cont)
    assert msgs == []

    inj_msg = UserMessage(
        role="user", content=[TextContent(type="text", text="injected")], timestamp=0.0,
    )
    inj = Injection(messages=(inj_msg,))
    msgs = render_trigger(inj)
    assert len(msgs) == 1
    assert msgs[0] is inj_msg

    bg = BackgroundCompletion(task_id="t1", payload="bg done")
    msgs = render_trigger(bg)
    assert len(msgs) == 1
    assert "background" in msgs[0].content[0].text  # type: ignore[union-attr]

    mon = MonitorFire(monitor_id="m1", payload="monitor fired")
    msgs = render_trigger(mon)
    assert len(msgs) == 1
    assert "monitor" in msgs[0].content[0].text  # type: ignore[union-attr]

    sub = SubagentResult(child_session_id="s1", payload="sub result")
    msgs = render_trigger(sub)
    assert len(msgs) == 1
    assert "subagent" in msgs[0].content[0].text  # type: ignore[union-attr]

    class UnknownTrigger:
        source = "custom"

    unk = UnknownTrigger()
    with pytest.raises(LookupError, match="no registered TriggerRenderer"):
        render_trigger(unk)


# ---------------------------------------------------------------------------
# GROUP 9: Codec
# ---------------------------------------------------------------------------


def test_codec_round_trip() -> None:
    codec = CodecRegistry()

    trigger = UserInput(content=(TextContent(type="text", text="test input"),))
    response = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="response text"),
            OpaqueThinkingBlock(
                type="opaque_thinking",
                provider="anthropic",
                payload={
                    "type": "redacted_thinking",
                    "data": "encrypted-reasoning",
                },
            ),
        ],
        timestamp=1234.0,
        stop_reason="end_turn",
    )
    outcome = Outcome(cause=ModelEndTurn())
    meta = TurnMeta(
        total_input_tokens=100, total_output_tokens=50,
        model_id="mock-model",
        resource_mutations=(
            ResourceMutation(
                ref=ResourceRef(namespace="workspace", path="out.txt"),
                op="write",
                after_version="v1",
                metadata={"tool": "write"},
            ),
        ),
    )
    turn = Turn(
        index=0, id="turn-abc", trigger=trigger,
        rounds=(Round(response=response, tool_results=()),),
        outcome=outcome, timestamp=1234.0, meta=meta,
    )

    data = codec.serialize_turn(turn)
    restored = codec.deserialize_turn(data)

    assert restored.index == turn.index
    assert restored.id == turn.id
    assert restored.timestamp == turn.timestamp
    assert type(restored.outcome.cause).__name__ == type(turn.outcome.cause).__name__
    assert isinstance(restored.outcome.cause, ModelEndTurn)
    assert restored.meta.total_input_tokens == 100
    assert restored.meta.model_id == "mock-model"
    assert restored.meta.resource_mutations == meta.resource_mutations
    assert isinstance(restored.trigger, UserInput)
    assert len(restored.rounds) == 1
    assert restored.rounds[0].response.content[0].text == "response text"  # type: ignore[union-attr]
    opaque = restored.rounds[0].response.content[1]
    assert isinstance(opaque, OpaqueThinkingBlock)
    assert dict(opaque.payload) == {
        "type": "redacted_thinking",
        "data": "encrypted-reasoning",
    }


def test_codec_custom_trigger() -> None:
    codec = CodecRegistry()

    class CustomTrigger:
        source = "custom_source"
        data_field = "hello"

    class CustomCodec:
        def serialize(self, trigger: Any) -> dict[str, Any]:
            return {"__source__": "custom_source", "data_field": trigger.data_field}

        def deserialize(self, data: dict[str, Any]) -> Any:
            t = CustomTrigger()
            t.data_field = data.get("data_field", "")
            return t

    codec.register_trigger_codec("custom_source", CustomCodec())  # type: ignore[arg-type]

    t = CustomTrigger()
    serialized = codec.serialize_trigger(t)
    assert serialized["__source__"] == "custom_source"
    deserialized = codec.deserialize_trigger(serialized)
    assert deserialized.data_field == "hello"

    unknown_data = {"__source__": "unregistered", "foo": "bar"}
    raw = codec.deserialize_trigger(unknown_data)
    assert isinstance(raw, RawTrigger)
    assert raw.source == "unregistered"


# ---------------------------------------------------------------------------
# GROUP 10: State Machine
# ---------------------------------------------------------------------------


def test_execution_state_errors() -> None:
    trigger = UserInput(content=(TextContent(type="text", text="x"),))
    ex = Execution(index=0, trigger=trigger)
    ex.abandon()

    with pytest.raises(StateError):
        response = AssistantMessage(
            role="assistant", content=[], timestamp=0.0,
        )
        ex.add_round(response, [])

    ex2 = Execution(index=0, trigger=trigger)
    outcome = Outcome(cause=ModelEndTurn())
    meta = TurnMeta()
    ex2.commit(outcome, meta)

    with pytest.raises(StateError):
        ex2.commit(outcome, meta)

    traj = Trajectory()
    traj.begin(trigger)
    with pytest.raises(StateError):
        traj.begin(trigger)


# ---------------------------------------------------------------------------
# GROUP 11: Robustness -- External
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_exception_becomes_error_result() -> None:
    async def bad_fn(args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("tool crashed")

    tool = FunctionTool(
        name="bad_tool", description="crashes",
        parameters={"type": "object"}, fn=bad_fn,
    )

    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("bad_tool", "bt1", {}),
        text_response("recovered"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[tool],
    ))
    collector = EventCollector(session.bus)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    tr = turn.rounds[0].tool_results[0]
    assert tr.result.is_error
    assert "tool crashed" in tr.result.content[0].text  # type: ignore[union-attr]
    assert len(collector.events[ToolErrorEvent.CHANNEL]) >= 1


@pytest.mark.asyncio
async def test_unknown_tool_error() -> None:
    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("nonexistent_tool", "nt1", {}),
        text_response("ok"),
    )
    session = Session(
        SessionRuntimeConfig(stream_fn=mock, model=make_model(), system="test")
    )
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    tr = turn.rounds[0].tool_results[0]
    assert tr.result.is_error
    assert "unknown tool" in tr.result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_tool_error_event_custom_text() -> None:
    async def bad_fn(args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("original error")

    tool = FunctionTool(
        name="err_tool", description="errors",
        parameters={"type": "object"}, fn=bad_fn,
    )

    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("err_tool", "et1", {}),
        text_response("done"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[tool],
    ))

    def custom_error(event: ToolErrorEvent) -> dict[str, str]:
        return {"text": "please retry"}

    session.bus.on(ToolErrorEvent.CHANNEL, custom_error)
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    turn = session.trajectory.turns[0]
    tr = turn.rounds[0].tool_results[0]
    assert tr.result.content[0].text == "please retry"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_empty_llm_response() -> None:
    mock = MockStreamFn()
    mock.enqueue(empty_response())
    session = Session(
        SessionRuntimeConfig(stream_fn=mock, model=make_model(), system="test")
    )
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert turn.outcome.cause is not None
    assert len(turn.rounds[0].response.content) == 0


@pytest.mark.asyncio
async def test_tool_empty_result() -> None:
    async def empty_fn(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[], is_error=False)

    tool = FunctionTool(
        name="empty_tool", description="returns empty",
        parameters={"type": "object"}, fn=empty_fn,
    )

    mock = MockStreamFn()
    mock.enqueue(
        tool_call_response("empty_tool", "emp1", {}),
        text_response("done"),
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        tools=[tool],
    ))
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    tr = turn.rounds[0].tool_results[0]
    assert not tr.result.is_error
    assert len(tr.result.content) == 0


# ---------------------------------------------------------------------------
# GROUP 12: Robustness -- Internal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_fn_exception_persists_non_replayable_turn() -> None:
    delegate = MockStreamFn()
    delegate.enqueue(text_response("ok after failure"))
    failing = FailingStreamFn(fail_count=1, delegate=delegate)

    session = Session(SessionRuntimeConfig(  # type: ignore[arg-type]
        stream_fn=failing,
        model=make_model(),
        system="test",
    ))
    session.start()
    receipt = await session.prompt("will fail")
    with pytest.raises(TriggerTerminated, match="stream failure"):
        await receipt.wait()
    assert await session.idle(timeout=1.0)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert isinstance(turn.outcome.cause, ProviderRequestFailed)
    assert turn.outcome.cause.detail == "stream failure #1"
    assert build_context_sync(session.trajectory.turns) == []


@pytest.mark.asyncio
async def test_policy_exception_abandons_turn() -> None:
    call_count = {"n": 0}

    class FailOncePolicy:
        async def transform(
            self, messages: list[AgentMessage], turns: Sequence[Turn],
        ) -> list[AgentMessage]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("policy exploded")
            return messages

        def bind(self, ctx: PolicyContext) -> None:
            pass

    mock = MockStreamFn()
    mock.enqueue(text_response("after policy fix"))

    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test",
        context_policies=[FailOncePolicy()],  # type: ignore[list-item]
    ))
    session.start()
    receipt = await session.prompt("will fail due to policy")
    with pytest.raises(RuntimeError, match="policy exploded"):
        await receipt.wait()
    await session.shutdown()

    assert len(session.trajectory) == 0


@pytest.mark.asyncio
async def test_store_append_failure_is_fail_stop() -> None:
    fail_count = {"n": 0}

    class FailingStore(InMemoryTrajectoryStore):
        def commit_turn(
            self,
            session_id: str,
            commit: TrajectoryCommit,
        ) -> None:
            fail_count["n"] += 1
            if fail_count["n"] == 1:
                raise IOError("store write failed")
            super().commit_turn(session_id, commit)

    store = FailingStore()
    mock = MockStreamFn()
    mock.enqueue(text_response("turn-1"))

    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test", store=store,
    ))
    store.create_session(
        SessionMeta(id=session.id, purpose="root"),
        head=_empty_head(session.id),
    )
    session.start()
    receipt = await session.prompt("go")
    with pytest.raises(OSError, match="store write failed"):
        await receipt.wait()
    await session.shutdown()

    assert len(session.trajectory) == 0
    assert store.load(session.id)[1] == []


@pytest.mark.asyncio
async def test_bus_handler_exception_suppressed() -> None:
    bus = EventBus()
    results_tracker: list[str] = []

    bus.on("ch", lambda _: results_tracker.append("before"))

    def raiser(_: Any) -> None:
        raise ValueError("boom")

    bus.on("ch", raiser)
    bus.on("ch", lambda _: results_tracker.append("after"))

    results = await bus.emit("ch", "data")
    assert results_tracker == ["before", "after"]
    assert results[1] is None


@pytest.mark.asyncio
async def test_consecutive_stream_failures() -> None:
    delegate = MockStreamFn()
    delegate.enqueue(text_response("success"))
    failing = FailingStreamFn(fail_count=2, delegate=delegate)

    session = Session(SessionRuntimeConfig(  # type: ignore[arg-type]
        stream_fn=failing,
        model=make_model(),
        system="test",
    ))
    session.start()
    receipt = await session.prompt("fail-1")
    with pytest.raises(TriggerTerminated, match="stream failure"):
        await receipt.wait()
    await session.shutdown()

    assert len(session.trajectory) == 1
    assert isinstance(
        session.trajectory.turns[0].outcome.cause,
        ProviderRequestFailed,
    )
    assert failing.call_count == 1


@pytest.mark.asyncio
async def test_durable_round_persist_failure() -> None:
    class FailingRoundStore(InMemoryTrajectoryStore):
        def append_round(
            self, session_id: str, turn_id: str, round_data: dict[str, object],
        ) -> None:
            raise IOError("round persist failed")

    store = FailingRoundStore()
    mock = MockStreamFn()
    mock.enqueue(text_response("ok"))

    session = Session(SessionRuntimeConfig(
        stream_fn=mock, model=make_model(), system="test", store=store,
    ))
    store.create_session(
        SessionMeta(id=session.id, purpose="root"),
        head=_empty_head(session.id),
    )
    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(session.trajectory) == 1
    turn = session.trajectory.turns[0]
    assert turn.outcome.cause is not None
