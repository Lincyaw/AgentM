"""SDK-user behavior contracts for durable trajectory workflows."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agentm import AgentSession, AgentSessionConfig, Model
from agentm.core.abi.cancel import CancelSignal, cancel_reason
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    text_message,
)
from agentm.core.abi.provider import ProviderPromptCacheRequest
from agentm.core.abi.store import TrajectoryNodeStore
from agentm.core.abi.stream import MessageEnd, TextDelta
from agentm.core.abi.trajectory import PromptCacheState
from agentm.core.runtime.stores.jsonl import JsonlTrajectoryStore
from agentm.extensions.builtin.llm_openai import (
    OpenAIPromptCacheAdapter,
    OpenAIStreamFn,
)
from agentm.storage.resources import LocalResourceStore
from agentm.storage.trajectory import (
    JsonlTrajectoryNodeStore,
    PostgresTrajectoryNodeStore,
)

_CONTEXT_PROJECTION = "agentm.extensions.builtin.context_projection"
_LLM_COMPACTION = "agentm.extensions.builtin.llm_compaction"
_PROMPT_CACHE = "agentm.extensions.builtin.prompt_cache"
_OBSERVABLE_CACHE_ADAPTER = "tests.fixtures.prompt_cache_adapter"
_WAIT_FOR_CANCEL = object()


@dataclass(frozen=True)
class _NodeStoreBackend:
    open: Callable[[], TrajectoryNodeStore]


class _StubProvider:
    """Deterministic provider double at the public StreamFn boundary."""

    def __init__(self, *actions: str | object) -> None:
        self._actions = list(actions)
        self.requests: list[tuple[AgentMessage, ...]] = []
        self.stream_started = asyncio.Event()
        self.observed_cancel_reason: str | None = None

    async def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: str = "off",
    ) -> AsyncIterator[TextDelta | MessageEnd]:
        del model, tools, system, thinking
        self.requests.append(tuple(messages))
        if not self._actions:
            raise RuntimeError("stub provider has no queued response")
        action = self._actions.pop(0)
        if action is _WAIT_FOR_CANCEL:
            if signal is None:
                raise AssertionError("SDK did not pass a cancellation signal")
            self.stream_started.set()
            await signal.wait()
            self.observed_cancel_reason = cancel_reason(signal)
            raise RuntimeError("provider request cancelled")
        if not isinstance(action, str):
            raise TypeError(f"unsupported provider action: {action!r}")
        response = AssistantMessage(
            role="assistant",
            content=(TextContent(type="text", text=action),),
            timestamp=0.0,
            stop_reason="end_turn",
        )
        yield TextDelta(text=action)
        yield MessageEnd(message=response)


@pytest.fixture(params=("jsonl", "postgres"))
def node_store_backend(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[_NodeStoreBackend]:
    if request.param == "jsonl":
        root = tmp_path / "trajectory-nodes"
        yield _NodeStoreBackend(open=lambda: JsonlTrajectoryNodeStore(root))
        return

    database_url = os.environ.get("AGENTM_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("set AGENTM_TEST_POSTGRES_URL to run Postgres behavior contracts")
    psycopg = pytest.importorskip("psycopg")
    schema = f"agentm_test_{uuid.uuid4().hex}"
    admin = psycopg.connect(database_url)
    connections: list[Any] = []
    with admin.cursor() as cursor:
        cursor.execute(f'CREATE SCHEMA "{schema}"')
    admin.commit()

    def open_store() -> TrajectoryNodeStore:
        connection = psycopg.connect(database_url)
        connections.append(connection)
        return PostgresTrajectoryNodeStore(connection, schema=schema)

    try:
        yield _NodeStoreBackend(open=open_store)
    finally:
        for connection in connections:
            connection.close()
        with admin.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.commit()
        admin.close()


def _model() -> Model:
    return Model(
        id="stub-model",
        provider="stub",
        context_window=128_000,
        max_output_tokens=4_096,
    )


def _trajectory_extensions(cache_key: str) -> list[tuple[str, dict[str, object]]]:
    return [
        (_CONTEXT_PROJECTION, {"mode": "exact_node_chain"}),
        (
            _PROMPT_CACHE,
            {
                "cache_key": cache_key,
                "tag_last_messages": 2,
                "register_provider_adapter": False,
            },
        ),
        (_OBSERVABLE_CACHE_ADAPTER, {}),
    ]


def _text(messages: Sequence[AgentMessage]) -> list[str]:
    return [
        block.text
        for message in messages
        for block in message.content
        if isinstance(block, TextContent)
    ]


def _cache_markers(messages: Sequence[AgentMessage]) -> set[str]:
    return {
        marker
        for message in messages
        if isinstance(
            marker := message.meta.tags.get("provider_cache_marker"),
            str,
        )
    }


@pytest.mark.asyncio
async def test_sdk_resume_replays_history_and_durable_cache_state(
    node_store_backend: _NodeStoreBackend,
    tmp_path: Path,
) -> None:
    turn_store_path = tmp_path / "turns"
    cache_key = "resume-prefix"
    provider = _StubProvider("answer-one", "answer-two")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=_trajectory_extensions(cache_key),
            stream_fn=provider,
            model=_model(),
            store=JsonlTrajectoryStore(turn_store_path),
            trajectory_node_store=node_store_backend.open(),
        )
    )
    session_id = session.session_id
    try:
        await session.run("question-one")
        await session.run("question-two")
    finally:
        await session.shutdown()

    assert _cache_markers(provider.requests[1]) == {"resume-prefix:1"}

    resumed_provider = _StubProvider("answer-three")
    resumed = await AgentSession.resume(
        session_id,
        JsonlTrajectoryStore(turn_store_path),
        config=AgentSessionConfig(
            extensions=_trajectory_extensions(cache_key),
            stream_fn=resumed_provider,
            model=_model(),
            trajectory_node_store=node_store_backend.open(),
        ),
    )
    try:
        transcript = await resumed.run("question-three")
    finally:
        await resumed.shutdown()

    assert _text(resumed_provider.requests[0]) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
        "question-three",
    ]
    assert _cache_markers(resumed_provider.requests[0]) == {"resume-prefix:2"}
    assert _text(transcript) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
        "question-three",
        "answer-three",
    ]


@pytest.mark.asyncio
async def test_sdk_fork_replays_only_the_selected_prefix(
    node_store_backend: _NodeStoreBackend,
    tmp_path: Path,
) -> None:
    provider = _StubProvider("answer-one", "parent-answer", "branch-answer")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=_trajectory_extensions("fork-prefix"),
            stream_fn=provider,
            model=_model(),
            store=JsonlTrajectoryStore(tmp_path / "turns"),
            trajectory_node_store=node_store_backend.open(),
        )
    )
    forked: AgentSession | None = None
    try:
        await session.run("question-one")
        await session.run("parent-only-question")
        forked = await AgentSession.fork(session, at=0, purpose="alternate")
        transcript = await forked.run("branch-only-question")
    finally:
        if forked is not None:
            await forked.shutdown()
        await session.shutdown()

    assert _text(provider.requests[2]) == [
        "question-one",
        "answer-one",
        "branch-only-question",
    ]
    assert _cache_markers(provider.requests[2]) == {"fork-prefix:2"}
    assert _text(transcript) == [
        "question-one",
        "answer-one",
        "branch-only-question",
        "branch-answer",
    ]


@pytest.mark.asyncio
async def test_sdk_interrupt_cancels_one_request_and_session_continues() -> None:
    provider = _StubProvider(_WAIT_FOR_CANCEL, "continued-answer")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[],
            stream_fn=provider,
            model=_model(),
        )
    )
    interrupted = asyncio.create_task(session.run("long-running-question"))
    try:
        await asyncio.wait_for(provider.stream_started.wait(), timeout=2.0)
        session.interrupt("user_cancel")
        await asyncio.wait_for(interrupted, timeout=2.0)
        transcript = await session.run("question-after-interrupt")
    finally:
        if not interrupted.done():
            interrupted.cancel()
        await session.shutdown()

    assert provider.observed_cancel_reason == "user_cancel"
    assert _text(provider.requests[1])[-1] == "question-after-interrupt"
    assert _text(transcript)[-2:] == [
        "question-after-interrupt",
        "continued-answer",
    ]
    assert session.status()["phase"] == "closed"


@pytest.mark.asyncio
async def test_sdk_compaction_persists_summary_across_resume(
    node_store_backend: _NodeStoreBackend,
    tmp_path: Path,
) -> None:
    turn_store_path = tmp_path / "compaction-turns"
    resource_root = tmp_path / "resources"

    def open_resources() -> LocalResourceStore:
        return LocalResourceStore(
            workspace_root=tmp_path,
            root=resource_root,
        )

    extensions = [
        (
            _LLM_COMPACTION,
            {
                "max_messages": 5,
                "keep_last_turns": 1,
            },
        )
    ]
    provider = _StubProvider(
        "answer-one",
        "answer-two",
        "answer-three",
        "durable-summary",
        "answer-four",
    )
    resources = open_resources()
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=extensions,
            stream_fn=provider,
            model=_model(),
            store=JsonlTrajectoryStore(turn_store_path),
            trajectory_node_store=node_store_backend.open(),
            resource_store=resources,
            resource_writer=resources,
        )
    )
    session_id = session.session_id
    try:
        await session.run("question-one")
        await session.run("question-two")
        await session.run("question-three")
        await session.run("question-four")
    finally:
        await session.shutdown()

    assert _text(provider.requests[3]) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
    ]
    assert _text(provider.requests[4]) == [
        "<conversation-summary>\ndurable-summary\n</conversation-summary>",
        "question-three",
        "answer-three",
        "question-four",
    ]

    resumed_provider = _StubProvider("answer-five")
    resumed_resources = open_resources()
    resumed = await AgentSession.resume(
        session_id,
        JsonlTrajectoryStore(turn_store_path),
        config=AgentSessionConfig(
            extensions=extensions,
            stream_fn=resumed_provider,
            model=_model(),
            trajectory_node_store=node_store_backend.open(),
            resource_store=resumed_resources,
            resource_writer=resumed_resources,
        ),
    )
    try:
        transcript = await resumed.run("question-five")
    finally:
        await resumed.shutdown()

    assert len(resumed_provider.requests) == 1
    assert _text(resumed_provider.requests[0]) == [
        "<conversation-summary>\ndurable-summary\n</conversation-summary>",
        "question-three",
        "answer-three",
        "question-four",
        "answer-four",
        "question-five",
    ]
    assert _text(transcript)[-2:] == ["question-five", "answer-five"]


@pytest.mark.asyncio
async def test_sdk_fork_inherits_compaction_at_matching_logical_leaf(
    node_store_backend: _NodeStoreBackend,
    tmp_path: Path,
) -> None:
    provider = _StubProvider(
        "answer-one",
        "answer-two",
        "answer-three",
        "shared-summary",
        "parent-answer",
        "branch-answer",
    )
    resources = LocalResourceStore(
        workspace_root=tmp_path,
        root=tmp_path / "fork-resources",
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[
                (
                    _LLM_COMPACTION,
                    {
                        "max_messages": 5,
                        "keep_last_turns": 1,
                    },
                )
            ],
            stream_fn=provider,
            model=_model(),
            store=JsonlTrajectoryStore(tmp_path / "fork-compaction-turns"),
            trajectory_node_store=node_store_backend.open(),
            resource_store=resources,
            resource_writer=resources,
        )
    )
    forked: AgentSession | None = None
    try:
        await session.run("question-one")
        await session.run("question-two")
        await session.run("question-three")
        await session.run("parent-only-question")
        forked = await AgentSession.fork(session, at=2, purpose="alternate")
        transcript = await forked.run("branch-only-question")
    finally:
        if forked is not None:
            await forked.shutdown()
        await session.shutdown()

    assert len(provider.requests) == 6
    assert _text(provider.requests[5]) == [
        "<conversation-summary>\nshared-summary\n</conversation-summary>",
        "question-three",
        "answer-three",
        "branch-only-question",
    ]
    assert _text(transcript)[-2:] == [
        "branch-only-question",
        "branch-answer",
    ]


@pytest.mark.asyncio
async def test_sdk_interrupt_cancels_compaction_and_session_continues(
    tmp_path: Path,
) -> None:
    provider = _StubProvider(
        "answer-one",
        "answer-two",
        _WAIT_FOR_CANCEL,
        "durable-summary",
        "continued-answer",
    )
    resources = LocalResourceStore(
        workspace_root=tmp_path,
        root=tmp_path / "cancel-resources",
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[
                (
                    _LLM_COMPACTION,
                    {
                        "max_messages": 3,
                        "keep_last_turns": 1,
                    },
                )
            ],
            stream_fn=provider,
            model=_model(),
            store=JsonlTrajectoryStore(tmp_path / "cancel-turns"),
            trajectory_node_store=JsonlTrajectoryNodeStore(
                tmp_path / "cancel-nodes"
            ),
            resource_store=resources,
            resource_writer=resources,
        )
    )
    interrupted: asyncio.Task[list[AgentMessage]] | None = None
    try:
        await session.run("question-one")
        await session.run("question-two")
        interrupted = asyncio.create_task(session.run("question-three"))
        await asyncio.wait_for(provider.stream_started.wait(), timeout=2.0)
        session.interrupt("user_cancel")
        await asyncio.wait_for(interrupted, timeout=2.0)
        transcript = await session.run("question-after-interrupt")
    finally:
        if interrupted is not None and not interrupted.done():
            interrupted.cancel()
        await session.shutdown()

    assert provider.observed_cancel_reason == "user_cancel"
    assert _text(provider.requests[3]) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
    ]
    assert _text(transcript)[-2:] == [
        "question-after-interrupt",
        "continued-answer",
    ]


class _EmptyOpenAIStream:
    def __aiter__(self) -> "_EmptyOpenAIStream":
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration

    async def close(self) -> None:
        return None


class _OpenAICompletionsStub:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def create(self, **body: Any) -> _EmptyOpenAIStream:
        self.requests.append(body)
        return _EmptyOpenAIStream()


class _OpenAIClientStub:
    def __init__(self) -> None:
        self.completions = _OpenAICompletionsStub()
        self.chat = type("_Chat", (), {"completions": self.completions})()


@pytest.mark.asyncio
async def test_openai_provider_materializes_prompt_cache_request_fields() -> None:
    model = Model(
        id="gpt-5-mini",
        provider="openai",
        context_window=128_000,
        max_output_tokens=1_024,
    )
    adapter = OpenAIPromptCacheAdapter(retention="24h")
    adapted = adapter.apply_prompt_cache(
        ProviderPromptCacheRequest(
            messages=[text_message("cached-prefix")],
            model=model,
            state=PromptCacheState(cache_key="stable-sdk-session"),
        )
    )
    client = _OpenAIClientStub()
    stream_fn = OpenAIStreamFn(client=client)

    _ = [
        event
        async for event in stream_fn(
            messages=list(adapted.messages),
            model=model,
            tools=[],
        )
    ]

    assert client.completions.requests[0]["prompt_cache_key"] == (
        "stable-sdk-session"
    )
    assert client.completions.requests[0]["prompt_cache_retention"] == "24h"
