"""Behavior contracts spanning trajectory and observability backends."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os
from pathlib import Path
import time
import uuid

import pytest

from agentm import AgentSession, AgentSessionConfig, Model
from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
)
from agentm.core.abi.query import EventRecord, SpanRecord
from agentm.core.abi.stream import MessageEnd, TextDelta
from agentm.core.abi.tool import Tool
from agentm.core.abi.trigger import UserInput
from agentm.core.runtime.stores.query import TrajectoryStoreQueryAdapter
from agentm.storage.query import (
    ClickHouseObservabilityQueryStore,
    CompositeTraceQueryStore,
)
from agentm.storage.trajectory import JsonlTrajectoryStore

_OBSERVABILITY = "agentm.extensions.builtin.observability"


class _StubProvider:
    """One-response provider double at the public StreamFn boundary."""

    async def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: str = "off",
    ) -> AsyncIterator[TextDelta | MessageEnd]:
        del messages, model, tools, system, signal, thinking
        message = AssistantMessage(
            role="assistant",
            content=(TextContent(type="text", text="observed-answer"),),
            timestamp=time.time(),
            stop_reason="end_turn",
        )
        yield TextDelta(text="observed-answer")
        yield MessageEnd(message=message)


@pytest.mark.asyncio
async def test_sdk_composes_one_trajectory_store_with_otlp_clickhouse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clickhouse_url = os.environ.get("AGENTM_TEST_CLICKHOUSE_URL")
    otlp_endpoint = os.environ.get("AGENTM_TEST_OTLP_ENDPOINT")
    if not clickhouse_url or not otlp_endpoint:
        pytest.skip(
            "set AGENTM_TEST_CLICKHOUSE_URL and AGENTM_TEST_OTLP_ENDPOINT "
            "to run the ClickHouse OTLP behavior contract"
        )
    clickhouse_connect = pytest.importorskip("clickhouse_connect")
    from agentm.extensions.observability.otel_export import (
        shutdown_process_telemetry,
    )

    shutdown_process_telemetry()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", otlp_endpoint)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "true")
    session_id = f"clickhouse-e2e-{uuid.uuid4().hex}"
    store_path = tmp_path / "trajectory"
    session = await AgentSession.create(
        AgentSessionConfig(
            session_id=session_id,
            purpose="root",
            extensions=[
                (
                    _OBSERVABILITY,
                    {
                        "export": "otlp",
                        "include_handler_records": False,
                    },
                )
            ],
            stream_fn=_StubProvider(),
            model=Model(
                id="stub-model",
                provider="stub",
                context_window=128_000,
                max_output_tokens=4_096,
            ),
            trajectory_store=JsonlTrajectoryStore(store_path),
        )
    )
    try:
        transcript = await session.run("observed-question")
    finally:
        await session.shutdown()
        shutdown_process_telemetry()

    assert [
        block.text
        for message in transcript
        for block in message.content
        if isinstance(block, TextContent)
    ] == ["observed-question", "observed-answer"]

    client = clickhouse_connect.get_client(dsn=clickhouse_url)
    query = CompositeTraceQueryStore(
        TrajectoryStoreQueryAdapter(JsonlTrajectoryStore(store_path)),
        ClickHouseObservabilityQueryStore(client),
    )
    try:
        sessions = list(query.sessions())
        turns = list(query.turns(session_id))
        events, spans = await _wait_for_observability(query, session_id)
    finally:
        client.close()
        shutdown_process_telemetry()

    assert [row.id for row in sessions] == [session_id]
    assert len(turns) == 1
    assert isinstance(turns[0].trigger, UserInput)
    assert [
        block.text
        for block in turns[0].trigger.content
        if isinstance(block, TextContent)
    ] == ["observed-question"]
    assert "agentm.session.start" in {event.name for event in events}
    assert "agentm.turn.committed" in {event.name for event in events}
    assert "agentm.session.end" in {event.name for event in events}
    span_names = {span.name for span in spans}
    assert "agentm.turn" in span_names
    assert "chat stub-model" in span_names


async def _wait_for_observability(
    query: CompositeTraceQueryStore,
    session_id: str,
) -> tuple[list[EventRecord], list[SpanRecord]]:
    for _ in range(30):
        events = list(query.events(session_id))
        spans = list(query.spans(session_id))
        if events and spans:
            return events, spans
        await asyncio.sleep(0.5)
    raise AssertionError(
        f"OTLP collector did not persist session {session_id!r} in time"
    )
