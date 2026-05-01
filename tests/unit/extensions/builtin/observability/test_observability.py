from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


@pytest.mark.asyncio
async def test_observability_streams_lifecycle_install_dispatch_and_handlers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "obs.jsonl"
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.observability", {"path": str(output)}),
                (
                    "agentm.extensions.builtin.trajectory",
                    {"path": str(tmp_path / "traj.jsonl"), "channels": ["agent_end"]},
                ),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    await session.prompt("hi")
    await session.shutdown()

    records = _read_records(output)
    kinds = [r["kind"] for r in records]

    assert kinds[0] == "session.start"
    assert "session.end" in kinds
    assert "session.ready" in kinds
    # The trajectory and provider extension installs are recorded.
    install_records = [r for r in records if r["kind"] == "extension.install"]
    install_modules = {
        r["attributes"]["module_path"] for r in install_records if r["attributes"]["phase"] == "end"
    }
    assert "agentm.extensions.builtin.trajectory" in install_modules
    assert "tests.unit.extensions.builtin._helpers" in install_modules

    # Every record carries the same trace_id.
    trace_ids = {r["trace_id"] for r in records}
    assert len(trace_ids) == 1

    # Dispatch records exist for kernel events fired during the prompt.
    dispatch_channels = {
        r["attributes"]["channel"] for r in records if r["kind"] == "event.dispatch"
    }
    assert "agent_start" in dispatch_channels
    assert "agent_end" in dispatch_channels

    # Handler records carry the channel and an extension attribution.
    handler_records = [r for r in records if r["kind"] == "handler.invoke"]
    assert handler_records, "expected per-handler records"
    sample = handler_records[0]
    assert "extension" in sample["attributes"]


@pytest.mark.asyncio
async def test_observability_handler_records_can_be_disabled(tmp_path: Path) -> None:
    output = tmp_path / "obs.jsonl"
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (
                    "agentm.extensions.builtin.observability",
                    {"path": str(output), "include_handler_records": False},
                ),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    await session.prompt("hi")
    await session.shutdown()

    records = _read_records(output)
    assert not any(r["kind"] == "handler.invoke" for r in records)
    # Dispatch records should still be present.
    assert any(r["kind"] == "event.dispatch" for r in records)


@pytest.mark.asyncio
async def test_observability_attributes_handlers_to_their_extension(tmp_path: Path) -> None:
    output = tmp_path / "obs.jsonl"
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.observability", {"path": str(output)}),
                (
                    "agentm.extensions.builtin.trajectory",
                    {"path": str(tmp_path / "traj.jsonl"), "channels": ["agent_start"]},
                ),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    await session.prompt("hi")
    await session.shutdown()

    records = _read_records(output)
    handlers_for_agent_start = [
        r
        for r in records
        if r["kind"] == "handler.invoke"
        and r["attributes"]["channel"] == "agent_start"
    ]
    extensions = {r["attributes"]["extension"] for r in handlers_for_agent_start}
    assert "agentm.extensions.builtin.trajectory" in extensions


@pytest.mark.asyncio
async def test_observability_records_api_register_and_llm_request_via_bus(
    tmp_path: Path,
) -> None:
    output = tmp_path / "obs.jsonl"
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.observability", {"path": str(output)}),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    await session.prompt("hi")
    await session.shutdown()

    records = _read_records(output)
    # api.register flowed through the bus (provider registration).
    api_kinds = {r["attributes"]["kind"] for r in records if r["kind"] == "api.register"}
    assert "provider" in api_kinds
    # LLM request lifecycle flowed through the bus.
    assert any(r["kind"] == "llm.request.start" for r in records)
    assert any(r["kind"] == "llm.request.end" for r in records)
    # turn.summary contains stop_reason for the assistant message.
    summaries = [r for r in records if r["kind"] == "turn.summary"]
    assert summaries and summaries[0]["attributes"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_observability_diffs_only_mutable_channels(tmp_path: Path) -> None:
    output = tmp_path / "obs.jsonl"
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.observability", {"path": str(output)}),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    await session.prompt("hi")
    await session.shutdown()

    records = _read_records(output)
    mutated = [r for r in records if r["kind"] == "handler.mutated"]
    # No handler.mutated should reference an immutable channel (e.g. agent_end).
    for r in mutated:
        assert r["attributes"]["channel"] in {
            "before_agent_start",
            "context",
            "tool_call",
            "tool_result",
            "before_compact",
        }


def test_event_bus_strict_sync_raises_on_async_handler() -> None:
    from agentm.core.kernel import EventBus

    bus = EventBus()
    bus.set_strict_sync(True)

    async def handler(event: object) -> None:
        return None

    bus.on("api_register", handler)
    with pytest.raises(RuntimeError, match="sync-only channel"):
        bus.emit_sync("api_register", object())
