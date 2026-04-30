from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core.kernel import text_message
from tests.unit.extensions.builtin import _helpers
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_micro_compact_compacts_and_records_entry(tmp_path: Path) -> None:
    seen_before: list[object] = []
    seen_after: list[object] = []
    initial_messages = [text_message("x" * 200, timestamp=float(i)) for i in range(10)]

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (
                    "agentm.extensions.builtin.micro_compact",
                    {"threshold_pct": 0.85, "keep_last": 3},
                ),
            ],
            provider=(
                "tests.unit.extensions.builtin._helpers",
                {"context_window": 1000},
            ),
            initial_messages=initial_messages,
            resource_loader=InMemoryResourceLoader(),
        )
    )
    session.bus.on("before_compact", lambda event: seen_before.append(event))
    session.bus.on("after_compact", lambda event: seen_after.append(event))

    await session.prompt("hello")

    assert len(seen_before) == 1
    assert len(seen_after) == 1
    assert _helpers.LAST_STREAM is not None
    sent_messages = _helpers.LAST_STREAM.seen_messages[-1]
    assert len(sent_messages) < len(initial_messages) + 1

    branch = session.session_manager.get_active_branch()
    compaction_entries = [entry for entry in branch if entry.type == "compaction"]
    assert len(compaction_entries) == 1
    assert compaction_entries[0].id != ""
    payload = compaction_entries[0].payload
    assert payload["entry_id"] != ""
    await session.shutdown()
