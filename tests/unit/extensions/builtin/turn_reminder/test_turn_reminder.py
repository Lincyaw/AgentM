from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core.kernel import AgentMessage, text_message
from tests.unit.extensions.builtin import _helpers
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_count", "should_remind"),
    [(2, True), (5, True), (8, True), (1, False), (4, False)],
)
async def test_turn_reminder_fires_every_n_turns(
    tmp_path: Path,
    initial_count: int,
    should_remind: bool,
) -> None:
    initial_messages: list[AgentMessage] = [
        text_message(f"m{i}", timestamp=float(i)) for i in range(initial_count)
    ]
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (
                    "agentm.extensions.builtin.turn_reminder",
                    {"reminder": "REMEMBER", "every_n_turns": 3},
                ),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            initial_messages=initial_messages,
            resource_loader=InMemoryResourceLoader(),
        )
    )

    await session.prompt("hello")

    assert _helpers.LAST_STREAM is not None
    seen_system = _helpers.LAST_STREAM.seen_systems[-1]
    if should_remind:
        assert seen_system == "REMEMBER"
    else:
        assert seen_system == ""
    await session.shutdown()
