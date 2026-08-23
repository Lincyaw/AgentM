"""The loop budget an atom declares has to be the one the loop runs under."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core.abi.session_api import ExtensionSpec, LoopConfig
from agentm.testing import probe_session


def test_two_budgets_compose_as_the_lower_ceiling() -> None:
    assert LoopConfig(max_turns=5).tightened_with(
        LoopConfig(max_turns=9)
    ) == LoopConfig(max_turns=5)
    assert LoopConfig(max_turns=9).tightened_with(
        LoopConfig(max_turns=5)
    ) == LoopConfig(max_turns=5)
    assert LoopConfig().tightened_with(LoopConfig(max_tool_calls=2)) == LoopConfig(
        max_tool_calls=2
    )


@pytest.mark.asyncio
async def test_a_declared_budget_reaches_the_driver(tmp_path: Path) -> None:
    """A scenario that caps its own loop is the only stop such a run has.

    The atom registered its budget as a service that the driver never read, so
    a scenario declaring ``max_turns`` ran unbounded and looked configured.
    """

    spec = ExtensionSpec.from_module(
        "agentm.extensions.builtin.loop_budget",
        {"max_turns": 3},
    )
    async with probe_session(str(tmp_path), extensions=[spec]) as session:
        assert session._loop_budget() == LoopConfig(max_turns=3)
