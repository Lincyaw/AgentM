"""Tests for ``AgentMAgent`` runtime-injecting the ``distill_binding`` atom.

Fail-stop position: harness.sync rollouts must auto-emit a meta sidecar so
``llmharness-distill export`` can pair the replay log to ground truth. If
this wiring breaks, every harness.sync run silently produces unjoinable
training data — the failure mode the binding atom exists to prevent.

We assert on the ``AgentSessionConfig.extra_extensions`` that
``AgentMAgent.run`` constructs, not on disk side effects: the binding atom
itself has its own unit tests in the llmharness package, and the only thing
this adapter contributes is the mount + sample_id derivation.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentm_rca.eval.agent import AgentMAgent


def _drive_run(
    *,
    scenario: str,
    data_dir: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Invoke ``AgentMAgent.run`` with the network calls stubbed out and
    return the ``extra_extensions`` that landed on the spawned
    ``AgentSessionConfig``. The rest of the run path is intentionally
    mocked to nothing — this test only cares about the runtime mount."""

    agent = AgentMAgent(scenario=scenario, provider="anthropic", model="stub")

    captured: dict[str, Any] = {}

    fake_session = MagicMock()
    fake_session.session_id = "sid"
    fake_session.root_session_id = "rid"
    fake_session.prompt = AsyncMock(return_value=[])
    fake_session.shutdown = AsyncMock(return_value=None)

    async def fake_create(_cls: Any, config: Any) -> Any:
        captured["config"] = config
        return fake_session

    with (
        patch(
            "agentm.core.runtime.session_factory.create_agent_session",
            side_effect=fake_create,
        ),
        patch("agentm.core.runtime.session.AgentSession", MagicMock()),
    ):
        asyncio.run(agent.run(incident="ignored", data_dir=data_dir))

    config = captured["config"]
    return list(config.extra_extensions)


def test_harness_sync_scenario_mounts_distill_binding() -> None:
    """When the scenario is harness.sync, the binding atom must be mounted
    with ``sample_id`` derived from the data_dir basename."""

    extra = _drive_run(
        scenario="rca:harness.sync",
        data_dir="/tmp/eval/data_28c9448b",
    )

    bindings = [e for e in extra if e[0] == "llmharness.distill.binding"]
    assert len(bindings) == 1, f"expected one binding atom, got {extra!r}"
    module, config = bindings[0]
    assert module == "llmharness.distill.binding"
    assert config["sample_id"] == "data_28c9448b"
    assert config["dataset_path"] == "/tmp/eval/data_28c9448b"




def test_baseline_scenario_does_not_mount_binding() -> None:
    """Without the harness adapter the replay sidecar is never produced,
    so binding would write a meta sidecar with no companion — dead
    weight. Mount must be skipped."""

    extra = _drive_run(
        scenario="rca:baseline",
        data_dir="/tmp/eval/data_xyz",
    )

    assert not any(e[0] == "llmharness.distill.binding" for e in extra)






if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
