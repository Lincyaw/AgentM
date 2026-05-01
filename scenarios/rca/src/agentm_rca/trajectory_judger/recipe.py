"""Recipe for the consumer-side trajectory_judger scenario."""

from __future__ import annotations

from typing import Any

SCENARIO_NAME = "agentm_rca.scenario.trajectory_judger"


def load() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("agentm_rca.trajectory_judger.extension", {}),
        ("agentm_rca.trajectory_judger.reader", {}),
    ]
