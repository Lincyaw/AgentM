"""Scenario-evaluator protocol for rescue-window branch results."""

from __future__ import annotations

from typing import Any, Protocol

from agentm.core.abi import AgentMessage

from .runner import BranchResult


class ScenarioEvaluator(Protocol):
    """Scenario-owned scoring adapter.

    Generic rescue-window code produces branch rollouts. Scenario packages own
    final-output extraction, judges, and domain-specific metrics.
    """

    async def evaluate(
        self,
        *,
        branch: BranchResult,
        final_messages: list[AgentMessage],
    ) -> dict[str, Any]:
        ...
