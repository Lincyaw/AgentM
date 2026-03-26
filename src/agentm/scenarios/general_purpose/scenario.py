"""General-purpose Scenario implementation using the Scenario protocol.

Minimal wiring: only answer_schemas and default hooks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentm.harness.scenario import ScenarioWiring, SetupContext


class GeneralPurposeScenario:
    """Scenario implementation for general-purpose task execution."""

    @property
    def name(self) -> str:
        return "general_purpose"

    def setup(self, ctx: SetupContext) -> ScenarioWiring:
        """Wire up GP scenario: only answer schemas needed."""
        from agentm.harness.scenario import ScenarioWiring
        from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer

        return ScenarioWiring(
            answer_schemas={"execute": GeneralAnswer},
        )
