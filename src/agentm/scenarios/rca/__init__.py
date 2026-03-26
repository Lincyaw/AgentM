"""Hypothesis-driven Root Cause Analysis scenario."""


def register() -> None:
    """Register RCA scenario with the SDK registry."""
    from agentm.harness.scenario import register_scenario
    from agentm.scenarios.rca.scenario import RCAScenario

    register_scenario(RCAScenario())
