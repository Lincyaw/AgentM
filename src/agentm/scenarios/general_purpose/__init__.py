"""General-purpose task execution scenario."""


def register() -> None:
    """Register general-purpose scenario with the SDK registry."""
    from agentm.harness.scenario import register_scenario
    from agentm.scenarios.general_purpose.scenario import GeneralPurposeScenario

    register_scenario(GeneralPurposeScenario())
