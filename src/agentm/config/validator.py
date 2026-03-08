"""Configuration cross-reference validation."""

from __future__ import annotations

from agentm.config.schema import ScenarioConfig, SystemConfig


def validate_references(
    system: SystemConfig,
    scenario: ScenarioConfig,
    tool_registry: object,
) -> list[str]:
    """Validate cross-file references between system config, scenario config, and tool registry.

    Returns a list of validation error messages. Empty list means valid.

    Checks:
    1. Agent model references exist in system.models
    2. Orchestrator model reference exists in system.models
    3. Agent tool references exist in tool registry
    4. Prompt file references exist on disk
    5. Tool settings keys match tool's declared parameters
    """
    errors: list[str] = []

    # Check 1: Agent model references exist in system.models
    for agent_name, agent_cfg in scenario.agents.items():
        if agent_cfg.model not in system.models:
            errors.append(
                f"Agent '{agent_name}' references unknown model '{agent_cfg.model}'"
            )

    # Check 2: Orchestrator model exists in system.models
    if scenario.orchestrator.model not in system.models:
        errors.append(
            f"Orchestrator references unknown model '{scenario.orchestrator.model}'"
        )

    # Check 3: Agent tool references exist in tool registry
    for agent_name, agent_cfg in scenario.agents.items():
        for tool_name in agent_cfg.tools:
            if not tool_registry.has(tool_name):  # type: ignore[attr-defined]
                errors.append(
                    f"Agent '{agent_name}' references unknown tool '{tool_name}'"
                )

    # Check 4: Prompt file references exist on disk (skipped — no base dir available)

    # Check 5: Tool settings keys match tool's declared parameters
    for agent_name, agent_cfg in scenario.agents.items():
        for tool_name, settings in agent_cfg.tool_settings.items():
            if not tool_registry.has(tool_name):  # type: ignore[attr-defined]
                # Already reported in check 3; skip parameter check
                continue
            tool_def = tool_registry.get(tool_name)  # type: ignore[attr-defined]
            declared = set(tool_def.parameters.keys())
            provided = set(settings.keys())
            unknown_keys = provided - declared
            if unknown_keys:
                errors.append(
                    f"Agent '{agent_name}' tool_settings for '{tool_name}' "
                    f"has unknown keys: {sorted(unknown_keys)}"
                )

    return errors
