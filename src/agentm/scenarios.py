"""Packaged scenario helpers outside the SDK core runtime."""

from __future__ import annotations

from agentm.core.abi.session_api import ExtensionSpec, ScenarioSpec


_PACKAGED_SCENARIOS: dict[str, ScenarioSpec] = {
    "empty": ScenarioSpec(extensions=()),
    "minimal": ScenarioSpec(
        extensions=(
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.operations", {}),
            ("agentm.extensions.builtin.retry_policy", {}),
            ("agentm.extensions.builtin.tool_result_cap", {}),
            ("agentm.extensions.builtin.tool_error_messages", {}),
            ("agentm.extensions.builtin.file_tools", {}),
            ("agentm.extensions.builtin.tool_bash", {}),
            ("agentm.extensions.builtin.system_prompt", {}),
        ),
    ),
}


def packaged_scenario_names() -> tuple[str, ...]:
    """Return scenario names bundled with the SDK distribution."""

    return tuple(sorted(_PACKAGED_SCENARIOS))


def builtin_scenario_loader(scenario: str) -> ScenarioSpec:
    """Resolve a packaged scenario name to extension specs.

    The core runtime factory intentionally does not own this registry. Hosts
    that want named packaged scenarios pass this function as
    ``AgentSessionConfig.scenario_loader`` or ``create_session(...,
    scenario_loader=...)``.
    """

    try:
        spec = _PACKAGED_SCENARIOS[scenario]
    except KeyError as exc:
        known = ", ".join(packaged_scenario_names())
        raise ValueError(f"unknown packaged scenario {scenario!r}; known: {known}") from exc
    extensions: tuple[ExtensionSpec, ...] = tuple(
        (module, dict(config)) for module, config in spec.extensions
    )
    return ScenarioSpec(extensions=extensions, base_dir=spec.base_dir)


__all__ = [
    "builtin_scenario_loader",
    "packaged_scenario_names",
]
