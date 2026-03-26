"""Scenario protocol, wiring dataclass, and module-level registry.

A Scenario encapsulates all domain-specific behavior that a harness needs
to run a particular use-case: tools, schemas, hooks, and context formatting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from agentm.harness.tool import Tool
    from agentm.models.data import OrchestratorHooks


# ---------------------------------------------------------------------------
# Helper for default format_context
# ---------------------------------------------------------------------------

def _empty_context() -> str:
    """Return an empty string. Used as default for ScenarioWiring.format_context."""
    return ""


# ---------------------------------------------------------------------------
# SetupContext — platform resources provided to a scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SetupContext:
    """Resources the harness provides to a scenario during setup."""

    vault: Any | None
    trajectory: Any | None
    tool_registry: Any


# ---------------------------------------------------------------------------
# ScenarioWiring — everything a scenario returns to the harness
# ---------------------------------------------------------------------------

@dataclass
class ScenarioWiring:
    """All wiring a scenario returns to the harness."""

    # Tools
    orchestrator_tools: list[Tool] = field(default_factory=list)
    worker_tools: list[Tool] = field(default_factory=list)

    # Dynamic context — always zero-arg; scenario binds its own state via closures.
    format_context: Callable[[], str] = field(default_factory=lambda: _empty_context)

    # Structured output
    answer_schemas: dict[str, type[BaseModel]] = field(default_factory=dict)
    output_schema: type[BaseModel] | None = None

    # Behavior customization
    hooks: OrchestratorHooks = field(default=None)  # type: ignore[assignment]

    # Termination logic (None = default <decision> tag parser)
    should_terminate: Callable[[Any], bool] | None = None

    # Scenario-specific middleware (appended after SDK middleware)
    orchestrator_middleware: list[Any] = field(default_factory=list)
    worker_middleware: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.hooks is None:
            from agentm.models.data import OrchestratorHooks as _Hooks

            object.__setattr__(self, "hooks", _Hooks())


# ---------------------------------------------------------------------------
# Scenario protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Scenario(Protocol):
    """A scenario provides domain-specific behavior to the harness.

    Only two things required: a name, and a setup() that returns wiring.
    """

    @property
    def name(self) -> str: ...

    def setup(self, ctx: SetupContext) -> ScenarioWiring: ...


# ---------------------------------------------------------------------------
# ScenarioRegistry — module-level functions
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, Scenario] = {}


def register_scenario(scenario: Scenario) -> None:
    """Register a scenario by its name."""
    _SCENARIOS[scenario.name] = scenario


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name. Raises ValueError if not found."""
    if name not in _SCENARIOS:
        available = ", ".join(sorted(_SCENARIOS)) or "(none)"
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {available}"
        )
    return _SCENARIOS[name]


def list_scenarios() -> list[str]:
    """Return all registered scenario names."""
    return list(_SCENARIOS)
