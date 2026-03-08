"""Phase management for StateGraph-based agent systems.

PhaseManager.__init__ and transition_to contain real logic (value object behavior).
from_config is a stub — raise NotImplementedError.
"""

from __future__ import annotations

from agentm.models.data import PhaseDefinition


class PhaseManager:
    """Manages phase transitions for StateGraph-based systems."""

    def __init__(self, phases: dict[str, PhaseDefinition], initial_phase: str) -> None:
        if initial_phase not in phases:
            raise ValueError(f"Initial phase '{initial_phase}' not in phases: {list(phases.keys())}")
        self.phases = phases
        self.current_phase = initial_phase

    def transition_to(self, next_phase: str) -> None:
        """Transition to the next phase. Validates the transition is allowed."""
        current_def = self.phases.get(self.current_phase)
        if current_def is None:
            raise ValueError(f"Current phase '{self.current_phase}' not found in phases")
        if next_phase not in current_def.next_phases:
            raise ValueError(
                f"Invalid transition: {self.current_phase} -> {next_phase}. "
                f"Allowed: {current_def.next_phases}"
            )
        self.current_phase = next_phase

    @classmethod
    def from_config(cls, config: dict) -> PhaseManager:
        """Build PhaseManager from YAML config dict.

        Expected format:
            {
                "exploration": {"next_phases": ["generation"]},
                "generation": {"next_phases": ["verification"]},
                ...
            }

        The initial_phase is the first key in the dict.
        Raises ValueError if config is empty.
        """
        if not config:
            raise ValueError("Phase config must not be empty")

        phases: dict[str, PhaseDefinition] = {}
        for phase_name, phase_data in config.items():
            next_phases = phase_data.get("next_phases", [])
            phases[phase_name] = PhaseDefinition(
                name=phase_name,
                description=phase_data.get("description", ""),
                handler=None,
                next_phases=next_phases,
            )

        initial_phase = next(iter(config))
        return cls(phases=phases, initial_phase=initial_phase)
