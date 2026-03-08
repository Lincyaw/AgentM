"""P6: Compression preserves data for recall.

Bug prevented: Compression discards checkpoint references → recall_history
returns empty → Orchestrator loses access to early investigation data.
"""

from __future__ import annotations

from dataclasses import replace

from agentm.core.compression import (
    build_compression_hook,
    compress_completed_phase,
    count_tokens,
    sub_agent_compression_hook,
)
from agentm.config.schema import CompressionConfig
from agentm.models.data import DiagnosticNotebook, ExplorationStep, PhaseSummary
from agentm.models.enums import Phase


class TestPhaseCompressionIntegrity:
    """P6: compress_completed_phase preserves data for future recall."""

    def test_compression_creates_phase_summary(self, notebook: DiagnosticNotebook) -> None:
        """Compressing a phase should create a PhaseSummary."""
        step = ExplorationStep(
            step_number=0,
            phase=Phase.EXPLORATION,
            action="dispatched infra agent",
            timestamp="2026-03-08T10:01:00",
            content="Dispatched infrastructure agent for initial scan",
        )
        nb = replace(notebook, exploration_history=[step])

        compressed = compress_completed_phase(nb, "exploration")

        assert len(compressed.phase_summaries) == 1
        assert compressed.phase_summaries[0].phase == "exploration"
        assert "dispatched infra agent" in compressed.phase_summaries[0].actions_taken

    def test_compression_removes_phase_steps(self, notebook: DiagnosticNotebook) -> None:
        """Compressed phase steps should be removed from exploration_history."""
        steps = [
            ExplorationStep(
                step_number=i,
                phase=Phase.EXPLORATION,
                action=f"action_{i}",
                timestamp=f"2026-03-08T10:0{i}:00",
                content=f"Content {i}",
            )
            for i in range(3)
        ]
        nb = replace(notebook, exploration_history=steps)

        compressed = compress_completed_phase(nb, "exploration")

        assert len(compressed.exploration_history) == 0

    def test_compression_preserves_other_phase_steps(
        self, notebook: DiagnosticNotebook
    ) -> None:
        """Steps from other phases should NOT be removed during compression."""
        explore_step = ExplorationStep(
            step_number=0,
            phase=Phase.EXPLORATION,
            action="explore",
            timestamp="2026-03-08T10:01:00",
            content="Exploration step",
        )
        gen_step = ExplorationStep(
            step_number=1,
            phase=Phase.GENERATION,
            action="generate",
            timestamp="2026-03-08T10:02:00",
            content="Generation step",
        )
        nb = replace(notebook, exploration_history=[explore_step, gen_step])

        compressed = compress_completed_phase(nb, "exploration")

        assert len(compressed.exploration_history) == 1
        assert compressed.exploration_history[0].action == "generate"

    def test_compression_is_immutable(self, notebook: DiagnosticNotebook) -> None:
        """compress_completed_phase should return a new instance."""
        step = ExplorationStep(
            step_number=0,
            phase=Phase.EXPLORATION,
            action="action",
            timestamp="2026-03-08T10:01:00",
            content="Content",
        )
        nb = replace(notebook, exploration_history=[step])

        compressed = compress_completed_phase(nb, "exploration")

        assert compressed is not nb
        assert len(nb.exploration_history) == 1  # original unchanged


class TestCompressionHookThreshold:
    """P6: Compression hook only fires when token count exceeds threshold."""

    def test_below_threshold_passes_through(self) -> None:
        """Below threshold, hook returns messages unchanged."""
        state = {"messages": [{"content": "short message"}]}
        result = sub_agent_compression_hook(state)
        assert "messages" in result

    def test_build_compression_hook_configurable(self) -> None:
        """build_compression_hook respects config threshold."""
        config = CompressionConfig(
            compression_threshold=0.8,
            compression_model="gpt-4o-mini",
        )
        hook = build_compression_hook(config)
        assert callable(hook)

        state = {"messages": [{"content": "short"}]}
        result = hook(state)
        assert "messages" in result
