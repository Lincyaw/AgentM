"""Tests for Notebook compression integration in the prompt pipeline.

Bug prevented: completed-phase exploration steps remain in the formatted prompt
indefinitely, bloating LLM context and wasting tokens. Compression replaces
detailed step records with a compact PhaseSummary before formatting.

Ref: designs/generic-state-wrapper.md -- Layer 2 Orchestrator compression
"""

from __future__ import annotations

from agentm.core.compression import compress_completed_phase
from agentm.core.notebook import format_notebook_for_llm, should_compress_phase
from agentm.models.data import DiagnosticNotebook, ExplorationStep
from agentm.models.enums import Phase


def _make_notebook_with_history() -> DiagnosticNotebook:
    """Create a notebook with exploration history across two phases."""
    return DiagnosticNotebook(
        task_id="test-task",
        task_description="Test RCA",
        start_time="2026-03-09T10:00:00",
        exploration_history=[
            ExplorationStep(
                step_number=1,
                phase=Phase.EXPLORATION,
                action="Scanned infrastructure",
                timestamp="2026-03-09T10:01:00",
                content="CPU 85%",
            ),
            ExplorationStep(
                step_number=2,
                phase=Phase.EXPLORATION,
                action="Checked database",
                timestamp="2026-03-09T10:02:00",
                content="Pool full",
            ),
            ExplorationStep(
                step_number=3,
                phase=Phase.GENERATION,
                action="Formed H1",
                timestamp="2026-03-09T10:05:00",
                content="Connection pool exhaustion",
            ),
        ],
        current_phase=Phase.GENERATION,
    )


class TestShouldCompressPhase:
    """should_compress_phase correctly identifies compressible phases.

    Bug prevented: compressing the current phase or an already-compressed phase
    would either lose active data or duplicate summaries.
    """

    def test_completed_phase_with_steps_is_compressible(self):
        notebook = _make_notebook_with_history()
        assert should_compress_phase(notebook, "exploration") is True

    def test_current_phase_is_not_compressible(self):
        notebook = _make_notebook_with_history()
        assert should_compress_phase(notebook, "generation") is False

    def test_already_compressed_phase_is_not_compressible(self):
        notebook = _make_notebook_with_history()
        compressed = compress_completed_phase(notebook, "exploration")
        assert should_compress_phase(compressed, "exploration") is False

    def test_phase_without_steps_is_not_compressible(self):
        notebook = _make_notebook_with_history()
        assert should_compress_phase(notebook, "verification") is False


class TestCompressThenFormat:
    """After compression, format_notebook_for_llm includes the PhaseSummary
    and excludes the original detailed exploration steps for that phase.

    Bug prevented: compressed phase steps still appear in formatted output,
    negating the token savings from compression.
    """

    def test_compressed_phase_appears_as_summary(self):
        notebook = _make_notebook_with_history()
        compressed = compress_completed_phase(notebook, "exploration")

        assert len(compressed.phase_summaries) == 1
        assert compressed.phase_summaries[0].phase == "exploration"

        formatted = format_notebook_for_llm(compressed)
        assert "Phase Summaries" in formatted
        assert "exploration" in formatted

    def test_compressed_phase_steps_removed_from_history(self):
        notebook = _make_notebook_with_history()
        compressed = compress_completed_phase(notebook, "exploration")

        assert all(
            step.phase != Phase.EXPLORATION for step in compressed.exploration_history
        )
        assert any(
            step.phase == Phase.GENERATION for step in compressed.exploration_history
        )

    def test_formatted_output_contains_active_phase_steps(self):
        notebook = _make_notebook_with_history()
        compressed = compress_completed_phase(notebook, "exploration")
        formatted = format_notebook_for_llm(compressed)

        assert "Formed H1" in formatted

    def test_formatted_output_excludes_compressed_step_details(self):
        notebook = _make_notebook_with_history()
        compressed = compress_completed_phase(notebook, "exploration")
        formatted = format_notebook_for_llm(compressed)

        # The detailed step content for exploration should not appear as
        # individual exploration history entries (only in the summary)
        assert "Step 1 [exploration]" not in formatted
        assert "Step 2 [exploration]" not in formatted


class TestCompressionImmutability:
    """Compression returns a new notebook instance; original is unchanged.

    Bug prevented: compressing for LLM formatting mutates the state notebook,
    causing the actual graph state to lose exploration history permanently.
    """

    def test_original_notebook_unchanged(self):
        notebook = _make_notebook_with_history()
        original_history_len = len(notebook.exploration_history)

        compressed = compress_completed_phase(notebook, "exploration")

        assert len(notebook.exploration_history) == original_history_len
        assert len(compressed.exploration_history) < original_history_len

    def test_original_phase_summaries_unchanged(self):
        notebook = _make_notebook_with_history()
        assert len(notebook.phase_summaries) == 0

        compress_completed_phase(notebook, "exploration")

        assert len(notebook.phase_summaries) == 0
