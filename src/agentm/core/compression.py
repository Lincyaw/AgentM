"""Context compression for Sub-Agent and Orchestrator layers.

All functions are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any, Callable

from agentm.config.schema import CompressionConfig
from agentm.models.data import DiagnosticNotebook


def sub_agent_compression_hook(state: dict[str, Any]) -> dict[str, Any]:
    """pre_model_hook: compress messages before LLM call when token limit approached.

    When compression is NOT needed:
        Returns {'messages': state['messages']} — passes messages through unchanged.

    When compression IS triggered (token count exceeds threshold):
        Returns {'llm_input_messages': compressed_messages} — the 'llm_input_messages'
        key tells create_react_agent to use these messages for the LLM call instead of
        the state's messages, while preserving the full history in state.
    """
    raise NotImplementedError


def compress_completed_phase(
    notebook: DiagnosticNotebook,
    completed_phase: str,
) -> DiagnosticNotebook:
    """Compress a completed phase's detailed records into a PhaseSummary.

    Returns a new DiagnosticNotebook with phase_summaries updated and
    exploration_history pruned for the completed phase.
    """
    raise NotImplementedError


def count_tokens(messages: list[Any], model: str = "gpt-4") -> int:
    """Count the number of tokens in a list of messages for a given model."""
    raise NotImplementedError


def build_compression_hook(config: CompressionConfig) -> Callable:
    """Build a pre_model_hook function for Sub-Agent compression, configured from agent config.

    Returns a callable with the same interface as sub_agent_compression_hook,
    but with threshold and model settings bound from the CompressionConfig.
    """
    raise NotImplementedError
