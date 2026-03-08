"""Context compression for Sub-Agent and Orchestrator layers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import tiktoken

from agentm.config.schema import CompressionConfig
from agentm.models.data import DiagnosticNotebook, PhaseSummary

_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_THRESHOLD_RATIO = 0.8
_DEFAULT_THRESHOLD_TOKENS = int(_DEFAULT_CONTEXT_WINDOW * _DEFAULT_THRESHOLD_RATIO)


def count_tokens(messages: list[Any], model: str = "gpt-4") -> int:
    """Count the number of tokens in a list of messages for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
        else:
            content = msg.get("content", "")
        if isinstance(content, str):
            total += len(encoding.encode(content))
    return total


def sub_agent_compression_hook(state: dict[str, Any]) -> dict[str, Any]:
    """pre_model_hook: compress messages before LLM call when token limit approached.

    When compression is NOT needed:
        Returns {'messages': state['messages']} — passes messages through unchanged.

    When compression IS triggered (token count exceeds threshold):
        Returns {'llm_input_messages': compressed_messages} — the 'llm_input_messages'
        key tells create_react_agent to use these messages for the LLM call instead of
        the state's messages, while preserving the full history in state.
    """
    messages = state.get("messages", [])
    token_count = count_tokens(messages)

    if token_count < _DEFAULT_THRESHOLD_TOKENS:
        return {"messages": messages}

    # Phase 1: pass-through — LLM summarization deferred to later phase
    return {"llm_input_messages": messages}


def build_compression_hook(config: CompressionConfig) -> Callable:
    """Build a pre_model_hook function for Sub-Agent compression, configured from agent config.

    Returns a callable with the same interface as sub_agent_compression_hook,
    but with threshold and model settings bound from the CompressionConfig.
    """
    threshold_tokens = int(_DEFAULT_CONTEXT_WINDOW * config.compression_threshold)

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        token_count = count_tokens(messages, model=config.compression_model)

        if token_count < threshold_tokens:
            return {"messages": messages}

        # Phase 1: pass-through — LLM summarization deferred to later phase
        return {"llm_input_messages": messages}

    return hook


def compress_completed_phase(
    notebook: DiagnosticNotebook,
    completed_phase: str,
) -> DiagnosticNotebook:
    """Compress a completed phase's detailed records into a PhaseSummary.

    Returns a new DiagnosticNotebook with phase_summaries updated and
    exploration_history pruned for the completed phase.
    """
    phase_steps = [
        step for step in notebook.exploration_history
        if step.phase.value == completed_phase
    ]
    remaining_steps = [
        step for step in notebook.exploration_history
        if step.phase.value != completed_phase
    ]

    started_at = phase_steps[0].timestamp if phase_steps else ""
    completed_at = phase_steps[-1].timestamp if phase_steps else ""
    actions_taken = [step.action for step in phase_steps]

    hypothesis_ids: list[str] = []
    for step in phase_steps:
        if step.target_hypothesis_id and step.target_hypothesis_id not in hypothesis_ids:
            hypothesis_ids.append(step.target_hypothesis_id)

    summary = PhaseSummary(
        phase=completed_phase,
        started_at=started_at,
        completed_at=completed_at,
        actions_taken=actions_taken,
        hypotheses_affected=hypothesis_ids,
    )

    new_phase_summaries = list(notebook.phase_summaries) + [summary]
    return replace(
        notebook,
        exploration_history=remaining_steps,
        phase_summaries=new_phase_summaries,
    )
