"""Context compression for Sub-Agent and Orchestrator layers."""

from __future__ import annotations

import contextvars
from dataclasses import replace
from typing import Any, Callable

import tiktoken
from langchain_core.messages import SystemMessage

from agentm.config.schema import CompressionConfig
from agentm.models.data import DiagnosticNotebook, PhaseSummary

# Thread-safe tracking of compression events within a single agent invocation.
# Callers (e.g. orchestrator/builder) read events via get_compression_events()
# and clear them after persisting to state.
_compression_events: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar(
    "compression_events", default=[]
)


def record_compression_event(event: dict[str, Any]) -> None:
    """Record that a compression occurred. Called by compression hooks."""
    events = list(_compression_events.get([]))
    events.append(event)
    _compression_events.set(events)


def get_compression_events() -> list[dict[str, Any]]:
    """Get all recorded compression events since last clear."""
    return list(_compression_events.get([]))


def clear_compression_events() -> None:
    """Clear recorded compression events."""
    _compression_events.set([])

_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_THRESHOLD_RATIO = 0.8
_DEFAULT_THRESHOLD_TOKENS = int(_DEFAULT_CONTEXT_WINDOW * _DEFAULT_THRESHOLD_RATIO)
_DEFAULT_PRESERVE_N = 2


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


def _summarize_messages(messages: list[Any], model: str = "gpt-4o-mini") -> str:
    """Summarize a list of messages into a structured text summary using an LLM."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage as LCHumanMessage

    formatted = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_info = ", ".join(tc.get("name", "?") for tc in msg.tool_calls)
            formatted.append(f"[{role}] Called tools: {tool_info}")
        elif content:
            preview = content[:500] + "..." if len(content) > 500 else content
            formatted.append(f"[{role}] {preview}")

    messages_text = "\n".join(formatted)

    prompt = (
        "Summarize the following agent execution history into a structured summary. "
        "Preserve: key findings, tool call results, data values, and decisions made. "
        "Be concise but retain specific data points (numbers, names, timestamps).\n\n"
        f"Messages:\n{messages_text}"
    )

    llm = ChatOpenAI(model=model, temperature=0)
    result = llm.invoke([LCHumanMessage(content=prompt)])
    content = result.content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


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

    if len(messages) <= _DEFAULT_PRESERVE_N:
        return {"messages": messages}

    older_messages = messages[:-_DEFAULT_PRESERVE_N]
    recent_messages = messages[-_DEFAULT_PRESERVE_N:]

    summary_text = _summarize_messages(older_messages)
    summary_msg = SystemMessage(content=f"[Compressed History Summary]\n{summary_text}")

    record_compression_event({
        "layer": "sub_agent",
        "step_count": len(older_messages),
        "reason": f"token_count={token_count} exceeded threshold={_DEFAULT_THRESHOLD_TOKENS}",
    })

    return {"llm_input_messages": [summary_msg] + recent_messages}


def build_compression_hook(config: CompressionConfig) -> Callable:
    """Build a pre_model_hook function for Sub-Agent compression, configured from agent config.

    Returns a callable with the same interface as sub_agent_compression_hook,
    but with threshold and model settings bound from the CompressionConfig.
    """
    threshold_tokens = int(_DEFAULT_CONTEXT_WINDOW * config.compression_threshold)
    preserve_n = config.preserve_latest_n

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        token_count = count_tokens(messages, model=config.compression_model)

        if token_count < threshold_tokens:
            return {"messages": messages}

        if len(messages) <= preserve_n:
            return {"messages": messages}

        older = messages[:-preserve_n]
        recent = messages[-preserve_n:]

        summary_text = _summarize_messages(older, model=config.compression_model)
        summary_msg = SystemMessage(content=f"[Compressed History Summary]\n{summary_text}")

        record_compression_event({
            "layer": "sub_agent",
            "step_count": len(older),
            "reason": f"token_count={token_count} exceeded threshold={threshold_tokens}",
        })

        return {"llm_input_messages": [summary_msg] + recent}

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
