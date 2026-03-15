"""Compression middleware — prevents token overflow via LLM-based summarization.

Two modes:
1. Default hook (``sub_agent_compression_hook``): uses hardcoded thresholds.
2. Configurable hook (``build_compression_hook``): respects ``CompressionConfig``.

Both follow the ``pre_model_hook`` protocol: receive state dict, return dict
with ``messages`` (passthrough) or ``llm_input_messages`` (compressed).

Ref: designs/orchestrator.md § Compression Architecture
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

import tiktoken
from langchain_core.messages import SystemMessage

from agentm.config.schema import CompressionConfig
from agentm.middleware import AgentMMiddleware

# ---------------------------------------------------------------------------
# Compression event tracking (contextvars)
# ---------------------------------------------------------------------------

# Thread-safe tracking of compression events within a single agent invocation.
# Callers (e.g. orchestrator/builder) read events via get_compression_events()
# and clear them after persisting to state.
_compression_events: contextvars.ContextVar[list[dict[str, Any]]] = (
    contextvars.ContextVar("compression_events", default=[])
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


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_THRESHOLD_RATIO = 0.8
_DEFAULT_THRESHOLD_TOKENS = int(_DEFAULT_CONTEXT_WINDOW * _DEFAULT_THRESHOLD_RATIO)
_DEFAULT_PRESERVE_N = 2


def count_tokens(messages: list[Any], model: str = "gpt-4o") -> int:
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


# ---------------------------------------------------------------------------
# LLM-based summarization
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = (
    "Summarize the following agent execution history into a structured summary. "
    "Preserve: key findings, tool call results, data values, and decisions made. "
    "Be concise but retain specific data points (numbers, names, timestamps)."
)

# Max tokens to send to the compression model per chunk (leave headroom for prompt + response)
_MAX_CHUNK_TOKENS = 100_000


def _format_messages_for_summary(messages: list[Any]) -> list[str]:
    """Format messages into text lines for the summarization prompt."""
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
    return formatted


def _summarize_messages(messages: list[Any], model: str = "gpt-4o-mini") -> str:
    """Summarize a list of messages using an LLM, chunking if needed.

    When the formatted messages exceed the compression model's context window,
    they are split into chunks. Each chunk is summarized independently, then
    the chunk summaries are combined into a final summary.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage as LCHumanMessage

    formatted_lines = _format_messages_for_summary(messages)

    # Split into chunks that fit the compression model's context window
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    for line in formatted_lines:
        line_tokens = len(encoding.encode(line))
        if current_tokens + line_tokens > _MAX_CHUNK_TOKENS and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    llm = ChatOpenAI(model=model, temperature=0)

    if len(chunks) == 1:
        # Single chunk — summarize directly
        prompt = f"{_SUMMARIZE_PROMPT}\n\nMessages:\n{chunks[0]}"
        result = llm.invoke([LCHumanMessage(content=prompt)])
        content = result.content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)

    # Multi-chunk — summarize each chunk, then combine
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        prompt = (
            f"{_SUMMARIZE_PROMPT}\n\n"
            f"This is part {i + 1} of {len(chunks)} of the execution history.\n\n"
            f"Messages:\n{chunk}"
        )
        result = llm.invoke([LCHumanMessage(content=prompt)])
        content = result.content
        if isinstance(content, list):
            chunk_summaries.append(" ".join(str(part) for part in content))
        else:
            chunk_summaries.append(str(content))

    # Combine chunk summaries into a final summary
    combined = "\n\n---\n\n".join(
        f"[Part {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    final_prompt = (
        "Combine the following partial summaries into one coherent, structured summary. "
        "Remove redundancy but preserve all key data points.\n\n"
        f"{combined}"
    )
    result = llm.invoke([LCHumanMessage(content=final_prompt)])
    content = result.content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


# ---------------------------------------------------------------------------
# Pre-model hooks
# ---------------------------------------------------------------------------


def sub_agent_compression_hook(state: dict[str, Any]) -> dict[str, Any]:
    """pre_model_hook: compress messages before LLM call when token limit approached.

    .. deprecated::
        Use ``build_compression_hook(config)`` for configurable compression.
        This function uses hardcoded defaults.

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

    record_compression_event(
        {
            "layer": "sub_agent",
            "step_count": len(older_messages),
            "reason": f"token_count={token_count} exceeded threshold={_DEFAULT_THRESHOLD_TOKENS}",
        }
    )

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
        summary_msg = SystemMessage(
            content=f"[Compressed History Summary]\n{summary_text}"
        )

        record_compression_event(
            {
                "layer": "sub_agent",
                "step_count": len(older),
                "reason": f"token_count={token_count} exceeded threshold={threshold_tokens}",
            }
        )

        return {"llm_input_messages": [summary_msg] + recent}

    return hook


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class CompressionMiddleware(AgentMMiddleware):
    """Pre-model middleware that compresses message history when threshold exceeded."""

    def __init__(self, config: CompressionConfig) -> None:
        self._config = config
        self._hook = build_compression_hook(config)

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._hook(state)
