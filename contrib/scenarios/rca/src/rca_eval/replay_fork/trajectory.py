"""Rehydrate OpenAI-style chat messages into AgentM ``AgentMessage`` objects.

This is the inverse of ``rca_eval.agent._build_trajectory``, which
serializes an AgentM session's ``final_messages`` into rcabench-platform's
``Message`` schema (``{role, content, tool_calls, tool_call_id}``). The
replay-fork driver needs the round-trip: a recorded baseline trajectory --
stored as OpenAI chat-completion messages -- is rehydrated into
``list[AgentMessage]`` so the offline audit pipeline and the fork-tree
engine treat it exactly like a freshly produced trajectory.

The function is generic to any OpenAI chat-completion source; it knows
nothing about where the messages came from. Source-specific unwrapping
(e.g. pulling the message list out of an eval.db row) lives in the
``CaseSource`` implementations.

The ``system`` message is returned separately as the system prompt rather
than as an ``AgentMessage`` -- AgentM carries the system prompt on the
session config, not in the message stream, matching how the live agent and
the offline pipeline consume it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentm.core.abi import (
    AgentMessage,
    AssistantContent,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)

_logger = logging.getLogger(__name__)

# Replayed messages carry no meaningful wall-clock time: the offline audit
# pipeline and the fork-tree prefix-slicer key off message order and type
# (``isinstance`` checks), never the timestamp. A single fixed value keeps
# the rehydrated trajectory deterministic.
_REPLAY_TS = 0.0

__all__ = ["openai_chat_to_agentm"]

def openai_chat_to_agentm(
    messages: list[dict[str, Any]],
) -> tuple[str, list[AgentMessage]]:
    """Convert OpenAI chat-completion message dicts to AgentM messages.

    Returns ``(system_prompt, agent_messages)``. ``system`` entries are
    concatenated into the system prompt; ``user`` / ``assistant`` / ``tool``
    entries become ``UserMessage`` / ``AssistantMessage`` /
    ``ToolResultMessage``. Unknown roles are skipped with a warning so a
    single malformed entry never aborts a whole-batch replay.
    """
    system_parts: list[str] = []
    out: list[AgentMessage] = []
    for raw in messages:
        role = raw.get("role")
        if role == "system":
            text = _as_text(raw.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            out.append(
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text=_as_text(raw.get("content")))],
                    timestamp=_REPLAY_TS,
                )
            )
        elif role == "assistant":
            out.append(_to_assistant(raw))
        elif role == "tool":
            out.append(_to_tool_result(raw))
        else:
            _logger.warning(
                "openai_chat_to_agentm: skipping message with unknown role %r", role
            )
    return "\n\n".join(system_parts), out

def _to_assistant(raw: dict[str, Any]) -> AssistantMessage:
    blocks: list[AssistantContent] = []
    text = _as_text(raw.get("content"))
    if text:
        blocks.append(TextContent(type="text", text=text))
    for tc in raw.get("tool_calls") or []:
        fn = tc.get("function") or {}
        blocks.append(
            ToolCallBlock(
                type="tool_call",
                id=str(tc.get("id") or ""),
                name=str(fn.get("name") or ""),
                arguments=_parse_arguments(fn.get("arguments"), call_id=tc.get("id")),
            )
        )
    if not blocks:
        # An assistant turn with neither text nor tool calls is degenerate
        # but must still materialise so message indices line up with the
        # recorded trajectory the auditor saw.
        blocks.append(TextContent(type="text", text=""))
    return AssistantMessage(role="assistant", content=blocks, timestamp=_REPLAY_TS)

def _to_tool_result(raw: dict[str, Any]) -> ToolResultMessage:
    return ToolResultMessage(
        role="tool_result",
        content=[
            ToolResultBlock(
                type="tool_result",
                tool_call_id=str(raw.get("tool_call_id") or ""),
                content=[TextContent(type="text", text=_as_text(raw.get("content")))],
                is_error=False,
            )
        ],
        timestamp=_REPLAY_TS,
    )

def _parse_arguments(arguments: Any, *, call_id: Any) -> dict[str, Any]:
    """Coerce an OpenAI ``tool_calls[].function.arguments`` to a dict.

    The field is a JSON-encoded string in the wire format. Fail soft to an
    empty dict (with a warning) rather than aborting the batch: the only
    consumer in replay is the extractor reading the call shape, and a single
    unparseable historical call should not sink the run.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            _logger.warning(
                "tool_call %r: arguments not valid JSON; using empty dict", call_id
            )
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

def _as_text(content: Any) -> str:
    """Flatten an OpenAI ``content`` field to plain text.

    Handles the three shapes seen in practice: ``None``, a plain string, and
    the structured content-parts list (each part a ``{"type","text"}`` dict).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)
