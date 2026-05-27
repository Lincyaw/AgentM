"""Pure collect helpers over a child session's returned messages.

These functions scrape a result out of a ``list[AgentMessage]`` produced
by driving a nested :class:`AgentSession` to completion. They have no I/O
and no side effects, importing only :mod:`agentm.core.abi.messages`.

They are llmharness-internal: the three audit paths that consume them
(:class:`~llmharness.audit.seams.live.LiveChildRunner`, the embedded
:func:`~llmharness.tools.engine.run_phase_standalone`, and the runner's
trajectory serializer) all live in this package. They are deliberately
NOT in the agentm core tree — llmharness is their sole consumer, and they
are composed entirely from the public ``core.abi`` surface, so the "keep
core small; compose existing ABI in contrib" rule applies. Promote to a
core-tree module (or a registered service) only when a second package
genuinely needs them.

The two result shapes are:

* a **terminal tool-call's arguments** — what the cognitive-audit
  extractor / auditor children produce (:func:`terminal_tool_arguments`);
* the child's **final free text** (:func:`final_assistant_text`).

:func:`flatten_assistant_blocks` (via :func:`serialize_block`) produces
the raw per-block view that the replay sidecar stores verbatim;
:func:`serialize_block` is public so the runner's trajectory serializer
shares the single block-shape definition rather than copying it.
"""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolCallBlock


def serialize_block(block: Any) -> dict[str, Any] | None:
    """Serialize one content block into the replay-sidecar block shape.

    Recognises text, tool-call, and tool-result blocks; anything else is
    captured by ``type`` + ``repr`` so nothing is silently dropped.
    """
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        block_type = getattr(block, "type", None)
        return {
            "type": block_type if isinstance(block_type, str) and block_type else "text",
            "text": text,
        }

    name = getattr(block, "name", None)
    arguments = getattr(block, "arguments", None)
    if isinstance(name, str) and isinstance(arguments, dict):
        return {
            "type": "tool_call",
            "id": getattr(block, "id", None),
            "name": name,
            "arguments": dict(arguments),
        }

    tool_call_id = getattr(block, "tool_call_id", None)
    inner_content = getattr(block, "content", None)
    if isinstance(tool_call_id, str) and isinstance(inner_content, list):
        inner_blocks: list[dict[str, Any]] = []
        for inner in inner_content:
            inner_text = getattr(inner, "text", None)
            if isinstance(inner_text, str):
                inner_blocks.append({"type": "text", "text": inner_text})
            else:
                inner_blocks.append(
                    {
                        "type": getattr(inner, "type", inner.__class__.__name__),
                        "repr": repr(inner),
                    }
                )
        return {
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "content": inner_blocks,
            "is_error": bool(getattr(block, "is_error", False)),
        }

    return {"type": getattr(block, "type", block.__class__.__name__), "repr": repr(block)}


def flatten_assistant_blocks(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Flatten every assistant message's content blocks into dicts.

    Order-preserving across messages and blocks; non-assistant messages
    are skipped. Used to populate the replay sidecar's
    ``raw_assistant_messages`` verbatim.
    """
    blocks: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for blk in content:
            serialized = serialize_block(blk)
            if serialized is not None:
                blocks.append(serialized)
    return blocks


def terminal_tool_arguments(messages: list[AgentMessage], tool_name: str) -> dict[str, Any] | None:
    """Last-match-wins scan for a ``tool_name`` tool-call's arguments.

    Reverse-iteration is deliberate: if a child session somehow emitted
    the terminal tool twice (kernel re-issue, flaky stream), we want the
    *latest* submission. Returns ``None`` when no matching call is found.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


def final_assistant_text(messages: list[AgentMessage]) -> str | None:
    """Concatenate the last assistant message's text blocks.

    Scans from the end for the most recent :class:`AssistantMessage` that
    carries any text block, then joins those blocks' ``text`` with a
    space. Returns ``None`` when no assistant message has text.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        chunks: list[str] = []
        for block in content:
            if getattr(block, "type", None) != "text":
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return " ".join(chunks)
    return None


__all__: Final = [
    "final_assistant_text",
    "flatten_assistant_blocks",
    "serialize_block",
    "terminal_tool_arguments",
]
