"""Pure collect helpers over a child session's returned messages.

These functions scrape a result out of a ``list[AgentMessage]`` produced
by driving a nested :class:`AgentSession` to completion. They have no I/O
and no side effects, importing only :mod:`agentm.core.abi.messages`.

They are llmharness-internal: the three audit paths that consume them
(:class:`~llmharness.runtime.live.LiveChildRunner`, the embedded
:func:`~llmharness.replay.engine.run_phase_standalone`, and the runner's
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

from collections.abc import Awaitable, Callable
from typing import Any, Final

from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolCallBlock

# Maximum re-prompts :func:`nudge_until_tool_call` issues to a child that
# emitted zero tool calls. See that helper for the full WHY.
MAX_EMPTY_TURN_NUDGES: Final = 1


def build_empty_turn_nudge(terminal_tool: str) -> str:
    """The nudge message sent to a child that produced zero tool calls.

    Parametrised on the terminal tool name so the same text serves the
    extractor (``finalize_extraction``) and the auditor (``submit_verdict``).
    """
    return (
        "You ended your turn without calling any tool. Your output is recorded "
        "ONLY through tool calls — prose and reasoning alone change nothing. Emit "
        "your work as tool calls now (for the graph: upsert_node / upsert_edge / "
        f"delete_node / delete_edge; then call {terminal_tool} when done). If there "
        f"is genuinely nothing to record, call {terminal_tool} with an empty result "
        "to end cleanly."
    )


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


def has_any_tool_call(messages: list[AgentMessage]) -> bool:
    """True iff any assistant block in ``messages`` is a tool call.

    Reuses :func:`flatten_assistant_blocks`, whose serialized blocks tag
    tool calls with ``"type": "tool_call"`` — the same shape the replay
    sidecar stores. Used by both child-running seams to detect a
    genuinely-empty turn (zero tool calls) that warrants an empty-turn
    nudge before the terminal-args scrape.
    """
    return any(block.get("type") == "tool_call" for block in flatten_assistant_blocks(messages))


async def nudge_until_tool_call(
    prompt_fn: Callable[[str], Awaitable[list[AgentMessage]]],
    messages: list[AgentMessage],
    terminal_tool: str,
) -> list[AgentMessage]:
    """Re-prompt a child that emitted zero tool calls, up to the bound.

    A reasoning-heavy child can end its turn producing only prose (zero
    tool calls), so nothing is recorded and the firing reports
    ``no_call``. While the collected ``messages`` carry no tool call,
    re-prompt the SAME session (``prompt_fn`` is the child's / session's
    ``prompt`` — the conversation persists across calls) with
    :func:`build_empty_turn_nudge`, bounded by :data:`MAX_EMPTY_TURN_NUDGES`,
    and append the returned messages. A ``prompt_fn`` exception breaks the
    loop and returns whatever messages exist — a nudge must never turn a
    result into a hard error. Shared by both child-running seams (live
    ``run_child_task``, offline ``run_phase_standalone``) so the design's
    live ≡ offline invariant holds.
    """
    for _ in range(MAX_EMPTY_TURN_NUDGES):
        if has_any_tool_call(messages):
            break
        try:
            nudged = await prompt_fn(build_empty_turn_nudge(terminal_tool))
        except Exception:
            break
        messages = messages + nudged
    return messages


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
    "MAX_EMPTY_TURN_NUDGES",
    "build_empty_turn_nudge",
    "final_assistant_text",
    "flatten_assistant_blocks",
    "has_any_tool_call",
    "nudge_until_tool_call",
    "serialize_block",
    "terminal_tool_arguments",
]
