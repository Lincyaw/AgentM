"""Tool output offload middleware -- truncates large tool outputs.

When a ``ToolMessage`` content exceeds ``max_chars``, it is replaced
with a truncated version that keeps the first ``head_chars`` and last
``tail_chars`` characters, separated by a note indicating how much
content was omitted.  This prevents context window bloat from very
large tool outputs (e.g. file reads, SQL results).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import ToolMessage

from agentm.middleware import AgentMMiddleware

# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def build_tool_output_offload_hook(
    max_chars: int = 4000,
    head_chars: int = 1500,
    tail_chars: int = 500,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a ``pre_model_hook`` that truncates large ``ToolMessage`` contents.

    Parameters
    ----------
    max_chars:
        Threshold above which a ``ToolMessage`` content is truncated.
    head_chars:
        Number of characters to keep from the beginning.
    tail_chars:
        Number of characters to keep from the end.
    """

    def _truncate_content(content: str) -> str:
        if len(content) <= max_chars:
            return content
        omitted = len(content) - head_chars - tail_chars
        return (
            f"{content[:head_chars]}\n\n"
            f"[... {omitted} characters omitted "
            f"(original length: {len(content)}) ...]\n\n"
            f"{content[-tail_chars:]}"
        )

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("llm_input_messages") or state.get("messages", [])
        use_llm_input = "llm_input_messages" in state
        out_key = "llm_input_messages" if use_llm_input else "messages"

        new_messages: list[Any] = []
        changed = False

        for msg in messages:
            if (
                isinstance(msg, ToolMessage)
                and isinstance(msg.content, str)
                and len(msg.content) > max_chars
            ):
                truncated = _truncate_content(msg.content)
                new_msg = ToolMessage(
                    content=truncated,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None) or "",
                )
                new_messages.append(new_msg)
                changed = True
            else:
                new_messages.append(msg)

        if changed:
            return {out_key: new_messages}
        return {out_key: messages}

    return hook


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class ToolOutputOffloadMiddleware(AgentMMiddleware):
    """Truncate large tool outputs to prevent context bloat.

    When a ``ToolMessage`` content exceeds ``max_chars``, replaces the
    content with head + tail of the output and a note about truncation.
    Operates on messages before they reach the LLM.
    """

    def __init__(
        self,
        max_chars: int = 4000,
        head_chars: int = 1500,
        tail_chars: int = 500,
    ) -> None:
        self._max_chars = max_chars
        self._head_chars = head_chars
        self._tail_chars = tail_chars
        self._hook = build_tool_output_offload_hook(
            max_chars=max_chars,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._hook(state)
