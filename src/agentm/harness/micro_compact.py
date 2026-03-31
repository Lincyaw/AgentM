# ruff: noqa: ARG002  — ctx parameter is required by the MiddlewareBase protocol
"""Micro-compact middleware: lightweight context cleanup for old tool results.

Clears stale tool results from compactable tools, replacing their content
with a placeholder message.  Runs before CompressionMiddleware to reduce
token counts cheaply — often avoiding the expensive LLM-based compression
entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agentm.harness.middleware import MiddlewareBase, msg_content, msg_role, msg_tool_calls
from agentm.harness.types import LoopContext, Message


@dataclass(frozen=True)
class MicroCompactConfig:
    """Configuration for MicroCompactMiddleware."""

    enabled: bool = True
    stale_after_steps: int = 6
    compactable_tools: frozenset[str] = field(
        default_factory=lambda: frozenset({"duckdb_sql", "vault_read", "vault_search"})
    )
    cleared_message: str = "[Old tool result content cleared]"


def _get_tool_call_id(msg: Message) -> str:
    """Extract tool_call_id from a tool-result message."""
    if isinstance(msg, dict):
        return str(msg.get("tool_call_id", ""))
    return str(getattr(msg, "tool_call_id", ""))




def _cleared_copy(msg: Message, cleared_message: str) -> Message:
    """Return a copy of the message with content replaced by cleared_message."""
    if isinstance(msg, dict):
        return {**msg, "content": cleared_message}
    # For LangChain message objects, create a dict representation
    return {
        "role": "tool",
        "content": cleared_message,
        "tool_call_id": _get_tool_call_id(msg),
    }


class MicroCompactMiddleware(MiddlewareBase):
    """Clears stale tool results from compactable tools.

    Uses turn counting (assistant messages from the end) to determine
    message age.  Only tool results from tools listed in
    ``compactable_tools`` are affected.
    """

    def __init__(self, config: MicroCompactConfig | None = None) -> None:
        self._config = config or MicroCompactConfig()

    async def on_llm_start(
        self, messages: list[Message], ctx: LoopContext
    ) -> list[Message]:
        cfg = self._config

        if not cfg.enabled:
            return messages

        if ctx.step < cfg.stale_after_steps * 2:
            return messages

        # Build tool_call_id → tool_name index from assistant messages
        tc_to_tool: dict[str, str] = {}
        for msg in messages:
            for tc in msg_tool_calls(msg):
                tc_to_tool[tc.get("id", "")] = tc.get("name", "")

        # Count turns from the end — each assistant message is a turn boundary
        msg_ages: dict[int, int] = {}
        turn_from_end = 0
        for msg in reversed(messages):
            role = msg_role(msg)
            if role in ("ai", "assistant"):
                turn_from_end += 1
            if role == "tool":
                msg_ages[id(msg)] = turn_from_end

        # Build a new message list, clearing stale compactable tool results
        new_messages: list[Message] = []
        changed = False
        for msg in messages:
            role = msg_role(msg)
            if role == "tool":
                age = msg_ages.get(id(msg), 0)
                tool_call_id = _get_tool_call_id(msg)
                tool_name = tc_to_tool.get(tool_call_id, "")
                content = msg_content(msg)

                if (
                    age >= cfg.stale_after_steps
                    and tool_name in cfg.compactable_tools
                    and content != cfg.cleared_message
                ):
                    new_messages.append(_cleared_copy(msg, cfg.cleared_message))
                    changed = True
                    continue

            new_messages.append(msg)

        return new_messages if changed else messages
