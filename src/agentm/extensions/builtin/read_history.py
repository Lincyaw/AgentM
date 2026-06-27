"""Tool atom: ``read_history`` — recover the original content of past turns.

The ``llm_compaction`` atom replaces old turns with a structured summary that
cites them as ``[Turn N]``. The raw turns are never deleted from the session
tree, so this tool reads ``api.session.get_branch()`` and returns the verbatim
messages for a turn (or turn range) on demand. Turn numbering is shared with
the compaction engine via :func:`agentm.core.lib.enumerate_turns`, so the
``[Turn N]`` markers in a summary line up with what this tool accepts.

In-session by design: it reads the live SessionManager (the source of truth),
not the observability JSONL — so there is no flush lag and no dependency on
the observability atom. For cross-session trace mining use ``query_traces`` /
``agentm trace`` instead.
"""

from __future__ import annotations

import json
from typing import Any, Final

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.lib import Turn, enumerate_turns, truncate_text_tokens
from agentm.extensions import ExtensionManifest

class ReadHistoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Per historical tool result. Prevents one old command output from
    # crowding out the rest of the recalled turn range.
    tool_result_max_tokens: int = Field(gt=0)
    # Whole read_history response. Prevents broad turn ranges from undoing
    # the context savings that compaction just created.
    total_max_tokens: int = Field(gt=0)

MANIFEST = ExtensionManifest(
    name="read_history",
    description=(
        "Recover the verbatim messages of past conversation turns by index "
        "(the [Turn N] markers cited in compaction summaries)."
    ),
    registers=("tool:read_history",),
    requires=(),  # Leaf atom: reads the session branch only.
    config_schema=ReadHistoryConfig,
)

_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "start": {
            "type": "integer",
            "minimum": 1,
            "description": "First turn index (1-based) to fetch.",
        },
        "end": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Last turn index to fetch (inclusive). Omit to fetch only `start`."
            ),
        },
    },
    "required": ["start"],
    "additionalProperties": False,
}

def install(api: ExtensionAPI, config: ReadHistoryConfig) -> None:
    tool_result_max_tokens = config.tool_result_max_tokens
    total_max_tokens = config.total_max_tokens

    async def _execute(args: dict[str, Any]) -> ToolResult:
        model_name = api.model.id if api.model is not None else None
        start = int(args["start"])
        end_raw = args.get("end")
        end = int(end_raw) if end_raw is not None else start
        if end < start:
            start, end = end, start

        turns = enumerate_turns(api.session.get_branch())
        if not turns:
            return _error("No turns recorded yet.")
        last = turns[-1].index
        if start > last:
            return _error(
                f"No turn {start}; the conversation currently has turns 1–{last}."
            )

        selected = [turn for turn in turns if start <= turn.index <= end]
        rendered = "\n\n".join(
            _render_turn(turn, tool_result_max_tokens, model_name) for turn in selected
        )
        truncated = truncate_text_tokens(
            rendered,
            total_max_tokens,
            model=model_name,
        )
        if truncated.was_truncated:
            rendered = (
                truncated.text
                + f"\n\n[... {truncated.truncated_tokens} tokens truncated; "
                "request a narrower turn range]"
            )
        return _ok(rendered)

    api.register_tool(
        FunctionTool(
            name="read_history",
            description=(
                "Return the verbatim messages of a past turn (or turn range) "
                "by 1-based index — use it to recover detail behind a [Turn N] "
                "reference in a compaction summary. Args: start (required), "
                "end (optional, inclusive)."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )

def _render_turn(turn: Turn, tool_result_cap: int, model_name: str | None) -> str:
    lines = [f"=== Turn {turn.index} ==="]
    for message in turn.messages:
        lines.append(_render_message(message, tool_result_cap, model_name))
    return "\n".join(lines)

def _render_message(
    message: AgentMessage,
    tool_result_cap: int,
    model_name: str | None,
) -> str:
    if isinstance(message, UserMessage):
        text = "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        return f"[user] {text}"

    if isinstance(message, AssistantMessage):
        a_parts: list[str] = []
        for a_block in message.content:
            if isinstance(a_block, ThinkingBlock):
                a_parts.append(f"[assistant thinking] {a_block.text}")
            elif isinstance(a_block, TextContent):
                a_parts.append(f"[assistant] {a_block.text}")
            elif isinstance(a_block, ToolCallBlock):
                a_parts.append(f"[tool call] {a_block.name}({_dump(a_block.arguments)})")
        return "\n".join(a_parts) if a_parts else "[assistant] (empty)"

    if isinstance(message, ToolResultMessage):
        r_parts: list[str] = []
        for r_block in message.content:
            text = "".join(
                part.text for part in r_block.content if isinstance(part, TextContent)
            )
            tag = "tool result error" if r_block.is_error else "tool result"
            r_parts.append(f"[{tag}] {_cap(text, tool_result_cap, model_name)}")
        return "\n".join(r_parts)

    return ""

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)

def _cap(text: str, max_tokens: int, model_name: str | None) -> str:
    truncated = truncate_text_tokens(text, max_tokens, model=model_name)
    if not truncated.was_truncated:
        return text
    return (
        truncated.text
        + f"\n[... {truncated.truncated_tokens} more tokens truncated]"
    )

def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])

def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
