"""Tool atom: ``read_history`` — recover the original content of past turns.

The ``llm_compaction`` atom replaces old turns with a structured summary that
cites them as ``[Turn N]``. The raw turns are never deleted from the
trajectory, so this tool reads ``api.get_turns()`` and returns the verbatim
messages for a turn (or turn range) on demand. Turns are numbered 1-based over
the committed sequence so the ``[Turn N]`` markers in a summary line up with
what this tool accepts.

In-session by design: it reads the live committed trajectory (the source of
truth), not the observability JSONL — so there is no flush lag and no
dependency on the observability atom. For cross-session trace mining use
``query_traces`` / ``agentm trace`` instead.
"""

from __future__ import annotations

import json

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from agentm.core.abi.context import turn_to_messages
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.lib.tokens import truncate_text_tokens
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
    requires=(),  # Leaf atom: reads the committed trajectory only.
    config_schema=ReadHistoryConfig,
    priority=AtomInstallPriority.TOOL,
)


class _ReadHistoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=1, description="First turn index (1-based) to fetch.")
    end: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Last turn index to fetch (inclusive). Omit to fetch only `start`."
        ),
    )


class _ReadHistoryRuntime:
    def __init__(self, api: AtomAPI, config: ReadHistoryConfig) -> None:
        self._api = api
        self._tool_result_max_tokens = config.tool_result_max_tokens
        self._total_max_tokens = config.total_max_tokens

    def install(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="read_history",
                description=(
                    "Return the messages of a past turn (or turn range) by "
                    "1-based index, rendered as a role-tagged transcript — "
                    "use it to recover detail behind a [Turn N] reference in "
                    "a compaction summary. Long tool results and the overall "
                    "response are token-capped; on a truncation marker, "
                    "request a narrower turn range. Args: start (required), "
                    "end (optional, inclusive)."
                ),
                parameters=_ReadHistoryArgs,
                fn=self.execute,
            )
        )

    async def execute(self, args: dict[str, object]) -> ToolResult:
        params = _ReadHistoryArgs.model_validate(args)
        model_name = self._api.model.id if self._api.model is not None else None
        start = params.start
        end = params.end if params.end is not None else start
        if end < start:
            start, end = end, start

        # get_turns() returns committed turns; number them 1-based over the
        # committed sequence so the tool's contract is independent of the
        # trajectory's internal index base.
        committed = list(self._api.get_turns())
        if not committed:
            return _error("No turns recorded yet.")
        last = len(committed)
        if start > last:
            return _error(
                f"No turn {start}; the conversation currently has turns 1–{last}."
            )

        selected = [
            (position, turn)
            for position, turn in enumerate(committed, start=1)
            if start <= position <= end
        ]
        rendered = "\n\n".join(
            _render_turn(
                position,
                turn_to_messages(turn),
                self._tool_result_max_tokens,
                model_name,
            )
            for position, turn in selected
        )
        truncated = truncate_text_tokens(
            rendered,
            self._total_max_tokens,
            model=model_name,
        )
        if truncated.was_truncated:
            rendered = (
                truncated.text
                + f"\n\n[... {truncated.truncated_tokens} tokens truncated; "
                "request a narrower turn range]"
            )
        return _ok(rendered)


def install(api: AtomAPI, config: ReadHistoryConfig) -> None:
    _ReadHistoryRuntime(api, config).install()


def _render_turn(
    index: int,
    messages: list[AgentMessage],
    tool_result_cap: int,
    model_name: str | None,
) -> str:
    lines = [f"=== Turn {index} ==="]
    for message in messages:
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
                a_parts.append(
                    f"[tool call] {a_block.name}({_dump(a_block.arguments)})"
                )
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


def _dump(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _cap(text: str, max_tokens: int, model_name: str | None) -> str:
    truncated = truncate_text_tokens(text, max_tokens, model=model_name)
    if not truncated.was_truncated:
        return text
    return (
        truncated.text + f"\n[... {truncated.truncated_tokens} more tokens truncated]"
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
