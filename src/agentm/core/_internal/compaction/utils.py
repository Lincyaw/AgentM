"""Shared utilities for LLM-driven compaction and branch summarization.

The kernel keeps **zero** scenario-specific knowledge:

- It does not enumerate tool names. Tool atoms self-describe via
  ``tool.metadata[FILE_OP_METADATA_KEY]`` and the engine reads that
  metadata through a tool registry passed in at call time.
- It does not ship any English summarization prompt. The
  ``compaction_prompts`` atom registers them via
  ``ExtensionAPI.prompt_templates`` and the harness/atom callers thread the
  resolved bodies into the engine functions as parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    Tool,
    ToolCallBlock,
    ToolResultMessage,
)
from agentm.core.abi.tool import (
    FILE_OP_EDIT,
    FILE_OP_METADATA_KEY,
    FILE_OP_READ,
    FILE_OP_WRITE,
)

DEFAULT_TOOL_RESULT_MAX_CHARS = 2_000


# A flexible tool-registry shape for ``extract_file_ops_from_message``:
# either a name->tool mapping, or any iterable of tools (we'll index by
# ``tool.name`` ourselves).
ToolRegistry = Mapping[str, Tool] | Sequence[Tool]


@dataclass(slots=True)
class FileOperations:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    return FileOperations()


def _normalize_registry(tools: ToolRegistry | None) -> Mapping[str, Tool]:
    if tools is None:
        return {}
    if isinstance(tools, Mapping):
        return tools
    return {tool.name: tool for tool in tools}


def extract_file_ops_from_message(
    message: AgentMessage,
    file_ops: FileOperations,
    tools: ToolRegistry | None = None,
) -> None:
    """Inspect ``message`` for tool calls and route ``path`` arguments into
    ``file_ops`` based on each tool's ``metadata["file_op"]`` value.

    ``tools`` is the live tool registry at the call site. When the registry
    is empty or a tool exposes no ``file_op`` metadata, the corresponding
    tool call contributes no file-ops — this is the graceful-degradation
    contract documented in issue #76 (compaction without the prompts/tool-
    metadata atom installed yields empty file lists, not a crash).
    """

    if not isinstance(message, AssistantMessage):
        return

    registry = _normalize_registry(tools)
    if not registry:
        return

    for block in message.content:
        if not isinstance(block, ToolCallBlock):
            continue
        path = block.arguments.get("path")
        if not isinstance(path, str) or not path:
            continue
        tool = registry.get(block.name)
        if tool is None:
            continue
        metadata = getattr(tool, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        file_op = metadata.get(FILE_OP_METADATA_KEY)
        if file_op == FILE_OP_READ:
            file_ops.read.add(path)
        elif file_op == FILE_OP_WRITE:
            file_ops.written.add(path)
        elif file_op == FILE_OP_EDIT:
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> tuple[list[str], list[str]]:
    modified = set(file_ops.edited)
    modified.update(file_ops.written)
    read_only = sorted(path for path in file_ops.read if path not in modified)
    return read_only, sorted(modified)


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append(
            "<modified-files>\n"
            + "\n".join(modified_files)
            + "\n</modified-files>"
        )
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return (
        f"{text[:max_chars]}\n\n"
        f"[... {truncated_chars} more characters truncated]"
    )


def serialize_conversation(
    messages: list[AgentMessage],
    *,
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> str:
    parts: list[str] = []

    for message in messages:
        if getattr(message, "role", None) == "user":
            content = "".join(
                block.text
                for block in getattr(message, "content", [])
                if isinstance(block, TextContent)
            )
            if content:
                parts.append(f"[User]: {content}")
            continue

        if isinstance(message, AssistantMessage):
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            for block in message.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif getattr(block, "type", None) == "thinking":
                    thinking_parts.append(getattr(block, "text", ""))
                elif isinstance(block, ToolCallBlock):
                    args = ", ".join(
                        f"{key}={value!r}" for key, value in block.arguments.items()
                    )
                    tool_calls.append(f"{block.name}({args})")
            if thinking_parts:
                parts.append("[Assistant thinking]: " + "\n".join(thinking_parts))
            if text_parts:
                parts.append("[Assistant]: " + "\n".join(text_parts))
            if tool_calls:
                parts.append("[Assistant tool calls]: " + "; ".join(tool_calls))
            continue

        if isinstance(message, ToolResultMessage):
            content = "".join(
                part.text
                for block in message.content
                for part in block.content
                if isinstance(part, TextContent)
            )
            if content:
                parts.append(
                    "[Tool result]: "
                    + _truncate_for_summary(content, tool_result_max_chars)
                )

    return "\n\n".join(parts)


__all__ = [
    "FileOperations",
    "ToolRegistry",
    "compute_file_lists",
    "create_file_ops",
    "extract_file_ops_from_message",
    "format_file_operations",
    "serialize_conversation",
]
