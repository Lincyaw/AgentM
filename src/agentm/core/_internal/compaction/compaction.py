"""Pure compaction logic adapted from pi-mono's LLM-driven compaction."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from agentm.core.abi.compaction import (
    CompactionDetails,
    CompactionResult,
    CompactionSettings,
    ContextUsageEstimate,
)
from agentm.core.abi.session import SessionEntry

from .utils import (
    FileOperations,
    SUMMARIZATION_SYSTEM_PROMPT,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

Summarizer = Callable[[str, str, int], Awaitable[str]]


_UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed
- UPDATE \"Next Steps\" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


@dataclass(frozen=True, slots=True)
class CutPointResult:
    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


@dataclass(slots=True)
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: list[AgentMessage]
    turn_prefix_messages: list[AgentMessage]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    file_ops: FileOperations = field(default_factory=create_file_ops)
    settings: CompactionSettings = field(default_factory=CompactionSettings)


def calculate_context_tokens(usage: Usage) -> int:
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read
        + usage.cache_write
    )


def _get_assistant_usage(message: AgentMessage) -> Usage | None:
    if isinstance(message, AssistantMessage):
        if message.stop_reason not in {"aborted", "error"}:
            return message.usage
    return None


def estimate_tokens(message: AgentMessage) -> int:
    chars = 0
    if isinstance(message, UserMessage):
        for user_block in message.content:
            if isinstance(user_block, TextContent):
                chars += len(user_block.text)
            else:
                chars += 4_800
        return max(1, (chars + 3) // 4)

    if isinstance(message, AssistantMessage):
        for assistant_block in message.content:
            if isinstance(assistant_block, TextContent):
                chars += len(assistant_block.text)
            elif isinstance(assistant_block, ToolCallBlock):
                chars += len(assistant_block.name) + len(repr(assistant_block.arguments))
            else:
                chars += len(getattr(assistant_block, "text", ""))
        return max(1, (chars + 3) // 4)

    if isinstance(message, ToolResultMessage):
        for result_block in message.content:
            for result_part in result_block.content:
                if isinstance(result_part, TextContent):
                    chars += len(result_part.text)
                else:
                    chars += 4_800
        return max(1, (chars + 3) // 4)

    return 0


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    last_usage: Usage | None = None
    last_usage_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[index])
        if usage is not None:
            last_usage = usage
            last_usage_index = index
            break

    if last_usage is None:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(
            tokens=estimated,
            usage_tokens=0,
            trailing_tokens=estimated,
            last_usage_index=None,
        )

    assert last_usage_index is not None
    usage_tokens = calculate_context_tokens(last_usage)
    trailing = sum(
        estimate_tokens(message) for message in messages[last_usage_index + 1 :]
    )
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=last_usage_index,
    )


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: CompactionSettings,
) -> bool:
    if not settings.enabled or context_window <= 0:
        return False
    return context_tokens > context_window - settings.reserve_tokens


def _branch_summary_message(summary: str, timestamp: float) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=f"Branch summary: {summary}")],
        timestamp=timestamp,
        stop_reason="end_turn",
    )


def _compaction_summary_message(summary: str, timestamp: float) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=summary)],
        timestamp=timestamp,
        stop_reason="end_turn",
    )


def get_message_from_entry(entry: SessionEntry) -> AgentMessage | None:
    if entry.type == "message" and isinstance(
        entry.payload, (UserMessage, AssistantMessage, ToolResultMessage)
    ):
        return entry.payload
    if entry.type == "branch_summary":
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        summary = payload.get("summary")
        if isinstance(summary, str) and summary:
            return _branch_summary_message(summary, entry.timestamp)
    if entry.type == "compaction":
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        summary = payload.get("summary")
        if isinstance(summary, str) and summary:
            return _compaction_summary_message(summary, entry.timestamp)
    return None


def get_message_from_entry_for_compaction(entry: SessionEntry) -> AgentMessage | None:
    if entry.type == "compaction":
        return None
    return get_message_from_entry(entry)


def _find_valid_cut_points(
    entries: list[SessionEntry], start_index: int, end_index: int
) -> list[int]:
    cut_points: list[int] = []
    for index in range(start_index, end_index):
        entry = entries[index]
        if entry.type == "branch_summary":
            cut_points.append(index)
            continue
        if entry.type != "message":
            continue
        if isinstance(entry.payload, ToolResultMessage):
            continue
        cut_points.append(index)
    return cut_points


def _find_turn_start_index(
    entries: list[SessionEntry], entry_index: int, start_index: int
) -> int:
    for index in range(entry_index, start_index - 1, -1):
        entry = entries[index]
        if entry.type == "branch_summary":
            return index
        if entry.type != "message":
            continue
        if isinstance(entry.payload, UserMessage):
            return index
    return -1


def find_cut_point(
    entries: list[SessionEntry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    cut_points = _find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(
            first_kept_entry_index=start_index,
            turn_start_index=-1,
            is_split_turn=False,
        )

    accumulated = 0
    cut_index = cut_points[0]
    for index in range(end_index - 1, start_index - 1, -1):
        entry = entries[index]
        if entry.type != "message":
            continue
        if not isinstance(entry.payload, (UserMessage, AssistantMessage, ToolResultMessage)):
            continue
        accumulated += estimate_tokens(entry.payload)
        if accumulated >= keep_recent_tokens:
            for cut_point in cut_points:
                if cut_point >= index:
                    cut_index = cut_point
                    break
            break

    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        if prev_entry.type == "compaction":
            break
        if prev_entry.type == "message":
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = cut_entry.type == "message" and isinstance(cut_entry.payload, UserMessage)
    turn_start_index = -1 if is_user_message else _find_turn_start_index(entries, cut_index, start_index)
    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=(not is_user_message and turn_start_index != -1),
    )


def prepare_compaction(
    path_entries: list[SessionEntry],
    settings: CompactionSettings,
    current_messages: list[AgentMessage] | None = None,
) -> CompactionPreparation | None:
    if not path_entries:
        return None
    if path_entries[-1].type == "compaction":
        return None

    prev_compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if path_entries[index].type == "compaction":
            prev_compaction_index = index
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev_payload = path_entries[prev_compaction_index].payload
        if isinstance(prev_payload, dict):
            raw_summary = prev_payload.get("summary")
            if isinstance(raw_summary, str) and raw_summary:
                previous_summary = raw_summary
            first_kept = prev_payload.get("first_kept_entry_id") or prev_payload.get(
                "firstKeptEntryId"
            )
            if isinstance(first_kept, str):
                boundary_start = next(
                    (
                        idx
                        for idx, entry in enumerate(path_entries)
                        if entry.id == first_kept
                    ),
                    prev_compaction_index + 1,
                )
            else:
                boundary_start = prev_compaction_index + 1

    tokens_before = estimate_context_tokens(current_messages or []).tokens
    cut_point = find_cut_point(
        path_entries,
        boundary_start,
        len(path_entries),
        settings.keep_recent_tokens,
    )
    first_kept_entry = path_entries[cut_point.first_kept_entry_index]
    history_end = (
        cut_point.turn_start_index
        if cut_point.is_split_turn
        else cut_point.first_kept_entry_index
    )

    messages_to_summarize: list[AgentMessage] = []
    for index in range(boundary_start, history_end):
        message = get_message_from_entry_for_compaction(path_entries[index])
        if message is not None:
            messages_to_summarize.append(message)

    turn_prefix_messages: list[AgentMessage] = []
    if cut_point.is_split_turn:
        for index in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            message = get_message_from_entry_for_compaction(path_entries[index])
            if message is not None:
                turn_prefix_messages.append(message)

    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev_payload = path_entries[prev_compaction_index].payload
        if isinstance(prev_payload, dict):
            read_files = prev_payload.get("read_files") or prev_payload.get("readFiles")
            modified_files = prev_payload.get("modified_files") or prev_payload.get(
                "modifiedFiles"
            )
            if isinstance(read_files, list):
                file_ops.read.update(path for path in read_files if isinstance(path, str))
            if isinstance(modified_files, list):
                paths = [path for path in modified_files if isinstance(path, str)]
                file_ops.edited.update(paths)

    for message in messages_to_summarize:
        extract_file_ops_from_message(message, file_ops)
    for message in turn_prefix_messages:
        extract_file_ops_from_message(message, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry.id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )


async def generate_summary(
    current_messages: list[AgentMessage],
    summarizer: Summarizer,
    reserve_tokens: int,
    summarization_prompt: str,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
) -> str:
    base_prompt = (
        _UPDATE_SUMMARIZATION_PROMPT
        if previous_summary
        else summarization_prompt
    )
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    conversation_text = serialize_conversation(current_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        )
    prompt_text += base_prompt
    max_tokens = max(256, int(0.8 * reserve_tokens))
    return await summarizer(SUMMARIZATION_SYSTEM_PROMPT, prompt_text, max_tokens)


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    summarizer: Summarizer,
    reserve_tokens: int,
) -> str:
    conversation_text = serialize_conversation(messages)
    prompt_text = (
        f"<conversation>\n{conversation_text}\n</conversation>\n\n"
        f"{_TURN_PREFIX_SUMMARIZATION_PROMPT}"
    )
    max_tokens = max(256, int(0.5 * reserve_tokens))
    return await summarizer(SUMMARIZATION_SYSTEM_PROMPT, prompt_text, max_tokens)


async def compact(
    preparation: CompactionPreparation,
    summarizer: Summarizer,
    summarization_prompt: str,
    custom_instructions: str | None = None,
) -> CompactionResult:
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_task = (
            generate_summary(
                preparation.messages_to_summarize,
                summarizer,
                preparation.settings.reserve_tokens,
                summarization_prompt,
                custom_instructions,
                preparation.previous_summary,
            )
            if preparation.messages_to_summarize
            else _immediate("No prior history.")
        )
        turn_task = _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            summarizer,
            preparation.settings.reserve_tokens,
        )
        history_summary, turn_prefix_summary = await asyncio.gather(
            history_task,
            turn_task,
        )
        summary = (
            f"{history_summary}\n\n---\n\n"
            f"**Turn Context (split turn):**\n\n{turn_prefix_summary}"
        )
    else:
        summary = await generate_summary(
            preparation.messages_to_summarize,
            summarizer,
            preparation.settings.reserve_tokens,
            summarization_prompt,
            custom_instructions,
            preparation.previous_summary,
        )

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        details=CompactionDetails(
            read_files=read_files,
            modified_files=modified_files,
        ),
    )


async def _immediate(value: str) -> str:
    return value


__all__ = [
    "CompactionDetails",
    "CompactionPreparation",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
    "DEFAULT_COMPACTION_SETTINGS",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "calculate_context_tokens",
    "compact",
    "estimate_context_tokens",
    "estimate_tokens",
    "find_cut_point",
    "generate_summary",
    "get_message_from_entry",
    "get_message_from_entry_for_compaction",
    "prepare_compaction",
    "should_compact",
]
