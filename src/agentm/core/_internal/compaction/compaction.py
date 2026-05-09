"""Pure compaction logic adapted from pi-mono's LLM-driven compaction.

Per issue #76, the kernel keeps **zero** literal English prompt text and
**zero** string-literal entry-type dispatch. Prompts are passed in as
parameters by callers (typically resolved via
``ExtensionAPI.prompt_templates``); entry materialization consults the
``ENTRY_MATERIALIZERS`` registry on ``agentm.core.abi.session``.
"""

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
    CompactionPrompts,
    CompactionResult,
    CompactionSettings,
    ContextUsageEstimate,
)
from agentm.core.abi.session import (
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
    SessionEntry,
)
from agentm.core.abi.termination import Aborted, ProviderError

from .utils import (
    FileOperations,
    ToolRegistry,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

Summarizer = Callable[[str, str, int], Awaitable[str]]


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


# A neutral fallback used by tests and by the graceful-degradation path when
# no prompts atom is installed. Empty strings mean "no extra instructions";
# the engine still serializes the conversation faithfully.
EMPTY_COMPACTION_PROMPTS = CompactionPrompts(
    summarization_system="",
    update_summarization="",
    turn_prefix_summarization="",
)


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
        # Prefer the kernel-canonical TerminationHint when both shipped
        # adapters set it. Fall back to the raw stop_reason string only
        # when termination is None (legacy / non-streaming code paths).
        if message.termination is not None:
            if not isinstance(message.termination, (Aborted, ProviderError)):
                return message.usage
        elif message.stop_reason not in {"aborted", "error"}:
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


def get_message_from_entry(entry: SessionEntry) -> AgentMessage | None:
    """Materialize a session entry into an ``AgentMessage`` via the registry.

    The ``ENTRY_MATERIALIZERS`` registry is populated by atoms at install
    time. When no materializer is registered for ``entry.type``, this
    function returns ``None`` (graceful degradation — the missing atom
    diagnostic is the harness's job; the pure engine layer stays
    side-effect free).
    """

    materializer = ENTRY_MATERIALIZERS.get(entry.type)
    if materializer is None:
        return None
    return materializer.to_message(entry)


def get_message_from_entry_for_compaction(entry: SessionEntry) -> AgentMessage | None:
    """Same as ``get_message_from_entry`` but skips ``compaction`` entries.

    The compaction engine emits the synthesized summary itself; pre-existing
    compaction entries on the path are intentionally elided so the new
    summary is built only over un-summarized history.
    """

    if entry.type == ENTRY_TYPE_COMPACTION:
        return None
    return get_message_from_entry(entry)


def _find_valid_cut_points(
    entries: list[SessionEntry], start_index: int, end_index: int
) -> list[int]:
    cut_points: list[int] = []
    for index in range(start_index, end_index):
        entry = entries[index]
        if entry.type == ENTRY_TYPE_BRANCH_SUMMARY:
            cut_points.append(index)
            continue
        if entry.type != ENTRY_TYPE_MESSAGE:
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
        if entry.type == ENTRY_TYPE_BRANCH_SUMMARY:
            return index
        if entry.type != ENTRY_TYPE_MESSAGE:
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
        if entry.type != ENTRY_TYPE_MESSAGE:
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
        if prev_entry.type == ENTRY_TYPE_COMPACTION:
            break
        if prev_entry.type == ENTRY_TYPE_MESSAGE:
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = cut_entry.type == ENTRY_TYPE_MESSAGE and isinstance(cut_entry.payload, UserMessage)
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
    tools: ToolRegistry | None = None,
) -> CompactionPreparation | None:
    if not path_entries:
        return None
    if path_entries[-1].type == ENTRY_TYPE_COMPACTION:
        return None

    prev_compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if path_entries[index].type == ENTRY_TYPE_COMPACTION:
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
        extract_file_ops_from_message(message, file_ops, tools)
    for message in turn_prefix_messages:
        extract_file_ops_from_message(message, file_ops, tools)

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
    prompts: CompactionPrompts,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    *,
    tool_result_max_chars: int | None = None,
) -> str:
    base_prompt = (
        prompts.update_summarization
        if previous_summary
        else summarization_prompt
    )
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    if tool_result_max_chars is None:
        conversation_text = serialize_conversation(current_messages)
    else:
        conversation_text = serialize_conversation(
            current_messages, tool_result_max_chars=tool_result_max_chars
        )
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        )
    prompt_text += base_prompt
    max_tokens = max(256, int(0.8 * reserve_tokens))
    return await summarizer(prompts.summarization_system, prompt_text, max_tokens)


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    summarizer: Summarizer,
    reserve_tokens: int,
    prompts: CompactionPrompts,
    *,
    tool_result_max_chars: int | None = None,
) -> str:
    if tool_result_max_chars is None:
        conversation_text = serialize_conversation(messages)
    else:
        conversation_text = serialize_conversation(
            messages, tool_result_max_chars=tool_result_max_chars
        )
    prompt_text = (
        f"<conversation>\n{conversation_text}\n</conversation>\n\n"
        f"{prompts.turn_prefix_summarization}"
    )
    max_tokens = max(256, int(0.5 * reserve_tokens))
    return await summarizer(prompts.summarization_system, prompt_text, max_tokens)


async def compact(
    preparation: CompactionPreparation,
    summarizer: Summarizer,
    summarization_prompt: str,
    custom_instructions: str | None = None,
    prompts: CompactionPrompts | None = None,
) -> CompactionResult:
    """Run the compaction pass and return a :class:`CompactionResult`.

    ``summarization_prompt`` is the body used for fresh summarizations.
    ``prompts`` carries the system prompt + the update / turn-prefix bodies
    used in the incremental and split-turn paths. When ``prompts`` is
    ``None`` (the prompts atom was not installed) the engine falls back to
    :data:`EMPTY_COMPACTION_PROMPTS` so the call still succeeds — callers
    should emit a diagnostic in that case.
    """

    resolved_prompts = prompts if prompts is not None else EMPTY_COMPACTION_PROMPTS
    tool_result_max_chars = preparation.settings.tool_result_max_chars
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_task = (
            generate_summary(
                preparation.messages_to_summarize,
                summarizer,
                preparation.settings.reserve_tokens,
                summarization_prompt,
                resolved_prompts,
                custom_instructions,
                preparation.previous_summary,
                tool_result_max_chars=tool_result_max_chars,
            )
            if preparation.messages_to_summarize
            else _immediate("No prior history.")
        )
        turn_task = _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            summarizer,
            preparation.settings.reserve_tokens,
            resolved_prompts,
            tool_result_max_chars=tool_result_max_chars,
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
            resolved_prompts,
            custom_instructions,
            preparation.previous_summary,
            tool_result_max_chars=tool_result_max_chars,
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
    "CompactionPrompts",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
    "DEFAULT_COMPACTION_SETTINGS",
    "EMPTY_COMPACTION_PROMPTS",
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
