"""Branch summarization helpers for session-tree navigation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agentm.core.abi import AgentMessage
from agentm.core.abi.session import (
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    SessionEntry,
    SessionTree,
)

from .compaction import estimate_tokens, get_message_from_entry
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

@dataclass(frozen=True, slots=True)
class BranchSummaryResult:
    summary: str | None = None
    read_files: list[str] | None = None
    modified_files: list[str] | None = None
    aborted: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CollectEntriesResult:
    entries: list[SessionEntry]
    common_ancestor_id: str | None


@dataclass(frozen=True, slots=True)
class BranchPreparation:
    messages: list[AgentMessage]
    file_ops: FileOperations
    total_tokens: int


@dataclass(frozen=True, slots=True)
class GenerateBranchSummaryOptions:
    reserve_tokens: int = 16_384
    custom_instructions: str | None = None
    replace_instructions: bool = False


def collect_entries_for_branch_summary(
    session: SessionTree,
    old_leaf_id: str | None,
    target_id: str,
) -> CollectEntriesResult:
    if not old_leaf_id:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)

    old_path = {entry.id for entry in session.get_branch(old_leaf_id)}
    target_path = session.get_branch(target_id)
    common_ancestor_id: str | None = None
    for entry in reversed(target_path):
        if entry.id in old_path:
            common_ancestor_id = entry.id
            break

    entries: list[SessionEntry] = []
    current: str | None = old_leaf_id
    while current and current != common_ancestor_id:
        current_entry = session.get_entry(current)
        if current_entry is None:
            break
        entries.append(current_entry)
        current = current_entry.parent_id
    entries.reverse()
    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


def prepare_branch_entries(
    entries: list[SessionEntry],
    token_budget: int = 0,
    tools: ToolRegistry | None = None,
) -> BranchPreparation:
    messages: list[AgentMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0

    for entry in entries:
        if entry.type != ENTRY_TYPE_BRANCH_SUMMARY:
            continue
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        read_files = payload.get("read_files") or payload.get("readFiles")
        modified_files = payload.get("modified_files") or payload.get("modifiedFiles")
        if isinstance(read_files, list):
            file_ops.read.update(path for path in read_files if isinstance(path, str))
        if isinstance(modified_files, list):
            paths = [path for path in modified_files if isinstance(path, str)]
            file_ops.edited.update(paths)

    for entry in reversed(entries):
        message = get_message_from_entry(entry)
        if message is None:
            continue
        extract_file_ops_from_message(message, file_ops, tools)
        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry.type in {ENTRY_TYPE_COMPACTION, ENTRY_TYPE_BRANCH_SUMMARY} and total_tokens < int(
                token_budget * 0.9
            ):
                messages.insert(0, message)
                total_tokens += tokens
            break
        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


async def generate_branch_summary(
    entries: list[SessionEntry],
    summarizer: Summarizer,
    branch_summary_prompt: str,
    branch_summary_preamble: str,
    summarization_system_prompt: str,
    options: GenerateBranchSummaryOptions | None = None,
    tools: ToolRegistry | None = None,
) -> BranchSummaryResult:
    opts = options or GenerateBranchSummaryOptions()
    prep = prepare_branch_entries(entries, tools=tools)
    if not prep.messages:
        return BranchSummaryResult(summary="No content to summarize")

    instructions = branch_summary_prompt
    if opts.replace_instructions and opts.custom_instructions:
        instructions = opts.custom_instructions
    elif opts.custom_instructions:
        instructions = f"{branch_summary_prompt}\n\nAdditional focus: {opts.custom_instructions}"

    prompt_text = (
        f"<conversation>\n{serialize_conversation(prep.messages)}\n</conversation>\n\n"
        f"{instructions}"
    )
    try:
        summary = await summarizer(
            summarization_system_prompt,
            prompt_text,
            2048,
        )
    except Exception as exc:  # noqa: BLE001
        return BranchSummaryResult(error=str(exc))

    summary = branch_summary_preamble + summary
    read_files, modified_files = compute_file_lists(prep.file_ops)
    summary += format_file_operations(read_files, modified_files)
    return BranchSummaryResult(
        summary=summary or "No summary generated",
        read_files=read_files,
        modified_files=modified_files,
    )


__all__ = [
    "BranchPreparation",
    "BranchSummaryResult",
    "CollectEntriesResult",
    "GenerateBranchSummaryOptions",
    "collect_entries_for_branch_summary",
    "generate_branch_summary",
    "prepare_branch_entries",
]
