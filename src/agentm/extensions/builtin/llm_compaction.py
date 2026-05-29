"""Builtin LLM-driven compaction extension.

Model: **full compress**. When the durable session branch crosses the
token threshold, every turn since the previous compaction is summarized into
a single ``user`` message that replaces the whole prior context — there is no
verbatim recent tail. Compression is **incremental / chained**: each pass
folds the new turns into the previous summary (an ``update`` rewrite, not a
raw append) so the running checkpoint stays bounded and internally
consistent.

The summary tags each chunk with a ``[Turn N]`` marker. The original turns
are never deleted from the session tree (``get_branch`` keeps them), so the
agent can recover exact detail for any turn via the ``read_history`` tool.
Turn numbering is shared with that tool through ``core.lib.enumerate_turns``.

Per issue #76 the compaction kernel owns no English prompt text; this atom
resolves the active bodies via ``api.get_service("prompt_templates").get_prompt``
(populated by the ``compaction_prompts`` atom) and threads them into the
engine. When the prompts atom is not installed, this atom falls back to
neutral empty strings and emits a diagnostic so users see the drift.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    BeforeSendToLlmEvent,
    MessageEnd,
    Model,
    TextContent,
    Tool,
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
from agentm.core.abi.events import DiagnosticEvent
from agentm.core.abi.session import ENTRY_TYPE_COMPACTION, SessionEntry
from agentm.core.abi.termination import Aborted, ProviderError
from agentm.core.abi.tool import (
    FILE_OP_EDIT,
    FILE_OP_METADATA_KEY,
    FILE_OP_READ,
    FILE_OP_WRITE,
)
from agentm.core.lib import Turn, enumerate_turns
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import AfterCompactEvent, BeforeCompactEvent
from agentm.core.abi.extension import CommandSpec, ExtensionAPI, ProviderConfig


# Prompt registry keys. Kept in sync with ``compaction_prompts.py``;
# §11 forbids atom-to-atom imports so we duplicate the canonical names
# here instead of importing them.
_PROMPT_SUMMARIZATION_SYSTEM = "compaction.summarization_system"
_PROMPT_SUMMARIZATION = "compaction.summarization"
_PROMPT_UPDATE_SUMMARIZATION = "compaction.update_summarization"


MANIFEST = ExtensionManifest(
    name="llm_compaction",
    description="LLM-driven full-compress compaction for long session branches.",
    registers=(
        "event:before_send_to_llm",
        "event:before_compact",
        "event:after_compact",
        "command:compact",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "default": True},
            "reserve_tokens": {"type": "integer", "minimum": 1, "default": 16_384},
            "tool_result_max_chars": {"type": "integer", "minimum": 1, "default": 2_000},
            "custom_instructions": {"type": "string"},
        },
        "additionalProperties": False,
    },
    requires=("compaction_prompts", "prompt_templates"),
    tier=2,
)


# === Compaction engine =====================================================
#
# Per issue #76 the engine keeps **zero** literal English prompt text: prompts
# are passed in as parameters by the caller (resolved via the prompt registry).

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


def serialize_messages(
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


def serialize_turns(
    turns: list[Turn],
    *,
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> str:
    """Render turns with ``[Turn N]`` headers so the summarizer can cite them."""

    blocks: list[str] = []
    for turn in turns:
        body = serialize_messages(turn.messages, tool_result_max_chars=tool_result_max_chars)
        blocks.append(f"[Turn {turn.index}]\n{body}")
    return "\n\n".join(blocks)


Summarizer = Callable[[str, str, int], Awaitable[str]]


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


# A neutral fallback used by tests and by the graceful-degradation path when
# no prompts atom is installed. Empty strings mean "no extra instructions";
# the engine still serializes the conversation faithfully.
EMPTY_COMPACTION_PROMPTS = CompactionPrompts(
    summarization_system="",
    update_summarization="",
)


@dataclass(slots=True)
class CompactionPreparation:
    turns_to_summarize: list[Turn]
    covered_through_turn: int
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
    threshold = context_window - settings.reserve_tokens
    if threshold <= 0:
        # ``reserve_tokens`` >= the whole window: there is no usable headroom
        # to reserve, so the threshold would be non-positive and fire on every
        # turn. Treat this as a misconfiguration (or a tiny test model) and
        # disable auto-compaction rather than thrash. Lower ``reserve_tokens``
        # to compact on small-window models.
        return False
    return context_tokens > threshold


def _find_previous_compaction(branch: list[SessionEntry]) -> SessionEntry | None:
    for entry in reversed(branch):
        if entry.type == ENTRY_TYPE_COMPACTION:
            return entry
    return None


def prepare_compaction(
    branch: list[SessionEntry],
    settings: CompactionSettings,
    current_messages: list[AgentMessage] | None = None,
    tools: ToolRegistry | None = None,
) -> CompactionPreparation | None:
    """Collect the turns to fold into the running summary.

    Full-compress: every turn after the previous compaction's
    ``covered_through_turn`` is summarized; nothing is kept verbatim. Returns
    ``None`` when there is no new material to compress.
    """

    all_turns = enumerate_turns(branch)
    if not all_turns:
        return None

    previous = _find_previous_compaction(branch)
    previous_summary: str | None = None
    covered_before = 0
    file_ops = create_file_ops()
    if previous is not None and isinstance(previous.payload, dict):
        raw_summary = previous.payload.get("summary")
        if isinstance(raw_summary, str) and raw_summary:
            previous_summary = raw_summary
        raw_covered = previous.payload.get("covered_through_turn")
        if isinstance(raw_covered, int):
            covered_before = raw_covered
        read_files = previous.payload.get("read_files")
        modified_files = previous.payload.get("modified_files")
        if isinstance(read_files, list):
            file_ops.read.update(p for p in read_files if isinstance(p, str))
        if isinstance(modified_files, list):
            file_ops.edited.update(p for p in modified_files if isinstance(p, str))

    new_turns = [turn for turn in all_turns if turn.index > covered_before]
    if not new_turns:
        return None

    for turn in new_turns:
        for message in turn.messages:
            extract_file_ops_from_message(message, file_ops, tools)

    return CompactionPreparation(
        turns_to_summarize=new_turns,
        covered_through_turn=all_turns[-1].index,
        tokens_before=estimate_context_tokens(current_messages or []).tokens,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )


async def generate_summary(
    turns: list[Turn],
    summarizer: Summarizer,
    reserve_tokens: int,
    summarization_prompt: str,
    prompts: CompactionPrompts,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    *,
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> str:
    base_prompt = (
        prompts.update_summarization
        if previous_summary
        else summarization_prompt
    )
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    conversation_text = serialize_turns(turns, tool_result_max_chars=tool_result_max_chars)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        )
    prompt_text += base_prompt
    max_tokens = max(256, int(0.8 * reserve_tokens))
    return await summarizer(prompts.summarization_system, prompt_text, max_tokens)


async def compact(
    preparation: CompactionPreparation,
    summarizer: Summarizer,
    summarization_prompt: str,
    custom_instructions: str | None = None,
    prompts: CompactionPrompts | None = None,
) -> CompactionResult:
    """Run a full-compress pass and return a :class:`CompactionResult`.

    ``summarization_prompt`` is the body used for the first summarization;
    ``prompts.update_summarization`` is used once a previous summary exists
    (incremental chaining). When ``prompts`` is ``None`` the engine falls back
    to :data:`EMPTY_COMPACTION_PROMPTS` so the call still succeeds — callers
    should emit a diagnostic in that case.
    """

    resolved_prompts = prompts if prompts is not None else EMPTY_COMPACTION_PROMPTS
    summary = await generate_summary(
        preparation.turns_to_summarize,
        summarizer,
        preparation.settings.reserve_tokens,
        summarization_prompt,
        resolved_prompts,
        custom_instructions,
        preparation.previous_summary,
        tool_result_max_chars=preparation.settings.tool_result_max_chars,
    )

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)
    return CompactionResult(
        summary=summary,
        covered_through_turn=preparation.covered_through_turn,
        tokens_before=preparation.tokens_before,
        details=CompactionDetails(
            read_files=read_files,
            modified_files=modified_files,
        ),
    )


# === Atom install ==========================================================


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    settings = CompactionSettings(
        enabled=bool(config.get("enabled", True)),
        reserve_tokens=int(config.get("reserve_tokens", 16_384)),
        tool_result_max_chars=int(config.get("tool_result_max_chars", 2_000)),
    )
    custom_instructions = config.get("custom_instructions")
    if not isinstance(custom_instructions, str):
        custom_instructions = None

    async def _run_compaction(
        reason: str, session_messages: list[AgentMessage], est_tokens: int
    ) -> list[AgentMessage] | None:
        """Compact ``session_messages`` and append the compaction entry.

        Returns the rebuilt session messages on success, or ``None`` when there
        is nothing to compact / no provider. Shared by the automatic overflow
        path and the on-demand ``/compact`` command.
        """
        provider = api.provider
        model = api.model
        if provider is None or model is None:
            return None

        branch = api.session.get_branch()
        preparation = prepare_compaction(
            branch, settings, current_messages=session_messages, tools=list(api.tools)
        )
        if preparation is None:
            return None

        prompts, summarization_body = await _resolve_prompts(api)

        await api.events.emit(
            BeforeCompactEvent.CHANNEL,
            BeforeCompactEvent(messages=session_messages, reason=reason),
        )

        result = await compact(
            preparation,
            _ProviderSummarizer(provider, model),
            summarization_body,
            custom_instructions,
            prompts=prompts,
        )

        # The entry records only ``covered_through_turn`` (the chaining
        # cursor): full-compress keeps no verbatim tail, so it omits the
        # kernel's optional ``first_kept_entry_id`` seam and lets
        # ``build_session_context`` rebuild the context as ``[user(summary)]``
        # plus anything appended after this entry.
        details = {
            "reason": reason,
            "reserve_tokens": settings.reserve_tokens,
            "covered_through_turn": result.covered_through_turn,
            "estimated_tokens_before": est_tokens,
            "summary": result.summary,
            "read_files": result.details.read_files,
            "modified_files": result.details.modified_files,
        }
        entry_id = api.session.append_entry("compaction", details)
        details["entry_id"] = entry_id

        rebuilt_messages = api.session.get_messages()
        await api.events.emit(
            AfterCompactEvent.CHANNEL,
            AfterCompactEvent(
                summary=result.summary,
                kept_message_count=len(rebuilt_messages),
                discarded_message_count=max(0, len(session_messages) - len(rebuilt_messages)),
                details=details,
            ),
        )
        return rebuilt_messages

    async def before_send_to_llm(event: BeforeSendToLlmEvent) -> None:
        model = api.model
        if model is None:
            return
        session_messages = api.session.get_messages()
        usage_estimate = estimate_context_tokens(session_messages)
        if not should_compact(usage_estimate.tokens, model.context_window, settings):
            return
        rebuilt = await _run_compaction(
            "llm_auto_overflow", session_messages, usage_estimate.tokens
        )
        if rebuilt is not None:
            event.messages[:] = rebuilt

    async def compact_command(_args: str, _api: ExtensionAPI) -> None:
        # On-demand compaction. Skips the overflow gate (the user asked for it)
        # but still no-ops when there is nothing summarisable. Feedback reaches
        # the user via the AfterCompactEvent the shared path emits.
        session_messages = api.session.get_messages()
        usage_estimate = estimate_context_tokens(session_messages)
        rebuilt = await _run_compaction("manual", session_messages, usage_estimate.tokens)
        if rebuilt is None:
            await api.events.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="info",
                    source="compaction",
                    message="Nothing to compact yet.",
                ),
            )

    api.on(BeforeSendToLlmEvent.CHANNEL, before_send_to_llm)
    api.register_command(
        "compact",
        CommandSpec(
            description="Compact this session's history now to free up context.",
            handler=compact_command,
        ),
    )


async def _resolve_prompts(api: ExtensionAPI) -> tuple[CompactionPrompts, str]:
    """Pull prompt bodies from the registry; emit a diagnostic if missing.

    Returns a 2-tuple ``(prompts, summarization_body)`` where ``prompts`` is
    a :class:`CompactionPrompts` and ``summarization_body`` is the
    fresh-summarization prompt body.

    When any required prompt is missing — i.e. the ``compaction_prompts``
    atom is not installed — we substitute empty strings and emit a single
    ``warning`` diagnostic. The compaction call still goes through; the LLM
    sees an empty instruction trailer and a neutral system prompt, which
    degrades quality but avoids a hard crash.
    """

    registry = api.get_service("prompt_templates")
    if registry is None:
        system = summarization = update = None
    else:
        system = registry.get_prompt(_PROMPT_SUMMARIZATION_SYSTEM)
        summarization = registry.get_prompt(_PROMPT_SUMMARIZATION)
        update = registry.get_prompt(_PROMPT_UPDATE_SUMMARIZATION)

    missing = [
        name
        for name, body in (
            (_PROMPT_SUMMARIZATION_SYSTEM, system),
            (_PROMPT_SUMMARIZATION, summarization),
            (_PROMPT_UPDATE_SUMMARIZATION, update),
        )
        if not body
    ]
    if missing:
        await api.events.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="warning",
                source="llm_compaction",
                message=(
                    "compaction_prompts atom not installed (missing prompts: "
                    f"{missing!r}); proceeding with empty prompt bodies — "
                    "summary quality will degrade. Install "
                    "extensions.builtin.compaction_prompts to restore defaults."
                ),
            ),
        )

    return (
        CompactionPrompts(
            summarization_system=system or "",
            update_summarization=update or "",
        ),
        summarization or "",
    )


class _ProviderSummarizer:
    def __init__(self, provider: ProviderConfig, model: Model) -> None:
        self._provider = provider
        self._model = model

    async def __call__(self, system_prompt: str, prompt_text: str, max_tokens: int) -> str:
        summary_model = type(self._model)(
            id=self._model.id,
            provider=self._model.provider,
            context_window=self._model.context_window,
            max_output_tokens=min(max_tokens, self._model.max_output_tokens),
            metadata=dict(getattr(self._model, "metadata", {})),
        )
        messages: list[AgentMessage] = [
            UserMessage(
                role="user",
                content=[TextContent(type="text", text=prompt_text)],
                timestamp=0.0,
            )
        ]
        final_message: AssistantMessage | None = None
        async for stream_event in self._provider.stream_fn(
            messages=messages,
            model=summary_model,
            tools=[],
            system=system_prompt,
            signal=None,
            thinking="off",
        ):
            if isinstance(stream_event, MessageEnd):
                final_message = stream_event.message
        if final_message is None:
            raise RuntimeError("Summarization stream ended without a final assistant message")
        if final_message.stop_reason == "error":
            raise RuntimeError("Summarization provider returned an error stop_reason")

        text = "\n".join(
            block.text for block in final_message.content if isinstance(block, TextContent)
        ).strip()
        if not text:
            raise RuntimeError("Summarization provider returned empty text")
        return text
