"""Builtin LLM-driven compaction extension.

Model: **full compress**. When the durable session branch crosses the
provider-reported token threshold, every turn since the previous compaction is
summarized into a single ``user`` message that replaces the whole prior
context — there is no verbatim recent tail. Compression is **incremental / chained**: each pass
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
engine. Prompt name constants live in ``core.abi.compaction`` so both atoms
share a single source of truth. When the prompts atom is not installed, this
atom falls back to neutral empty strings and emits a diagnostic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    Aborted,
    AfterCompactEvent,
    AgentMessage,
    AssistantMessage,
    BeforeCompactEvent,
    BeforeSendToLlmEvent,
    CommandSpec,
    CompactionDetails,
    CompactionPrompts,
    CompactionResult,
    CompactionSettings,
    ContextUsageSnapshot,
    DiagnosticEvent,
    ENTRY_TYPE_COMPACTION,
    ExtensionAPI,
    FILE_OP_EDIT,
    FILE_OP_METADATA_KEY,
    FILE_OP_READ,
    FILE_OP_WRITE,
    MessageEnd,
    Model,
    PROMPT_SUMMARIZATION,
    PROMPT_SUMMARIZATION_SYSTEM,
    PROMPT_UPDATE_SUMMARIZATION,
    ProviderConfig,
    ProviderError,
    SessionEntry,
    TextContent,
    Tool,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from agentm.core.lib import (
    Turn,
    count_text_tokens,
    enumerate_turns,
    truncate_text_tokens,
)
from agentm.extensions import ExtensionManifest


class LlmCompactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_result_max_tokens: int = Field(gt=0)
    enabled: bool = True
    reserve_tokens: int = Field(default=16_384, gt=0)
    custom_instructions: str | None = None


COMPACTION_CONTROL_SERVICE = "llm_compaction.control"


MANIFEST = ExtensionManifest(
    name="llm_compaction",
    description="LLM-driven full-compress compaction for long session branches.",
    registers=(
        "event:before_send_to_llm",
        "event:before_compact",
        "event:after_compact",
        "command:compact",
    ),
    config_schema=LlmCompactionConfig,
    requires=("compaction_prompts", "prompt_templates"),
    api_version=1,
    tier=2,
)

# === Compaction engine =====================================================
#
# Per issue #76 the engine keeps **zero** literal English prompt text: prompts
# are passed in as parameters by the caller (resolved via the prompt registry).

# A flexible tool-registry shape for ``extract_file_ops_from_message``:
# either a name->tool mapping, or any iterable of tools (we'll index by
# ``tool.name`` ourselves).
ToolRegistry = Mapping[str, Tool] | Sequence[Tool]


@dataclass(slots=True)
class _FileOpTracker:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def create_file_ops() -> _FileOpTracker:
    return _FileOpTracker()


def _normalize_registry(tools: ToolRegistry | None) -> Mapping[str, Tool]:
    if tools is None:
        return {}
    if isinstance(tools, Mapping):
        return tools
    return {tool.name: tool for tool in tools}


def extract_file_ops_from_message(
    message: AgentMessage,
    file_ops: _FileOpTracker,
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


def compute_file_lists(file_ops: _FileOpTracker) -> tuple[list[str], list[str]]:
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
            "<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>"
        )
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _truncate_for_summary(
    text: str,
    max_tokens: int,
    *,
    model_name: str | None,
) -> str:
    truncated = truncate_text_tokens(text, max_tokens, model=model_name)
    if not truncated.was_truncated:
        return text
    return (
        truncated.text
        + f"\n\n[... {truncated.truncated_tokens} more tokens truncated]"
    )


def serialize_messages(
    messages: list[AgentMessage],
    *,
    tool_result_max_tokens: int,
    model_name: str | None = None,
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
                    + _truncate_for_summary(
                        content,
                        tool_result_max_tokens,
                        model_name=model_name,
                    )
                )

    return "\n\n".join(parts)


def serialize_turns(
    turns: list[Turn],
    *,
    tool_result_max_tokens: int,
    model_name: str | None = None,
) -> str:
    """Render turns with ``[Turn N]`` headers so the summarizer can cite them."""

    blocks: list[str] = []
    for turn in turns:
        body = serialize_messages(
            turn.messages,
            tool_result_max_tokens=tool_result_max_tokens,
            model_name=model_name,
        )
        blocks.append(f"[Turn {turn.index}]\n{body}")
    return "\n\n".join(blocks)


Summarizer = Callable[[str, str, int], Awaitable[str]]

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
    measured_tokens_before: int
    estimated_trailing_tokens_before: int
    previous_summary: str | None
    settings: CompactionSettings
    file_ops: _FileOpTracker = field(default_factory=create_file_ops)


def calculate_context_tokens(usage: Usage) -> int:
    return (
        usage.input_tokens + usage.output_tokens + usage.cache_read + usage.cache_write
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


def count_message_tokens(message: AgentMessage, *, model_name: str | None) -> int:
    text_parts: list[str] = []
    if isinstance(message, UserMessage):
        for user_block in message.content:
            if isinstance(user_block, TextContent):
                text_parts.append(user_block.text)
            else:
                text_parts.append(repr(user_block))
        return count_text_tokens("\n".join(text_parts), model=model_name)

    if isinstance(message, AssistantMessage):
        for assistant_block in message.content:
            if isinstance(assistant_block, TextContent):
                text_parts.append(assistant_block.text)
            elif isinstance(assistant_block, ToolCallBlock):
                text_parts.append(
                    f"{assistant_block.name}({repr(assistant_block.arguments)})"
                )
            else:
                text_parts.append(str(getattr(assistant_block, "text", "")))
        return count_text_tokens("\n".join(text_parts), model=model_name)

    if isinstance(message, ToolResultMessage):
        for result_block in message.content:
            for result_part in result_block.content:
                if isinstance(result_part, TextContent):
                    text_parts.append(result_part.text)
                else:
                    text_parts.append(repr(result_part))
        return count_text_tokens("\n".join(text_parts), model=model_name)

    return 0


def capture_context_usage(
    messages: list[AgentMessage],
    *,
    model_name: str | None,
) -> ContextUsageSnapshot:
    last_usage: Usage | None = None
    last_usage_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[index])
        if usage is not None:
            last_usage = usage
            last_usage_index = index
            break

    if last_usage is None:
        estimated = sum(
            count_message_tokens(message, model_name=model_name)
            for message in messages
        )
        return ContextUsageSnapshot(
            tokens=estimated,
            measured_tokens=0,
            estimated_trailing_tokens=estimated,
            last_usage_index=None,
        )

    assert last_usage_index is not None
    measured_tokens = calculate_context_tokens(last_usage)
    estimated_trailing_tokens = sum(
        count_message_tokens(message, model_name=model_name)
        for message in messages[last_usage_index + 1 :]
    )
    return ContextUsageSnapshot(
        tokens=measured_tokens + estimated_trailing_tokens,
        measured_tokens=measured_tokens,
        estimated_trailing_tokens=estimated_trailing_tokens,
        last_usage_index=last_usage_index,
    )


def should_compact(
    tokens: int,
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
    return tokens > threshold


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
    model_name: str | None = None,
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
    if len(new_turns) < 2:
        return None

    for turn in new_turns:
        for message in turn.messages:
            extract_file_ops_from_message(message, file_ops, tools)

    usage_snapshot = capture_context_usage(
        current_messages or [],
        model_name=model_name,
    )
    return CompactionPreparation(
        turns_to_summarize=new_turns,
        covered_through_turn=all_turns[-1].index,
        tokens_before=usage_snapshot.tokens,
        measured_tokens_before=usage_snapshot.measured_tokens,
        estimated_trailing_tokens_before=usage_snapshot.estimated_trailing_tokens,
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
    tool_result_max_tokens: int,
    model_name: str | None = None,
) -> str:
    base_prompt = (
        prompts.update_summarization if previous_summary else summarization_prompt
    )
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    conversation_text = serialize_turns(
        turns,
        tool_result_max_tokens=tool_result_max_tokens,
        model_name=model_name,
    )
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
        tool_result_max_tokens=preparation.settings.tool_result_max_tokens,
        model_name=getattr(summarizer, "model_name", None),
    )

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)
    return CompactionResult(
        summary=summary,
        covered_through_turn=preparation.covered_through_turn,
        tokens_before=preparation.tokens_before,
        measured_tokens_before=preparation.measured_tokens_before,
        estimated_trailing_tokens_before=(
            preparation.estimated_trailing_tokens_before
        ),
        details=CompactionDetails(
            read_files=read_files,
            modified_files=modified_files,
        ),
    )


# === Atom install ==========================================================


class _LlmCompactionRuntime:
    """Install-time wiring and event handlers for LLM compaction."""

    def __init__(self, api: ExtensionAPI, config: LlmCompactionConfig) -> None:
        self._api = api
        self._settings = CompactionSettings(
            enabled=config.enabled,
            reserve_tokens=config.reserve_tokens,
            tool_result_max_tokens=config.tool_result_max_tokens,
        )
        self._custom_instructions = config.custom_instructions

    def install(self) -> None:
        self._api.set_service(COMPACTION_CONTROL_SERVICE, _CompactionControl(self))
        self._api.on(BeforeSendToLlmEvent.CHANNEL, self.before_send_to_llm)
        self._api.register_command(
            "compact",
            CommandSpec(
                description="Compact this session's history now to free up context.",
                handler=self.compact_command,
            ),
        )

    def set_auto_compaction_enabled(self, enabled: bool) -> bool:
        self._settings = replace(self._settings, enabled=enabled)
        return self._settings.enabled

    @property
    def auto_compaction_enabled(self) -> bool:
        return self._settings.enabled
    async def _run_compaction(
        self,
        reason: str,
        session_messages: list[AgentMessage],
    ) -> list[AgentMessage] | None:
        """Compact ``session_messages`` and append the compaction entry.

        Returns the rebuilt session messages on success, or ``None`` when there
        is nothing to compact / no provider. Shared by the automatic overflow
        path and the on-demand ``/compact`` command.
        """
        provider = self._api.provider
        model = self._api.model
        if provider is None or model is None:
            return None

        branch = self._api.session.get_branch()
        preparation = prepare_compaction(
            branch,
            self._settings,
            current_messages=session_messages,
            tools=list(self._api.tools),
            model_name=model.id,
        )
        if preparation is None:
            return None

        prompts, summarization_body = await _resolve_prompts(self._api)

        await self._api.events.emit(
            BeforeCompactEvent.CHANNEL,
            BeforeCompactEvent(messages=session_messages, reason=reason),
        )

        result = await compact(
            preparation,
            _ProviderSummarizer(provider, model),
            summarization_body,
            self._custom_instructions,
            prompts=prompts,
        )

        session_meta_lines = [
            "<!-- session-metadata",
            f"session_id: {self._api.session_id}",
            f"trace_id: {self._api.root_session_id}",
        ]
        if self._api.scenario:
            session_meta_lines.append(f"scenario: {self._api.scenario}")
        session_meta_lines.append(
            f"covered_through_turn: {result.covered_through_turn}"
        )
        session_meta_lines.append("-->")
        final_summary = "\n".join(session_meta_lines) + "\n\n" + result.summary

        details = {
            "reason": reason,
            "reserve_tokens": self._settings.reserve_tokens,
            "covered_through_turn": result.covered_through_turn,
            "tokens_before": result.tokens_before,
            "measured_tokens_before": result.measured_tokens_before,
            "estimated_trailing_tokens_before": (
                result.estimated_trailing_tokens_before
            ),
            "summary": final_summary,
            "read_files": result.details.read_files,
            "modified_files": result.details.modified_files,
        }
        entry_id = self._api.session.append_entry("compaction", details)
        details["entry_id"] = entry_id

        rebuilt_messages = self._api.session.get_messages()
        await self._api.events.emit(
            AfterCompactEvent.CHANNEL,
            AfterCompactEvent(
                summary=final_summary,
                kept_message_count=len(rebuilt_messages),
                discarded_message_count=max(
                    0, len(session_messages) - len(rebuilt_messages)
                ),
                details=details,
            ),
        )
        return rebuilt_messages

    async def before_send_to_llm(self, event: BeforeSendToLlmEvent) -> None:
        model = self._api.model
        if model is None:
            return
        session_messages = self._api.session.get_messages()
        usage_snapshot = capture_context_usage(
            session_messages,
            model_name=model.id,
        )
        if not should_compact(
            usage_snapshot.tokens,
            model.context_window,
            self._settings,
        ):
            return
        rebuilt = await self._run_compaction(
            "llm_auto_overflow",
            session_messages,
        )
        if rebuilt is not None:
            event.messages[:] = rebuilt

    async def compact_command(self, _args: str, _api: ExtensionAPI) -> None:
        # On-demand compaction. Skips the overflow gate (the user asked for it)
        # but still no-ops when there is nothing summarisable. Feedback reaches
        # the user via the AfterCompactEvent the shared path emits.
        session_messages = self._api.session.get_messages()
        rebuilt = await self._run_compaction("manual", session_messages)
        if rebuilt is None:
            await self._api.events.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="warning",
                    source="compaction",
                    message="Not enough messages to compact.",
                ),
            )


class _CompactionControl:
    def __init__(self, runtime: _LlmCompactionRuntime) -> None:
        self._runtime = runtime

    def set_enabled(self, enabled: bool) -> bool:
        return self._runtime.set_auto_compaction_enabled(enabled)

    @property
    def enabled(self) -> bool:
        return self._runtime.auto_compaction_enabled


def install(api: ExtensionAPI, config: LlmCompactionConfig) -> None:
    _LlmCompactionRuntime(api, config).install()


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

    from agentm.core.abi import PROMPT_TEMPLATES_SERVICE
    registry = api.get_service(PROMPT_TEMPLATES_SERVICE)
    if registry is None:
        system = summarization = update = None
    else:
        system = registry.get_prompt(PROMPT_SUMMARIZATION_SYSTEM)
        summarization = registry.get_prompt(PROMPT_SUMMARIZATION)
        update = registry.get_prompt(PROMPT_UPDATE_SUMMARIZATION)

    missing = [
        name
        for name, body in (
            (PROMPT_SUMMARIZATION_SYSTEM, system),
            (PROMPT_SUMMARIZATION, summarization),
            (PROMPT_UPDATE_SUMMARIZATION, update),
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
        self.model_name = model.id

    async def __call__(
        self, system_prompt: str, prompt_text: str, max_tokens: int
    ) -> str:
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
            raise RuntimeError(
                "Summarization stream ended without a final assistant message"
            )
        if final_message.stop_reason == "error":
            raise RuntimeError("Summarization provider returned an error stop_reason")

        text = "\n".join(
            block.text
            for block in final_message.content
            if isinstance(block, TextContent)
        ).strip()
        if not text:
            raise RuntimeError("Summarization provider returned empty text")
        return text
