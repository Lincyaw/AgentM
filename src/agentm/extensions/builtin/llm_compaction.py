"""Builtin LLM-driven compaction extension."""

from __future__ import annotations

from typing import Any

from agentm.core.abi.compaction import CompactionSettings
from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    BeforeSendToLlmEvent,
    MessageEnd,
    Model,
    TextContent,
    UserMessage,
)
from agentm.extensions import ExtensionManifest
from agentm.harness.events import AfterCompactEvent, BeforeCompactEvent
from agentm.harness.extension import ExtensionAPI, ProviderConfig


_SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\n"
    "Summary of that exploration:\n\n"
)

_BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


MANIFEST = ExtensionManifest(
    name="llm_compaction",
    description="LLM-driven semantic compaction for long session branches.",
    registers=("event:before_send_to_llm", "event:before_compact", "event:after_compact"),
    config_schema={
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "reserve_tokens": {"type": "integer", "minimum": 1},
            "keep_recent_tokens": {"type": "integer", "minimum": 1},
            "custom_instructions": {"type": "string"},
        },
        "additionalProperties": False,
    },
    tier=2,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    settings = CompactionSettings(
        enabled=bool(config.get("enabled", True)),
        reserve_tokens=int(config.get("reserve_tokens", 16_384)),
        keep_recent_tokens=int(config.get("keep_recent_tokens", 20_000)),
    )
    custom_instructions = config.get("custom_instructions")
    if not isinstance(custom_instructions, str):
        custom_instructions = None

    async def before_send_to_llm(event: BeforeSendToLlmEvent) -> None:
        provider = api.provider
        model = api.model
        if provider is None or model is None:
            return

        session_messages = api.session.get_messages()
        usage_estimate = api.compaction.estimate_context_tokens(session_messages)
        if not api.compaction.should_compact(
            usage_estimate.tokens, model.context_window, settings
        ):
            return

        branch = api.session.get_branch()
        preparation = api.compaction.prepare_compaction(
            branch, settings, current_messages=session_messages
        )
        if preparation is None:
            return
        if not preparation.messages_to_summarize and not preparation.turn_prefix_messages:
            return

        before = BeforeCompactEvent(messages=event.messages, reason="llm_auto_overflow")
        await api.events.emit(BeforeCompactEvent.CHANNEL, before)

        result = await api.compaction.compact(
            preparation,
            _ProviderSummarizer(provider, model),
            _SUMMARIZATION_PROMPT,
            custom_instructions,
        )

        details = {
            "reason": "llm_auto_overflow",
            "reserve_tokens": settings.reserve_tokens,
            "keep_recent_tokens": settings.keep_recent_tokens,
            "estimated_tokens_before": usage_estimate.tokens,
            "summary": result.summary,
            "first_kept_entry_id": result.first_kept_entry_id,
            "read_files": result.details.read_files,
            "modified_files": result.details.modified_files,
        }
        entry_id = api.session.append_entry("compaction", details)
        details["entry_id"] = entry_id

        rebuilt_messages = api.session.get_messages()
        event.messages[:] = rebuilt_messages
        await api.events.emit(
            AfterCompactEvent.CHANNEL,
            AfterCompactEvent(
                summary=result.summary,
                kept_message_count=len(rebuilt_messages),
                discarded_message_count=max(0, len(session_messages) - len(rebuilt_messages)),
                details=details,
            ),
        )

    api.on(BeforeSendToLlmEvent.CHANNEL, before_send_to_llm)


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
