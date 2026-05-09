"""Builtin LLM-driven compaction extension.

Per issue #76 the compaction kernel owns no English prompt text; this atom
resolves the active bodies via ``api.prompt_templates.get_prompt`` (populated
by the ``compaction_prompts`` atom) and threads them into the engine. When
the prompts atom is not installed, this atom falls back to neutral empty
strings and emits a diagnostic so users see the configuration drift.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi.compaction import CompactionPrompts, CompactionSettings
from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    BeforeSendToLlmEvent,
    MessageEnd,
    Model,
    TextContent,
    UserMessage,
)
from agentm.core.abi.events import DiagnosticEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.events import AfterCompactEvent, BeforeCompactEvent
from agentm.harness.extension import ExtensionAPI, ProviderConfig


# Prompt registry keys. Kept in sync with ``compaction_prompts.py``;
# §11 forbids atom-to-atom imports so we duplicate the canonical names
# here instead of importing them.
_PROMPT_SUMMARIZATION_SYSTEM = "compaction.summarization_system"
_PROMPT_SUMMARIZATION = "compaction.summarization"
_PROMPT_UPDATE_SUMMARIZATION = "compaction.update_summarization"
_PROMPT_TURN_PREFIX_SUMMARIZATION = "compaction.turn_prefix_summarization"


MANIFEST = ExtensionManifest(
    name="llm_compaction",
    description="LLM-driven semantic compaction for long session branches.",
    registers=("event:before_send_to_llm", "event:before_compact", "event:after_compact"),
    config_schema={
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "default": True},
            "reserve_tokens": {"type": "integer", "minimum": 1, "default": 16_384},
            "keep_recent_tokens": {"type": "integer", "minimum": 1, "default": 20_000},
            "custom_instructions": {"type": "string"},
        },
        "additionalProperties": False,
    },
    requires=("compaction_prompts",),
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
            branch, settings, current_messages=session_messages, tools=list(api.tools)
        )
        if preparation is None:
            return
        if not preparation.messages_to_summarize and not preparation.turn_prefix_messages:
            return

        prompts, summarization_body = await _resolve_prompts(api)

        before = BeforeCompactEvent(messages=event.messages, reason="llm_auto_overflow")
        await api.events.emit(BeforeCompactEvent.CHANNEL, before)

        result = await api.compaction.compact(
            preparation,
            _ProviderSummarizer(provider, model),
            summarization_body,
            custom_instructions,
            prompts=prompts,
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

    system = api.prompt_templates.get_prompt(_PROMPT_SUMMARIZATION_SYSTEM)
    summarization = api.prompt_templates.get_prompt(_PROMPT_SUMMARIZATION)
    update = api.prompt_templates.get_prompt(_PROMPT_UPDATE_SUMMARIZATION)
    turn_prefix = api.prompt_templates.get_prompt(_PROMPT_TURN_PREFIX_SUMMARIZATION)

    missing = [
        name
        for name, body in (
            (_PROMPT_SUMMARIZATION_SYSTEM, system),
            (_PROMPT_SUMMARIZATION, summarization),
            (_PROMPT_UPDATE_SUMMARIZATION, update),
            (_PROMPT_TURN_PREFIX_SUMMARIZATION, turn_prefix),
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
            turn_prefix_summarization=turn_prefix or "",
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
