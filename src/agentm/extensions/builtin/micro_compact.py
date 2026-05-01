"""Builtin ``micro_compact`` atom per extension-as-scenario §7."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from agentm.core.kernel import (
    AgentMessage,
    AssistantMessage,
    BeforeSendToLlmEvent,
    TextContent,
)
from agentm.extensions import ExtensionManifest
from agentm.harness.events import AfterCompactEvent, BeforeCompactEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="micro_compact",
    description="Auto-compacts context when token usage approaches the model window.",
    registers=("event:before_send_to_llm", "event:before_compact", "event:after_compact"),
    config_schema={
        "type": "object",
        "properties": {
            "threshold_pct": {"type": "number", "minimum": 0.0},
            "keep_last": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
    },
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    threshold_pct = float(config.get("threshold_pct", 0.85))
    keep_last = int(config.get("keep_last", 8))

    async def before_send_to_llm(event: BeforeSendToLlmEvent) -> None:
        model = api.model
        if model is None or model.context_window <= 0:
            return
        messages = event.messages
        estimated_tokens = _estimate_messages(messages)
        if estimated_tokens <= int(model.context_window * threshold_pct):
            return
        if len(messages) <= keep_last:
            return

        before = BeforeCompactEvent(messages=messages, reason="auto_overflow")
        await api.events.emit("before_compact", before)

        original_messages = list(before.messages)
        compacted_messages, summary_text = _compact_messages(original_messages, keep_last)
        branch = api.session.get_branch()
        message_entries = [entry for entry in branch if entry.type == "message"]
        first_kept_entry_id = (
            message_entries[-keep_last].id
            if len(message_entries) >= keep_last
            else (message_entries[0].id if message_entries else None)
        )
        details = {
            "reason": "auto_overflow",
            "threshold_pct": threshold_pct,
            "keep_last": keep_last,
            "estimated_tokens_before": estimated_tokens,
            "estimated_tokens_after": _estimate_messages(compacted_messages),
            "discarded_message_count": max(0, len(original_messages) - len(compacted_messages)),
            "summary": summary_text,
        }
        if first_kept_entry_id is not None:
            details["first_kept_entry_id"] = first_kept_entry_id
        entry_id = api.session.append_entry("compaction", details)
        details["entry_id"] = entry_id

        # Rebuild from SessionManager so the in-flight context matches the
        # persisted compaction entry instead of an ad-hoc synthetic message.
        messages[:] = api.session.get_messages()
        await api.events.emit(
            "after_compact",
            AfterCompactEvent(
                summary=summary_text,
                kept_message_count=len(messages),
                discarded_message_count=max(0, len(original_messages) - len(compacted_messages)),
                details=details,
            ),
        )

    api.on("before_send_to_llm", before_send_to_llm)


def _compact_messages(
    messages: list[AgentMessage],
    keep_last: int,
) -> tuple[list[AgentMessage], str]:
    kept_tail = list(messages[-keep_last:])
    dropped = list(messages[:-keep_last])
    summary_text = _summarize_messages(dropped)
    summary_message = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=summary_text)],
        timestamp=0.0,
        stop_reason="end_turn",
    )
    return [summary_message, *kept_tail], summary_text


def _summarize_messages(messages: list[AgentMessage]) -> str:
    if not messages:
        return "Compaction summary: no earlier messages."
    fragments: list[str] = []
    for msg in messages[-4:]:
        role = getattr(msg, "role", "unknown")
        fragments.append(f"{role}: {_message_text(msg)}")
    joined = " | ".join(fragments)
    return f"Compaction summary of {len(messages)} earlier messages: {joined}"[:1200]


def _message_text(message: AgentMessage) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
        elif getattr(block, "type", None) == "tool_call":
            parts.append(f"tool:{getattr(block, 'name', '?')}")
        elif getattr(block, "type", None) == "tool_result":
            parts.append("tool_result")
    if parts:
        return " ".join(parts)
    return _safe_string(message)


def _estimate_messages(messages: list[AgentMessage]) -> int:
    total = 0
    for message in messages:
        text = _message_text(message)
        total += max(1, len(text))
    return total


def _safe_string(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        return str(asdict(value))
    return str(value)
