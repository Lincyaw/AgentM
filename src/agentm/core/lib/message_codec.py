"""Round-trip serialization for AgentMessage types.

Serialize: dataclass → JSON-compatible dict (``serialize_payload``).
Deserialize: JSON dict → typed ``UserMessage``/``AssistantMessage``/
``ToolResultMessage`` (``deserialize_payload``).

Used by ``SessionManager`` for on-disk persistence. The bytes encoding
(``{"__bytes__": [...]}``\ ) is the established on-disk format; do not
change without a migration.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from agentm.core.abi.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def serialize_payload(payload: Any) -> Any:
    if is_dataclass(payload) and not isinstance(payload, type):
        return {
            f.name: serialize_payload(getattr(payload, f.name))
            for f in fields(payload)
        }
    if isinstance(payload, (list, tuple)):
        return [serialize_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {str(k): serialize_payload(v) for k, v in payload.items()}
    if isinstance(payload, bytes):
        return {"__bytes__": list(payload)}
    return payload


def deserialize_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        if isinstance(payload, list):
            return [deserialize_payload(item) for item in payload]
        return payload

    if "__bytes__" in payload:
        raw = payload["__bytes__"]
        return bytes(int(item) for item in raw) if isinstance(raw, list) else b""

    role = payload.get("role")
    if role == "user":
        return UserMessage(
            role="user",
            content=_user_blocks(payload.get("content", [])),
            timestamp=float(payload.get("timestamp", 0.0)),
        )
    if role == "assistant":
        return AssistantMessage(
            role="assistant",
            content=_assistant_blocks(payload.get("content", [])),
            timestamp=float(payload.get("timestamp", 0.0)),
            stop_reason=payload.get("stop_reason"),
            usage=_usage(payload.get("usage")),
        )
    if role == "tool_result":
        return ToolResultMessage(
            role="tool_result",
            content=_tool_result_blocks(payload.get("content", [])),
            timestamp=float(payload.get("timestamp", 0.0)),
        )

    return {str(k): deserialize_payload(v) for k, v in payload.items()}


def _usage(payload: Any) -> Usage | None:
    if not isinstance(payload, dict):
        return None
    return Usage(
        input_tokens=int(payload.get("input_tokens", 0)),
        output_tokens=int(payload.get("output_tokens", 0)),
        cache_read=int(payload.get("cache_read", 0)),
        cache_write=int(payload.get("cache_write", 0)),
    )


def _user_blocks(payload: Any) -> list[TextContent | ImageContent]:
    if not isinstance(payload, list):
        return []
    blocks: list[TextContent | ImageContent] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") == "text":
            blocks.append(TextContent(type="text", text=str(raw.get("text", ""))))
        elif raw.get("type") == "image":
            blocks.append(
                ImageContent(
                    type="image",
                    data=deserialize_payload(raw.get("data", {"__bytes__": []})),
                    mime_type=str(raw.get("mime_type", "application/octet-stream")),
                )
            )
    return blocks


def _assistant_blocks(
    payload: Any,
) -> list[TextContent | ToolCallBlock | ThinkingBlock]:
    if not isinstance(payload, list):
        return []
    blocks: list[TextContent | ToolCallBlock | ThinkingBlock] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("type")
        if kind == "text":
            blocks.append(TextContent(type="text", text=str(raw.get("text", ""))))
        elif kind == "tool_call":
            args = raw.get("arguments", {})
            blocks.append(
                ToolCallBlock(
                    type="tool_call",
                    id=str(raw.get("id", "")),
                    name=str(raw.get("name", "")),
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        elif kind == "thinking":
            signature = raw.get("signature")
            blocks.append(
                ThinkingBlock(
                    type="thinking",
                    text=str(raw.get("text", "")),
                    signature=str(signature) if isinstance(signature, str) else None,
                )
            )
    return blocks


def _tool_result_blocks(payload: Any) -> list[ToolResultBlock]:
    if not isinstance(payload, list):
        return []
    blocks: list[ToolResultBlock] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        blocks.append(
            ToolResultBlock(
                type="tool_result",
                tool_call_id=str(raw.get("tool_call_id", "")),
                content=_user_blocks(raw.get("content", [])),
                is_error=bool(raw.get("is_error", False)),
            )
        )
    return blocks


__all__ = ["deserialize_payload", "serialize_payload"]
