# code-health: ignore-file[AM025] -- execution worker and wire boundaries validate cross-process payloads
"""Strict JSON wire format for process-isolated tool results."""

# code-health: ignore-file[AM022] -- validates untyped process-wire JSON

from __future__ import annotations

import base64
from collections.abc import Mapping
import json
import math
from typing import Any, Literal, cast

from agentm.core.abi.messages import ImageContent, TextContent
from agentm.core.abi.tool import (
    ToolContinue,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)


PROCESS_RESULT_SCHEMA_VERSION = 1
_OutcomeKind = Literal["result", "continue", "terminate"]


def encode_tool_arguments(arguments: Mapping[str, object]) -> str:
    """Encode one process invocation argument object."""

    encoded = _encode_json_value(arguments, "tool arguments")
    if not isinstance(encoded, dict):
        raise TypeError("tool arguments must be an object")
    return (
        json.dumps(
            encoded,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def decode_tool_arguments(payload: str) -> dict[str, Any]:
    """Decode one process invocation argument object."""

    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError("tool arguments are not valid JSON") from exc
    decoded = _decode_json_value(value, "tool arguments")
    if not isinstance(decoded, dict):
        raise ValueError("tool arguments must be an object")
    return decoded


def encode_tool_output(output: ToolResult | ToolOutcome) -> str:
    """Encode a process result without executable object deserialization."""

    kind: _OutcomeKind
    result: ToolResult
    reason: str | None = None
    if isinstance(output, ToolResult):
        kind = "result"
        result = output
    elif isinstance(output, ToolContinue):
        kind = "continue"
        result = output.result
    elif isinstance(output, ToolTerminate):
        kind = "terminate"
        result = output.result
        reason = output.reason
        if not isinstance(reason, str) or not reason:
            raise ValueError("terminating tool output requires a non-empty reason")
    else:
        raise TypeError(
            f"process entrypoint returned unsupported result: {type(output).__name__}"
        )

    payload: dict[str, Any] = {
        "schema_version": PROCESS_RESULT_SCHEMA_VERSION,
        "kind": kind,
        "result": _encode_result(result),
    }
    if reason is not None:
        payload["reason"] = reason
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def decode_tool_output(payload: str) -> ToolResult | ToolOutcome:
    """Decode and validate one process result document."""

    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError("process result is not valid JSON") from exc
    data = _object(value, "process result")
    _only_fields(data, {"schema_version", "kind", "result", "reason"}, "process result")
    version = data.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != PROCESS_RESULT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported process result schema version: {version!r}")
    kind = data.get("kind")
    if kind not in {"result", "continue", "terminate"}:
        raise ValueError(f"invalid process result kind: {kind!r}")
    result = _decode_result(data.get("result"))
    reason = data.get("reason")
    if kind == "terminate":
        if not isinstance(reason, str) or not reason:
            raise ValueError("terminating process result requires a non-empty reason")
        return ToolTerminate(result=result, reason=reason)
    if reason is not None:
        raise ValueError(f"{kind} process result cannot carry a reason")
    if kind == "continue":
        return ToolContinue(result=result)
    return result


def _encode_result(result: ToolResult) -> dict[str, Any]:
    if not isinstance(result, ToolResult):
        raise TypeError("tool outcome result must be a ToolResult")
    if not isinstance(result.is_error, bool):
        raise TypeError("tool result is_error must be a bool")
    content: list[dict[str, Any]] = []
    for block in result.content:
        if isinstance(block, TextContent):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            content.append(
                {
                    "type": "image",
                    "data_base64": base64.b64encode(block.data).decode("ascii"),
                    "mime_type": block.mime_type,
                }
            )
        else:
            raise TypeError(
                "tool result content must contain TextContent or ImageContent"
            )
    return {
        "content": content,
        "is_error": result.is_error,
        "extras": _encode_json_value(result.extras, "tool result extras"),
    }


def _decode_result(value: object) -> ToolResult:
    data = _object(value, "tool result")
    _only_fields(data, {"content", "is_error", "extras"}, "tool result")
    raw_content = data.get("content")
    if not isinstance(raw_content, list):
        raise ValueError("tool result content must be a list")
    content: list[TextContent | ImageContent] = []
    for index, raw_block in enumerate(raw_content):
        block = _object(raw_block, f"tool result content[{index}]")
        block_type = block.get("type")
        if block_type == "text":
            _only_fields(
                block,
                {"type", "text"},
                f"tool result content[{index}]",
            )
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"tool result content[{index}].text must be a string")
            content.append(TextContent(type="text", text=text))
            continue
        if block_type == "image":
            _only_fields(
                block,
                {"type", "data_base64", "mime_type"},
                f"tool result content[{index}]",
            )
            encoded = block.get("data_base64")
            mime_type = block.get("mime_type")
            if not isinstance(encoded, str) or not isinstance(mime_type, str):
                raise ValueError(
                    f"tool result content[{index}] has invalid image fields"
                )
            try:
                image = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError(
                    f"tool result content[{index}] has invalid base64 data"
                ) from exc
            content.append(ImageContent(type="image", data=image, mime_type=mime_type))
            continue
        raise ValueError(
            f"tool result content[{index}] has invalid type {block_type!r}"
        )
    is_error = data.get("is_error")
    if not isinstance(is_error, bool):
        raise ValueError("tool result is_error must be a bool")
    return ToolResult(
        content=content,
        is_error=is_error,
        extras=_decode_json_value(data.get("extras"), "tool result extras"),
    )


def _encode_json_value(value: object, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} object keys must be strings")
        return {
            cast(str, key): _encode_json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} is not JSON-safe: {type(value).__name__}")


def _decode_json_value(value: object, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} object keys must be strings")
        return {
            key: _decode_json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _decode_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} is not JSON-safe: {type(value).__name__}")


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object")
    return value


def _only_fields(data: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"process result contains invalid JSON constant {value!r}")


__all__ = [
    "PROCESS_RESULT_SCHEMA_VERSION",
    "decode_tool_arguments",
    "decode_tool_output",
    "encode_tool_arguments",
    "encode_tool_output",
]
