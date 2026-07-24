# code-health: ignore-file[AM025] -- normalization decodes SQLite JSON payloads
"""Normalize source-specific tool records into IFG source events."""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping, Sequence

from agentm.core.abi.messages import ImageContent, TextContent, ToolResultBlock
from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from sqlalchemy.engine import Connection, RowMapping

from .types import IfgToolEvent
from .utils import _first_str, _loads, _mapping_str, _row_value


def tool_event_from_policy_row(row: RowMapping) -> IfgToolEvent:
    args = _loads(row["args_json"])
    result = _loads(row["result_json"])
    processed = _loads(row["processed_json"])
    state = _loads(row["state_json"])
    event_id = row["id"]
    cwd = _first_str(_row_value(row, "cwd"), _mapping_str(state, "cwd"))
    raw_evidence = {
        "policy_tool_event_id": event_id,
        "phase": row["phase"],
        "tool_name": row["tool_name"],
        "args": args,
        "result": result,
        "processed": processed,
        "state": state,
    }
    return IfgToolEvent(
        session_id=str(row["session_id"]),
        turn=int(row["turn"]),
        event_id=int(event_id) if event_id is not None else None,
        tool_call_id=str(row["tool_call_id"]) if row["tool_call_id"] else None,
        phase=str(row["phase"]),
        tool_name=str(row["tool_name"]),
        args=args,
        result=result,
        processed=processed,
        state=state,
        cwd=cwd,
        ts=float(row["ts"] or 0.0),
        source="policy_tool_events",
        raw_evidence=raw_evidence,
    )


def tool_events_from_turns(
    turns: Iterable[Turn | TurnCheckpoint],
    *,
    session_id: str,
    cwd: str | None = None,
) -> tuple[IfgToolEvent, ...]:
    """Convert committed/checkpoint trajectory turns to IFG post events.

    A trajectory ``ToolRecord`` already contains both the assistant call and final
    tool result, so direct IFG backfill emits only ``post`` events. Runtime and
    policy-persisted sources can still preserve distinct pre/post phases.
    """

    events: list[IfgToolEvent] = []
    for turn in turns:
        ts = _turn_timestamp(turn)
        for record in turn.tool_results:
            args = dict(record.call.arguments)
            result_text = _tool_result_text(record.result.content)
            extras = _mapping_or_none(record.result.extras) or {}
            result = _tool_result_json(record.result, text=result_text)
            processed = _processed_result(
                result_text=result_text,
                is_error=record.result.is_error,
                extras=extras,
            )
            events.append(
                IfgToolEvent(
                    session_id=session_id,
                    turn=turn.index,
                    event_id=None,
                    tool_call_id=record.call.id,
                    phase="post",
                    tool_name=record.call.name,
                    args=args,
                    result=result,
                    processed=processed,
                    state={"processed": processed},
                    cwd=cwd,
                    ts=ts,
                    source="trajectory",
                    raw_evidence={
                        "source": "trajectory",
                        "turn_id": turn.id,
                        "turn_index": turn.index,
                        "run_id": turn.run_id,
                        "run_step": turn.run_step,
                        "tool_call_id": record.call.id,
                        "tool_name": record.call.name,
                        "args": args,
                        "result": result,
                        "checkpoint": isinstance(turn, TurnCheckpoint),
                    },
                )
            )
    return tuple(events)


def preferred_tool_rows(conn: Connection, session_id: str) -> list[RowMapping]:
    rows = list(
        conn.exec_driver_sql(
            """
            SELECT *
            FROM policy_tool_events
            WHERE session_id = ?
            ORDER BY turn ASC, id ASC
            """,
            (session_id,),
        )
        .mappings()
        .all()
    )
    by_call: dict[str, RowMapping] = {}
    order: list[str] = []
    for row in rows:
        key = str(row["tool_call_id"] or f"event:{row['id']}")
        if key not in by_call:
            order.append(key)
            by_call[key] = row
            continue
        if by_call[key]["phase"] != "post" and row["phase"] == "post":
            by_call[key] = row
    return [by_call[key] for key in order]


def _turn_timestamp(turn: Turn | TurnCheckpoint) -> float:
    if isinstance(turn, TurnCheckpoint):
        return turn.updated_at
    return turn.timestamp


def _tool_result_text(content: Sequence[object]) -> str | None:
    text = "\n".join(block.text for block in content if isinstance(block, TextContent))
    return text or None


def _tool_result_json(
    result: ToolResultBlock,
    *,
    text: str | None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "is_error": result.is_error,
        "content": [_content_block_json(block) for block in result.content],
        "extras": _mapping_or_value(result.extras),
    }
    if text is not None:
        payload["text"] = text
    return payload


def _content_block_json(block: object) -> dict[str, object]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "mime_type": block.mime_type,
            "byte_length": len(block.data),
            "data_base64": base64.b64encode(block.data).decode("ascii"),
        }
    return {"type": type(block).__name__}


def _processed_result(
    *,
    result_text: str | None,
    is_error: bool,
    extras: Mapping[str, object],
) -> Mapping[str, object | None]:
    metadata = _result_metadata(extras, result_text=result_text)
    processed: dict[str, object | None] = {
        "is_error": is_error,
        "text_length": len(result_text or ""),
        "error": result_text if is_error else None,
        "exit_code": metadata.get("exit_code"),
        "duration_ms": metadata.get("duration_ms"),
        "result_content_hash": metadata.get("result_content_hash"),
        "content_hash": metadata.get("content_hash"),
        "previous_content_hash": metadata.get("previous_content_hash"),
        "result_length": metadata.get("result_length"),
    }
    for key in (
        "args_hash",
        "file_op",
        "path",
        "timed_out",
        "timeout_s",
        "bytes",
        "stdout_lines",
        "stderr_lines",
    ):
        if key in metadata:
            processed[key] = metadata[key]
    return processed


def _result_metadata(
    extras: Mapping[str, object],
    *,
    result_text: str | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {"result_length": len(result_text or "")}
    runtime = extras.get("agentm_runtime")
    runtime_extras = runtime if isinstance(runtime, Mapping) else {}
    exit_code = _first_int(
        extras.get("exit_code"),
        extras.get("return_code"),
        extras.get("returncode"),
        runtime_extras.get("exit_code"),
        runtime_extras.get("return_code"),
        runtime_extras.get("returncode"),
    )
    if exit_code is not None:
        metadata["exit_code"] = exit_code
    duration_ms = _first_int(
        extras.get("duration_ms"), runtime_extras.get("duration_ms")
    )
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms
    result_content_hash = _first_str(
        extras.get("result_content_hash"),
        runtime_extras.get("result_content_hash"),
    )
    if result_content_hash is not None:
        metadata["result_content_hash"] = result_content_hash
    for key in (
        "args_hash",
        "content_hash",
        "previous_content_hash",
        "file_op",
        "path",
        "timed_out",
        "timeout_s",
        "bytes",
        "stdout_lines",
        "stderr_lines",
    ):
        value = extras.get(key, runtime_extras.get(key))
        if value is not None:
            metadata[key] = value
    return metadata


def _mapping_or_none(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _mapping_or_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mapping_or_value(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mapping_or_value(item) for item in value]
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, bytearray):
        return {
            "encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    return value


def _first_int(*values: object) -> int | None:
    for value in values:
        coerced = _coerce_int(value)
        if coerced is not None:
            return coerced
    return None


def _coerce_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
