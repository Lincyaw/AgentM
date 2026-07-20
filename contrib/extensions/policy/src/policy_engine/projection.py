# code-health: ignore-file[AM025] -- policy engine normalizes untyped YAML, SQLite, and runtime event payloads
"""Reusable offline policy projection over trajectory-derived events."""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from agentm.core.abi.messages import ImageContent, TextContent, ToolResultBlock
from agentm.core.abi.trajectory import Turn, TurnCheckpoint

from .evaluator import PolicyEvaluator
from .entity import EntityRegistry
from .ifc import IFCEngine
from .persistence import PolicyPersistence
from .state import PolicyState
from .types import RuleInstance, ToolArgs, ToolLogEntry


@dataclass(frozen=True, slots=True)
class PolicyToolCallProjectionEvent:
    turn_index: int
    round_index: int
    tool_call_id: str
    tool_name: str
    args: ToolArgs


@dataclass(frozen=True, slots=True)
class PolicyToolResultProjectionEvent:
    turn_index: int
    round_index: int
    tool_call_id: str
    tool_name: str
    args: ToolArgs
    result_text: str | None
    is_error: bool
    extras: Mapping[str, object] | None = None
    raw_result: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PolicyTurnEndProjectionEvent:
    turn_index: int


PolicyProjectionEvent = (
    PolicyToolCallProjectionEvent
    | PolicyToolResultProjectionEvent
    | PolicyTurnEndProjectionEvent
)


@dataclass(frozen=True, slots=True)
class PolicyProjectionResult:
    session_id: str
    turns: int
    tool_calls: int
    effects: int
    eval_errors: int


class PolicyProjector:
    """Project source-agnostic policy events into state and persistence rows."""

    def __init__(
        self,
        *,
        session_id: str,
        rules: Sequence[RuleInstance],
        persistence: PolicyPersistence | None = None,
        ifc: IFCEngine | None = None,
    ) -> None:
        self._session_id = session_id
        self._state = PolicyState()
        self._entity_registry = EntityRegistry()
        self._ifc = ifc
        self._persistence = persistence
        self._evaluator = PolicyEvaluator(list(rules), self._state)
        self._evaluator.set_entity_evidence_fn(self._entity_registry.entity_evidence)
        if self._ifc:
            self._evaluator.set_ifc_engine(self._ifc)
        if self._persistence:
            self._evaluator.set_cross_session_fn(self._persistence.count_cross_session)
            self._evaluator.set_eval_error_fn(self._record_eval_error)
        self._pending_tool_args: dict[str, ToolArgs] = {}
        self._persisted_effect_count = 0
        self._turns = 0
        self._tool_calls = 0
        self._eval_errors = 0

    @property
    def state(self) -> PolicyState:
        return self._state

    @property
    def entity_registry(self) -> EntityRegistry:
        return self._entity_registry

    def process(self, event: PolicyProjectionEvent) -> None:
        if isinstance(event, PolicyToolCallProjectionEvent):
            self._on_tool_call(event)
        elif isinstance(event, PolicyToolResultProjectionEvent):
            self._on_tool_result(event)
        else:
            self._on_turn_end(event)

    def finish(self) -> PolicyProjectionResult:
        self._persist_new_effects()
        self._queue_state_snapshot()
        self._queue_session_summary()
        self._flush_policy_persistence()
        return PolicyProjectionResult(
            session_id=self._session_id,
            turns=self._turns,
            tool_calls=self._tool_calls,
            effects=self._persisted_effect_count,
            eval_errors=self._eval_errors,
        )

    def _on_tool_call(self, event: PolicyToolCallProjectionEvent) -> None:
        self._sync_to_turn(event.turn_index)
        self._entity_registry.extract_structural(
            event.tool_name,
            event.args,
            self._state.turn_count,
        )
        self._pending_tool_args[event.tool_call_id] = event.args
        self._evaluator.evaluate(
            channel="tool_call",
            tool_name=event.tool_name,
            args=event.args,
        )
        self._persist_new_effects()
        self._queue_tool_event_pre(event)
        self._queue_state_snapshot()
        self._flush_policy_persistence()

    def _on_tool_result(self, event: PolicyToolResultProjectionEvent) -> None:
        self._sync_to_turn(event.turn_index)
        args = self._pending_tool_args.pop(event.tool_call_id, event.args)
        error = event.result_text if event.is_error else None

        result_for_state: dict[str, object] = {}
        if event.result_text:
            result_for_state["text"] = event.result_text
        if error:
            result_for_state["error"] = error
        metadata = _projection_metadata(event)
        result_for_state.update(metadata)

        self._state.record_tool_call(
            tool_name=event.tool_name,
            args=args,
            result=result_for_state,
        )
        self._tool_calls += 1

        if error:
            self._entity_registry.record_tool_failure(
                event.tool_name,
                args,
                self._state.turn_count,
                error,
            )
        else:
            self._entity_registry.record_tool_success(
                event.tool_name,
                args,
                self._state.turn_count,
            )

        if self._ifc and event.result_text:
            self._ifc.process_tool_result(event.tool_name, event.result_text)
        if event.result_text:
            self._entity_registry.extract_lexical(
                event.result_text,
                self._state.turn_count,
                "tool_result",
            )

        result_for_eval: dict[str, str | None] = {
            "text": event.result_text,
            "error": error,
        }
        self._evaluator.evaluate(
            channel="tool_result",
            tool_name=event.tool_name,
            result=result_for_eval,
        )
        self._persist_new_effects()
        self._queue_tool_event_post(
            event,
            args,
            raw_result=event.raw_result or _projection_raw_result(event),
            processed=_projection_processed_result(event, error, metadata),
        )
        self._queue_state_snapshot()
        self._flush_policy_persistence()

    def _on_turn_end(self, event: PolicyTurnEndProjectionEvent) -> None:
        self._sync_to_turn(event.turn_index)
        source_turn = event.turn_index
        recent_entries = [
            entry
            for entry in self._state.tool_log.entries()
            if entry.get("turn") == source_turn
        ]
        turn_summary = {
            "turn_index": source_turn,
            "tool_calls_count": len(recent_entries),
            "tool_names_set": {entry.get("tool") for entry in recent_entries},
            "error_count": sum(1 for entry in recent_entries if entry.get("error")),
            "files_modified": {
                entry.get("path")
                for entry in recent_entries
                if entry.get("path") and entry.get("tool") in ("edit", "write")
            },
        }
        self._state.turn_summary.record(source_turn, turn_summary)

        self._state.advance_turn()
        self._turns += 1
        self._evaluator.evaluate(channel="turn_committed", tool_name="")
        self._persist_new_effects()
        self._queue_turn_summary(source_turn, turn_summary)
        self._queue_state_snapshot()
        self._flush_policy_persistence()

    def _sync_to_turn(self, turn_index: int) -> None:
        while self._state.turn_count < turn_index:
            self._state.advance_turn()

    def _record_eval_error(
        self,
        rule_id: str,
        channel: str,
        tool_name: str,
        error: str,
    ) -> None:
        self._eval_errors += 1
        if not self._persistence:
            return
        self._persistence.queue_eval_error(
            session_id=self._session_id,
            turn=self._state.turn_count,
            rule_id=rule_id,
            channel=channel,
            tool_name=tool_name,
            error=error,
        )

    def _persist_new_effects(self) -> None:
        if not self._persistence:
            return
        records = list(self._state.effect_log.entries())
        for record in records[self._persisted_effect_count :]:
            self._persistence.queue_effect(self._session_id, record)
        self._persisted_effect_count = len(records)

    def _queue_tool_event_pre(self, event: PolicyToolCallProjectionEvent) -> None:
        if not self._persistence:
            return
        self._persistence.queue_tool_event(
            session_id=self._session_id,
            turn=self._state.turn_count,
            phase="pre",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=event.args,
            taint_labels=self._taint_labels(event.args),
        )

    def _queue_tool_event_post(
        self,
        event: PolicyToolResultProjectionEvent,
        args: ToolArgs,
        *,
        raw_result: Mapping[str, object | None],
        processed: dict[str, object | None],
    ) -> None:
        if not self._persistence:
            return
        entry = self._latest_tool_log_entry()
        if entry is not None:
            processed["error_fingerprint"] = entry.error_fingerprint
            processed["error_category"] = entry.error_category
        self._persistence.queue_tool_event(
            session_id=self._session_id,
            turn=self._state.turn_count,
            phase="post",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=args,
            entry=entry,
            result=raw_result,
            processed=processed,
        )

    def _queue_turn_summary(
        self,
        turn_index: int,
        turn_summary: Mapping[str, object],
    ) -> None:
        if self._persistence:
            self._persistence.queue_turn_summary(
                self._session_id,
                turn_index,
                turn_summary,
            )

    def _queue_state_snapshot(self) -> None:
        if not self._persistence:
            return
        self._persistence.queue_file_state_snapshot(
            self._session_id,
            self._state.file_state.entries(),
        )
        self._persistence.queue_entity_state_snapshot(
            self._session_id,
            self._entity_registry.entries(),
        )

    def _queue_session_summary(self) -> None:
        if not self._persistence:
            return
        tool_entries = list(self._state.tool_log.entries())
        effect_entries = list(self._state.effect_log.entries())
        file_entries = list(self._state.file_state.entries())
        tool_counts = Counter(entry.tool for entry in tool_entries)
        exit_code_counts = Counter(
            str(entry.exit_code)
            for entry in tool_entries
            if entry.exit_code is not None
        )
        durations = [entry.duration_ms for entry in tool_entries]
        duration_total = sum(durations)
        self._persistence.queue_session_summary(
            self._session_id,
            {
                "session_id": self._session_id,
                "turn_count": self._state.turn_count,
                "tool_call_count": len(tool_entries),
                "tool_error_count": sum(1 for entry in tool_entries if entry.error),
                "tool_log_entries": len(tool_entries),
                "effect_log_entries": len(effect_entries),
                "entities_tracked": len(self._entity_registry.entries()),
                "tool_counts": dict(tool_counts),
                "exit_code_counts": dict(exit_code_counts),
                "duration_ms_total": duration_total,
                "duration_ms_avg": (
                    round(duration_total / len(durations), 2) if durations else 0.0
                ),
                "duration_ms_max": max(durations, default=0),
                "file_count": len(file_entries),
                "file_read_count": sum(entry.read_count for entry in file_entries),
                "file_write_count": sum(entry.write_count for entry in file_entries),
                "error_count": sum(1 for entry in tool_entries if entry.error),
            },
        )

    def _flush_policy_persistence(self) -> None:
        if self._persistence:
            self._persistence.flush()

    def _taint_labels(self, args: ToolArgs) -> tuple[str, ...]:
        if not self._ifc:
            return ()
        return tuple(sorted(self._ifc.check_event_taint(args)))

    def _latest_tool_log_entry(self) -> ToolLogEntry | None:
        entries = self._state.tool_log.entries()
        return entries[-1] if entries else None


def project_events(
    *,
    session_id: str,
    events: Iterable[PolicyProjectionEvent],
    rules: Sequence[RuleInstance],
    persistence: PolicyPersistence | None = None,
    ifc: IFCEngine | None = None,
) -> PolicyProjectionResult:
    projector = PolicyProjector(
        session_id=session_id,
        rules=rules,
        persistence=persistence,
        ifc=ifc,
    )
    for event in events:
        projector.process(event)
    return projector.finish()


def events_from_turns(
    turns: Iterable[Turn | TurnCheckpoint],
) -> Iterable[PolicyProjectionEvent]:
    for turn in turns:
        for round_index, round_ in enumerate(turn.rounds):
            for record in round_.tool_results:
                args = dict(record.call.arguments)
                yield PolicyToolCallProjectionEvent(
                    turn_index=turn.index,
                    round_index=round_index,
                    tool_call_id=record.call.id,
                    tool_name=record.call.name,
                    args=args,  # type: ignore[arg-type]
                )
                yield PolicyToolResultProjectionEvent(
                    turn_index=turn.index,
                    round_index=round_index,
                    tool_call_id=record.call.id,
                    tool_name=record.call.name,
                    args=args,  # type: ignore[arg-type]
                    result_text=_tool_result_text(record.result.content),
                    is_error=record.result.is_error,
                    extras=_mapping_or_none(record.result.extras),
                    raw_result=_tool_result_json(record.result),
                )
        yield PolicyTurnEndProjectionEvent(turn_index=turn.index)


def _tool_result_text(content: Sequence[object]) -> str | None:
    text = "".join(block.text for block in content if isinstance(block, TextContent))
    return text or None


def _mapping_or_none(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _tool_result_json(result: ToolResultBlock) -> Mapping[str, object]:
    return {
        "is_error": result.is_error,
        "content": [_content_block_json(block) for block in result.content],
        "extras": _mapping_or_value(result.extras),
    }


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


def _projection_raw_result(
    event: PolicyToolResultProjectionEvent,
) -> Mapping[str, object | None]:
    return {
        "is_error": event.is_error,
        "text": event.result_text,
        "extras": _mapping_or_value(event.extras or {}),
    }


def _projection_processed_result(
    event: PolicyToolResultProjectionEvent,
    error: str | None,
    metadata: Mapping[str, object],
) -> dict[str, object | None]:
    processed: dict[str, object | None] = {
        "is_error": event.is_error,
        "text_length": len(event.result_text or ""),
        "error": error,
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


def _projection_metadata(
    event: PolicyToolResultProjectionEvent,
) -> dict[str, object]:
    metadata: dict[str, object] = {"result_length": len(event.result_text or "")}
    extras = event.extras or {}
    runtime = extras.get("agentm_runtime")
    runtime_extras = runtime if isinstance(runtime, Mapping) else {}

    exit_code = _first_int(
        extras.get("exit_code"),
        extras.get("return_code"),
        extras.get("returncode"),
    )
    if exit_code is not None:
        metadata["exit_code"] = exit_code

    duration_ms = _first_int(
        extras.get("duration_ms"),
        runtime_extras.get("duration_ms"),
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


def _mapping_or_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mapping_or_value(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mapping_or_value(item) for item in value]
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
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


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None
