"""Builtin observability atom — streaming OTel-flavored JSONL of every
event-bus dispatch, handler invocation (with mutation diff), API
registration, LLM request, and turn summary. See
``.claude/designs/observability.md``.
"""

from __future__ import annotations

import atexit
import inspect
import json
import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import IO, Any

from agentm.core.abi import (
    BusPriority,
    ContextEvent,
    EventBusObserver,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.core.abi.messages import ToolCallBlock
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import discover_builtin
from agentm.harness.events import (
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    BeforeAgentStartEvent,
    BeforeCompactEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.harness.extension import (
    ExtensionAPI,
    Handler,
    current_installing_extension,
)
from agentm.harness.resource_writer import GitBackedResourceWriter

logger = logging.getLogger(__name__)


# Channels whose events are documented mutable (handlers may rewrite payload
# in place). We restrict mutation diffing to these to avoid the O(N·M) cost
# of double-serializing every event on every handler invocation.
_MUTABLE_CHANNELS = frozenset(
    {
        BeforeAgentStartEvent.CHANNEL,
        ContextEvent.CHANNEL,
        ToolCallEvent.CHANNEL,
        ToolResultEvent.CHANNEL,
        BeforeCompactEvent.CHANNEL,
    }
)


MANIFEST = ExtensionManifest(
    name="observability",
    description=(
        "Stream every event dispatch, handler invocation (with mutation "
        "diff), API registration, LLM request, and turn summary to an "
        "OTel-flavored JSONL file for diagnostics."
    ),
    registers=(
        "event:agent_start",
        "event:agent_end",
        "event:turn_start",
        "event:turn_end",
        "event:tool_call",
        "event:tool_result",
        "event:llm_request_start",
        "event:llm_request_end",
        "event:api_register",
        "event:api_send_user_message",
        "event:extension_install",
        "event:extension_reload",
        "event:session_ready",
        "event:session_shutdown",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "include_handler_records": {"type": "boolean"},
            "include_mutation_diff": {"type": "boolean"},
            "strict_sync_handlers": {"type": "boolean"},
            "max_queue": {"type": "integer"},
            "exclude_channels": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    },
)


# Channels whose per-emission spans add no diagnostic value over the
# already-recorded summary events. ``stream_delta`` fires once per LLM
# token chunk; assembled assistant messages arrive on ``turn_end``, so the
# raw deltas are pure bloat for trace consumers. Anything in this set is
# skipped from BOTH ``event.dispatch`` and ``handler.invoke`` records.
_DEFAULT_EXCLUDE_CHANNELS: frozenset[str] = frozenset({StreamDeltaEvent.CHANNEL})


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_ns() -> int:
    return time.time_ns()


def _serialize(value: Any, _depth: int = 0) -> Any:
    if _depth > 12:
        return "<max-depth>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', repr(value))}>"
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _serialize(getattr(value, f.name), _depth + 1)
            for f in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _serialize(val, _depth + 1) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize(item, _depth + 1) for item in value]
    return repr(value)


def _handler_label(handler: Handler) -> str:
    mod = getattr(handler, "__module__", None) or "<unknown>"
    qual = getattr(handler, "__qualname__", None) or repr(handler)
    return f"{mod}.{qual}"


def _deep_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        out: list[dict[str, Any]] = []
        keys = list(before.keys()) + [k for k in after if k not in before]
        for key in keys:
            out.extend(
                _deep_diff(
                    before.get(key, "<missing>"),
                    after.get(key, "<missing>"),
                    f"{path}.{key}" if path else str(key),
                )
            )
            if len(out) > 50:
                out.append({"path": path or "<root>", "_truncated": True})
                break
        return out
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [
                {
                    "path": path or "<root>",
                    "before_len": len(before),
                    "after_len": len(after),
                }
            ]
        out2: list[dict[str, Any]] = []
        for i, (b, a) in enumerate(zip(before, after, strict=False)):
            out2.extend(_deep_diff(b, a, f"{path}[{i}]"))
            if len(out2) > 50:
                out2.append({"path": path or "<root>", "_truncated": True})
                break
        return out2
    return [{"path": path or "<root>", "before": before, "after": after}]


def _format_traceback(exc: BaseException | None) -> str | None:
    if exc is None or exc.__traceback__ is None:
        return None
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class _Sink:
    """Async file sink: hot path is ``queue.put_nowait`` (O(1), no I/O).

    A daemon thread drains the queue and does the actual ``json.dumps`` +
    ``write`` + ``flush``. Bounded queue (drops on full so a stuck disk
    cannot OOM the agent). ``atexit`` flushes on process exit so we don't
    silently lose the tail.
    """

    _SENTINEL: Any = object()

    def __init__(self, path: Path, max_queue: int = 10_000) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = path.open("a", encoding="utf-8")
        self._closed = False
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="agentm-obs-sink"
        )
        self._thread.start()
        atexit.register(self.close)

    def write(self, record: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._queue.task_done()
                return
            try:
                line = json.dumps(item, default=str, ensure_ascii=False)
                self._handle.write(line)
                self._handle.write("\n")
                self._handle.flush()
            except Exception:
                logger.exception("agentm observability sink write failed")
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._dropped:
            try:
                # Best-effort tail note about drops.
                self._queue.put_nowait(
                    {
                        "schema": "otel/span/v0",
                        "kind": "sink.drop_summary",
                        "name": "sink.drop_summary",
                        "attributes": {"dropped_record_count": self._dropped},
                        "status": {"code": "ERROR"},
                    }
                )
            except queue.Full:
                pass
        try:
            self._queue.put(self._SENTINEL, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        try:
            self._handle.close()
        except OSError:
            pass


_HANDLER_OWNER_ATTR = "_agentm_obs_owner"


class _Observer(EventBusObserver):
    def __init__(
        self,
        sink: _Sink,
        trace_id: str,
        include_handlers: bool,
        exclude_channels: frozenset[str],
    ) -> None:
        self._sink = sink
        self._trace_id = trace_id
        self._include_handlers = include_handlers
        self._exclude_channels = exclude_channels
        # Stack frame: (channel, span_id, start_ns, suppressed). When
        # suppressed=True, on_emit_end skips the sink write and
        # on_handler_done skips per-handler spans, but the frame still
        # cycles through push/pop so parent-span bookkeeping stays consistent
        # for non-suppressed siblings.
        self._stack: list[tuple[str, str, int, bool]] = []

    def on_emit_start(self, channel: str, event: Any) -> None:
        suppressed = channel in self._exclude_channels
        self._stack.append((channel, _new_id(), _now_ns(), suppressed))

    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
    ) -> None:
        if not self._include_handlers or not self._stack:
            return
        parent = self._stack[-1]
        if parent[3]:
            return
        end_ns = _now_ns()
        owner = getattr(handler, _HANDLER_OWNER_ATTR, None) or current_installing_extension()
        self._sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "handler.invoke",
                "trace_id": self._trace_id,
                "span_id": _new_id(),
                "parent_span_id": parent[1],
                "name": f"handler:{channel}",
                "start_time_unix_nano": end_ns - duration_ns,
                "end_time_unix_nano": end_ns,
                "attributes": {
                    "channel": channel,
                    "handler": _handler_label(handler),
                    "extension": owner,
                    "result": _serialize(result),
                    "duration_ns": duration_ns,
                },
                "status": {
                    "code": "ERROR" if error is not None else "OK",
                    "message": repr(error) if error is not None else None,
                    "traceback": _format_traceback(error),
                },
            }
        )

    def on_emit_end(self, channel: str, event: Any, results: list[Any]) -> None:
        if not self._stack:
            return
        ch, span_id, start_ns, suppressed = self._stack.pop()
        if suppressed:
            return
        end_ns = _now_ns()
        # Walk back through the stack for the nearest non-suppressed
        # ancestor — siblings emitted underneath a suppressed parent still
        # link to the closest real span instead of orphaning.
        parent_span_id: str | None = None
        for ancestor in reversed(self._stack):
            if not ancestor[3]:
                parent_span_id = ancestor[1]
                break
        self._sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "event.dispatch",
                "trace_id": self._trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": f"emit:{ch}",
                "start_time_unix_nano": start_ns,
                "end_time_unix_nano": end_ns,
                "attributes": {
                    "channel": ch,
                    "event": _serialize(event),
                    "handler_count": len(results),
                },
                "status": {"code": "OK"},
            }
        )


def _wrap_handler_for_diff(
    handler: Handler,
    sink: _Sink,
    trace_id: str,
    extension: str,
    channel: str,
) -> Handler:
    """Wrap a handler so we record a ``handler.mutated`` line whenever it
    mutates the event payload in place. Diff is computed even on the
    exception path. Skipped entirely for channels not in
    ``_MUTABLE_CHANNELS`` to bound serialization cost.
    """

    if channel not in _MUTABLE_CHANNELS:
        return handler

    def _record(diff: list[dict[str, Any]], error: BaseException | None) -> None:
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "handler.mutated",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": f"mutate:{channel}",
                "start_time_unix_nano": _now_ns(),
                "attributes": {
                    "channel": channel,
                    "handler": _handler_label(handler),
                    "extension": extension,
                    "mutations": diff[:100],
                    "raised": error is not None,
                },
                "status": {
                    "code": "ERROR" if error is not None else "OK",
                    "message": repr(error) if error is not None else None,
                },
            }
        )

    def _safe_diff(before: Any, event: Any, error: BaseException | None) -> None:
        try:
            diff = _deep_diff(before, _serialize(event))
        except Exception:
            return  # diff failure must never break the handler chain
        if diff:
            _record(diff, error)

    def _proxy(event: Any) -> Any:
        try:
            before = _serialize(event)
        except Exception:
            return handler(event)  # snapshot failed; skip diff entirely

        try:
            result = handler(event)
        except BaseException as exc:
            _safe_diff(before, event, exc)
            raise

        if inspect.isawaitable(result):

            async def _await() -> Any:
                try:
                    inner = await result
                except BaseException as exc:
                    _safe_diff(before, event, exc)
                    raise
                _safe_diff(before, event, None)
                return inner

            return _await()
        _safe_diff(before, event, None)
        return result

    _proxy.__module__ = getattr(handler, "__module__", "<unknown>")
    _proxy.__qualname__ = getattr(handler, "__qualname__", "<handler>")
    return _proxy


class _TurnAggregator:
    """Writes ``turn.summary`` spans on ``turn_end``.

    Tool-call counts and names come from the assistant message's content
    blocks (the source of truth) rather than from ``on_tool_call``
    subscriptions: ``AgentLoop`` emits ``turn_end`` BEFORE the per-call
    ``tool_call`` events for the same turn, so any subscription-based
    accumulator would always see an empty list at write time.

    ``tool_error_count`` still has to come from ``on_tool_result``
    observed during the *previous* turn's tool-execution phase, since
    error status isn't on the assistant message. We snapshot and reset
    that counter at each ``turn_start``.
    """

    def __init__(self, sink: _Sink, trace_id: str) -> None:
        self._sink = sink
        self._trace_id = trace_id
        self._turn_start_ns: int = 0
        self._previous_tool_errors: int = 0
        self._current_tool_errors: int = 0

    def on_turn_start(self, event: TurnStartEvent) -> None:
        self._turn_start_ns = _now_ns()
        # Tool errors observed between the previous turn's turn_end and
        # this turn_start belong to the previous turn but couldn't be
        # written into its summary (already flushed). We attribute them
        # to whichever turn opens next via ``_previous_tool_errors``.
        self._previous_tool_errors = self._current_tool_errors
        self._current_tool_errors = 0

    def on_tool_call(self, event: ToolCallEvent) -> None:
        # No-op: tool calls are now read from ``event.message.content``
        # in ``on_turn_end``. Kept subscribed so observability still
        # records the dispatch span via the generic event.dispatch path.
        del event

    def on_tool_result(self, event: ToolResultEvent) -> None:
        if getattr(event.result, "is_error", False):
            self._current_tool_errors += 1

    def on_turn_end(self, event: TurnEndEvent) -> None:
        end_ns = _now_ns()
        tool_calls = [
            block.name
            for block in event.message.content
            if isinstance(block, ToolCallBlock)
        ]
        self._sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "turn.summary",
                "trace_id": self._trace_id,
                "span_id": _new_id(),
                "name": f"turn:{event.turn_index}",
                "start_time_unix_nano": self._turn_start_ns,
                "end_time_unix_nano": end_ns,
                "attributes": {
                    "turn_index": event.turn_index,
                    "duration_ns": end_ns - self._turn_start_ns,
                    "tool_calls": tool_calls,
                    "tool_call_count": len(tool_calls),
                    # Errors from the prior turn's tool-execution phase,
                    # which run between turn_end[N-1] and turn_start[N].
                    "tool_error_count": self._previous_tool_errors,
                    "stop_reason": event.message.stop_reason,
                    "content_block_types": [
                        getattr(c, "type", type(c).__name__)
                        for c in event.message.content
                    ],
                },
                "status": {"code": "OK"},
            }
        )
        # Once written, the prior-turn carry no longer applies.
        self._previous_tool_errors = 0


def _make_simple_writer(sink: _Sink, trace_id: str, kind: str, name: str = ""):  # type: ignore[no-untyped-def]
    """Return a handler that turns any dataclass event into one OTel record."""

    def _write(event: Any) -> None:
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": kind,
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": name or kind,
                "start_time_unix_nano": _now_ns(),
                "attributes": _serialize(event),
                "status": {"code": "OK"},
            }
        )

    return _write


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    bus = api.events
    if getattr(bus, "_observer", None) is not None:
        return

    trace_id = api.session_id
    raw_path = config.get("path", ".agentm/observability/{session_id}.jsonl")
    materialized = str(raw_path).replace("{session_id}", api.session_id)
    file_path = Path(materialized)
    if not file_path.is_absolute():
        file_path = Path(api.cwd) / file_path

    include_handlers = bool(config.get("include_handler_records", True))
    include_diff = bool(config.get("include_mutation_diff", True))
    strict_sync = bool(config.get("strict_sync_handlers", False))
    max_queue = int(config.get("max_queue", 10_000))
    user_excludes = config.get("exclude_channels")
    exclude_channels = frozenset(_DEFAULT_EXCLUDE_CHANNELS) | (
        frozenset(str(ch) for ch in user_excludes) if user_excludes else frozenset()
    )

    if strict_sync:
        bus.set_strict_sync(True)

    sink = _Sink(file_path, max_queue=max_queue)
    observer = _Observer(sink, trace_id, include_handlers, exclude_channels)
    bus.set_observer(observer)

    # Wrap api.on for handler→extension attribution and (optional) mutation
    # diff. This is the only API-level monkey-patch — every other signal
    # flows through the bus as a normal event.
    original_on = api.on

    def _tracking_on(  # type: ignore[no-untyped-def]
        channel: str,
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
    ):
        ext = current_installing_extension()
        wrapped = (
            _wrap_handler_for_diff(handler, sink, trace_id, ext, channel)
            if include_diff
            else handler
        )
        try:
            setattr(wrapped, _HANDLER_OWNER_ATTR, ext)
        except (AttributeError, TypeError):
            pass
        return original_on(channel, wrapped, priority=priority)

    api.on = _tracking_on  # type: ignore[method-assign]

    # session.start: emitted directly because no other extension is loaded
    # yet to receive it via the bus, and we want one anchor record.
    sink.write(
        {
            "schema": "otel/span/v0",
            "kind": "session.start",
            "trace_id": trace_id,
            "span_id": _new_id(),
            "name": "session.start",
            "start_time_unix_nano": _now_ns(),
            "attributes": {"cwd": api.cwd, "log_path": str(file_path)},
            "status": {"code": "OK"},
        }
    )

    # All other signals: subscribe to bus channels.
    aggregator = _TurnAggregator(sink, trace_id)
    discovered_builtin = discover_builtin()
    builtin_by_module_path = {
        entry.module_path: entry for entry in discovered_builtin.values()
    }
    loaded_builtin_module_paths: set[str] = set()
    active_atom_hashes: dict[str, str] = {}
    current_fingerprint: dict[str, Any] | None = None

    def _compute_loaded_fingerprint(*, scenario: str | None = None) -> dict[str, Any]:
        """Build the active-set fingerprint for the builtins loaded in this session.

        R10 deviation: ``session.start`` fires during ``install()`` before the
        final loaded set exists, so the canonical fingerprint is emitted later
        as ``session.fingerprint`` at ``session_ready``.
        """

        nonlocal active_atom_hashes
        active_atom_hashes = {}
        writer = api.get_resource_writer()
        for module_path in sorted(loaded_builtin_module_paths):
            entry = builtin_by_module_path.get(module_path)
            if entry is None:
                continue
            source_path = inspect.getsourcefile(entry.module)
            if source_path is None:
                continue
            version_hash: str | None = None
            if isinstance(writer, GitBackedResourceWriter):
                version_hash = writer.current_version_for_path(source_path)
            if version_hash is None:
                source = inspect.getsource(entry.module)
                version_hash = api.catalog.compute_atom_hash(source)
            active_atom_hashes[entry.name] = version_hash
        return api.catalog.compute_active_set_fingerprint(
            loaded=active_atom_hashes,
            scenario=scenario,
            core_hash=None,
        )

    def _on_extension_install(event: ExtensionInstallEvent) -> None:
        if event.phase == "end" and event.module_path in builtin_by_module_path:
            loaded_builtin_module_paths.add(event.module_path)
        now = _now_ns()
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "extension.install",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": f"install:{event.module_path}",
                "start_time_unix_nano": now - event.duration_ns,
                "end_time_unix_nano": now,
                "attributes": {
                    "module_path": event.module_path,
                    "config": _serialize(event.config),
                    "phase": event.phase,
                    "duration_ns": event.duration_ns,
                },
                "status": {
                    "code": "ERROR" if event.error else "OK",
                    "message": event.error,
                },
            }
        )

    def _on_api_register(event: ApiRegisterEvent) -> None:
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "api.register",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": f"register:{event.kind}:{event.name}",
                "start_time_unix_nano": _now_ns(),
                "attributes": {
                    "kind": event.kind,
                    "name": event.name,
                    "extension": event.extension,
                    "payload": _serialize(event.payload),
                },
                "status": {"code": "OK"},
            }
        )

    def _on_api_send(event: ApiSendUserMessageEvent) -> None:
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "api.send_user_message",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": "send_user_message",
                "start_time_unix_nano": _now_ns(),
                "attributes": {
                    "extension": event.extension,
                    "content": _serialize(event.content),
                    "content_chars": (
                        len(event.content) if isinstance(event.content, str) else None
                    ),
                },
                "status": {"code": "OK"},
            }
        )

    def _on_llm_start(event: LlmRequestStartEvent) -> None:
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "llm.request.start",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": "llm.request",
                "start_time_unix_nano": _now_ns(),
                "attributes": _serialize(event),
                "status": {"code": "OK"},
            }
        )

    def _on_llm_end(event: LlmRequestEndEvent) -> None:
        now = _now_ns()
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "llm.request.end",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": "llm.request.end",
                "start_time_unix_nano": now - event.duration_ns,
                "end_time_unix_nano": now,
                "attributes": _serialize(event),
                "status": {
                    "code": "ERROR" if event.error else "OK",
                    "message": event.error,
                },
            }
        )

    def _on_ready(event: SessionReadyEvent) -> None:
        nonlocal current_fingerprint
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "session.ready",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": "session.ready",
                "start_time_unix_nano": _now_ns(),
                "attributes": {
                    "session_id": event.session_id,
                    "cwd": event.cwd,
                    "tool_names": list(event.tool_names),
                    "command_names": list(event.command_names),
                    "extension_module_paths": list(event.extension_module_paths),
                    "model": _serialize(event.model),
                },
                "status": {"code": "OK"},
            }
        )
        current_fingerprint = _compute_loaded_fingerprint()
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "session.fingerprint",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": "session.fingerprint",
                "start_time_unix_nano": _now_ns(),
                "attributes": {
                    **current_fingerprint,
                    "task_meta": {
                        "type": None,
                        "difficulty": None,
                        "external_id": None,
                    },
                },
                "status": {"code": "OK"},
            }
        )

    def _on_extension_reload(event: ExtensionReloadEvent) -> None:
        nonlocal current_fingerprint
        if event.name in active_atom_hashes or event.old_hash is None:
            active_atom_hashes[event.name] = event.new_hash
        fingerprint_after = api.catalog.compute_active_set_fingerprint(
            loaded=active_atom_hashes,
            scenario=(
                None if current_fingerprint is None else current_fingerprint.get("scenario")
            ),
            core_hash=None,
        )
        current_fingerprint = fingerprint_after
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "atom.reload",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": f"atom.reload:{event.name}",
                "start_time_unix_nano": _now_ns(),
                "attributes": {
                    "name": event.name,
                    "old_hash": event.old_hash,
                    "new_hash": event.new_hash,
                    "trigger": event.trigger,
                    "tier": event.tier,
                    "fingerprint_after": fingerprint_after,
                },
                "status": {
                    "code": "ERROR" if event.error else "OK",
                    "message": event.error,
                },
            }
        )

    def _on_shutdown(_: SessionShutdownEvent) -> None:
        sink.write(
            {
                "schema": "otel/span/v0",
                "kind": "session.end",
                "trace_id": trace_id,
                "span_id": _new_id(),
                "name": "session.end",
                "start_time_unix_nano": _now_ns(),
                "attributes": {},
                "status": {"code": "OK"},
            }
        )
        sink.close()

    # Use original_on so our own bookkeeping handlers don't get wrapped
    # for mutation diffing (they don't mutate events).
    original_on(ExtensionInstallEvent.CHANNEL, _on_extension_install)
    original_on(ApiRegisterEvent.CHANNEL, _on_api_register)
    original_on(ApiSendUserMessageEvent.CHANNEL, _on_api_send)
    original_on(LlmRequestStartEvent.CHANNEL, _on_llm_start)
    original_on(LlmRequestEndEvent.CHANNEL, _on_llm_end)
    original_on(ExtensionReloadEvent.CHANNEL, _on_extension_reload)
    original_on(SessionReadyEvent.CHANNEL, _on_ready)
    original_on(SessionShutdownEvent.CHANNEL, _on_shutdown)
    original_on(TurnStartEvent.CHANNEL, aggregator.on_turn_start)
    original_on(ToolCallEvent.CHANNEL, aggregator.on_tool_call)
    original_on(ToolResultEvent.CHANNEL, aggregator.on_tool_result)
    original_on(TurnEndEvent.CHANNEL, aggregator.on_turn_end)
