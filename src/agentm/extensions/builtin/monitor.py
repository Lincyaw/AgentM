"""Builtin ``monitor`` atom: agent-defined subscriptions and wakeups.

Agent-facing tool surface:

* :func:`schedule_wakeup(delay, note=None)` — one-shot timer that pushes a
  ``MonitorFire`` trigger after ``delay`` seconds.
* :func:`create_monitor(watch, note=None)` — subscribe to a bus channel; each
  fire pushes a ``MonitorFire`` trigger.
* :func:`create_monitor(condition, poll_interval=..., note=None)` — recurring
  condition poll that re-surfaces a free-text predicate.
* :func:`create_monitor(cron, note, recurring=True)` — persistent gateway cron
  monitor when the host injected ``gateway_scheduler``.
* :func:`list_monitors` / :func:`cancel_monitor`.

Migration note (v2-trajectory branch): the unified input path is
``api.push_trigger`` with a ``MonitorFire`` trigger, not ``api.post_inbox``;
the ``MonitorFire.payload`` is a string, so the structured payload is
JSON-encoded. ``BackgroundActivityEvent`` and the presenter activity surface do
not exist on this branch, so activity emission is a no-op. Trigger de-dup
(replace-undrained) is not yet available. Deviations are flagged with
``# TODO(migration):``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Literal

from loguru import logger

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    MonitorFire,
    ServiceNotFound,
    SessionShutdownEvent,
    TextContent,
    ToolResult,
    Unsubscribe,
)
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.lib import pydantic_to_tool_schema
from agentm.core.lib import to_jsonable
from agentm.extensions import ExtensionManifest

_PENDING: Literal["pending"] = "pending"
_FIRED: Literal["fired"] = "fired"
_ACTIVE: Literal["active"] = "active"
_CANCELLED: Literal["cancelled"] = "cancelled"
_Status = Literal["pending", "fired", "active", "cancelled"]

_KIND_WAKEUP: Literal["wakeup"] = "wakeup"
_KIND_CHANNEL: Literal["channel"] = "channel"
_KIND_CONDITION: Literal["condition"] = "condition"
_KIND_CRON: Literal["cron"] = "cron"
_Kind = Literal["wakeup", "channel", "condition", "cron"]
_GATEWAY_SCHEDULE_PREFIX = "schedule:"

# TODO(migration): shared lib constant on main; inlined until reintroduced.
_DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0

# TODO(migration): ``GATEWAY_SCHEDULER_SERVICE`` role constant is absent on this
# branch; the service key is inlined.
_GATEWAY_SCHEDULER_SERVICE = "gateway_scheduler"

_DEFAULT_CONDITION_POLL = 30.0
_DEFAULT_CONDITION_POLL_MIN = 5.0

# Channels the agent MUST NOT subscribe to via ``create_monitor`` — every fire
# would push a trigger, the loop-alive floor would keep the loop running, and
# the next turn's context drain would re-fire the same channels.
_KERNEL_CONTROL_CHANNELS: frozenset[str] = frozenset(
    {
        "context",
        "decide",
        "before_send",
        "before_run",
        "run_end",
        "turn_begin",
        "turn_committed",
        "stream_delta",
        "llm_request_start",
        "llm_request_end",
        "message_persisted",
        "message_appended",
        "entry_appended",
        "before_compact",
        "after_compact",
        "session_header_emitted",
        "session_ready",
        "session_shutdown",
        "child_session_start",
        "child_session_end",
        "child_session_extending",
        "extension_install",
        "extension_reload",
        "extension_unload",
        "api_register",
        "api_send_user_message",
        "background_activity",
    }
)


def _truncate_chars(text: str, max_tokens: int) -> str:
    """Approximate token truncation by characters (~4 chars/token).

    TODO(migration): main used ``agentm.core.lib.truncate_text_tokens`` for a
    model-accurate token count; that helper is absent on this branch.
    """
    max_chars = max(1, max_tokens * 4)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shutdown_grace_seconds: float = _DEFAULT_SHUTDOWN_GRACE_SECONDS
    event_summary_max_tokens: int = Field(gt=0)
    condition_poll_min_seconds: float = _DEFAULT_CONDITION_POLL_MIN


MANIFEST = ExtensionManifest(
    name="monitor",
    description=(
        "Agent-defined wakeups (schedule_wakeup) and bus-channel subscriptions "
        "(create_monitor) that push triggers into the session. Cron monitors use "
        "the optional gateway scheduler service for durable host-level "
        "wakeups; other monitors are per-session, in-memory state only."
    ),
    registers=(
        "tool:schedule_wakeup",
        "tool:create_monitor",
        "tool:list_monitors",
        "tool:cancel_monitor",
        "event:session_shutdown",
    ),
    config_schema=MonitorConfig,
    requires=(),
    priority=AtomInstallPriority.SERVICE,
)


@dataclass(slots=True, kw_only=True)
class _Monitor:
    """One live monitor's bookkeeping."""

    monitor_id: str
    kind: _Kind
    status: _Status
    watch: str | None = None
    note: str | None = None
    delay: float | None = None
    condition: str | None = None
    poll_interval: float | None = None
    task: asyncio.Task[Any] | None = None
    unsubscribe: Unsubscribe | None = None


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload)))],
        is_error=is_error,
        extras=payload,
    )


def _event_summary(
    event: Any,
    *,
    max_tokens: int,
    model_name: str | None,
) -> str:
    """Short, bounded repr of a bus event for the trigger payload."""
    del model_name
    return _truncate_chars(repr(event), max_tokens)


def _monitor_view(state: _Monitor) -> dict[str, Any]:
    view: dict[str, Any] = {
        "monitor_id": state.monitor_id,
        "kind": state.kind,
        "status": state.status,
    }
    if state.watch is not None:
        view["watch"] = state.watch
    if state.delay is not None:
        view["delay"] = state.delay
    if state.condition is not None:
        view["condition"] = state.condition
    if state.poll_interval is not None:
        view["poll_interval"] = state.poll_interval
    if state.note is not None:
        view["note"] = state.note
    return view


def _schedule_monitor_id(job_id: Any) -> str:
    return f"{_GATEWAY_SCHEDULE_PREFIX}{job_id}"


def _schedule_job_id(monitor_id: str) -> str | None:
    if not monitor_id.startswith(_GATEWAY_SCHEDULE_PREFIX):
        return None
    job_id = monitor_id.removeprefix(_GATEWAY_SCHEDULE_PREFIX)
    return job_id or None


def _schedule_view(job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job.get("id") or "")
    view: dict[str, Any] = {
        "monitor_id": _schedule_monitor_id(job_id),
        "kind": _KIND_CRON,
        "status": _ACTIVE if job.get("enabled", True) else _CANCELLED,
        "persistent": True,
    }
    if job_id:
        view["schedule_id"] = job_id
    if job.get("cron") is not None:
        view["cron"] = str(job["cron"])
    if job.get("prompt") is not None:
        view["note"] = str(job["prompt"])
    if job.get("next_fire_at") is not None:
        view["next_fire_at"] = job["next_fire_at"]
    if job.get("recurring") is not None:
        view["recurring"] = bool(job["recurring"])
    if job.get("fire_count") is not None:
        view["fire_count"] = int(job["fire_count"])
    if job.get("last_fire_at") is not None:
        view["last_fire_at"] = job["last_fire_at"]
    if job.get("last_error") is not None:
        view["last_error"] = str(job["last_error"])
    return view


def _monitor_label(state: _Monitor) -> str:
    if state.kind == _KIND_WAKEUP:
        return "wakeup"
    if state.kind == _KIND_CONDITION:
        return "monitor condition"
    if state.kind == _KIND_CRON:
        return "scheduled monitor"
    if state.watch:
        return f"monitor {state.watch}"
    return "monitor"


def _monitor_note(state: _Monitor) -> str | None:
    if state.note:
        return state.note
    if state.condition:
        return state.condition
    if state.watch:
        return state.watch
    if state.delay is not None:
        return f"{state.delay:g}s"
    return None


class _MonitorManager:
    """Per-session registry + asyncio/bus handles for live monitors."""

    def __init__(
        self,
        *,
        api: AtomAPI,
        shutdown_grace_seconds: float = _DEFAULT_SHUTDOWN_GRACE_SECONDS,
        event_summary_max_tokens: int,
        condition_poll_min_seconds: float = _DEFAULT_CONDITION_POLL_MIN,
    ) -> None:
        self._api = api
        self._monitors: dict[str, _Monitor] = {}
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._event_summary_max_tokens = event_summary_max_tokens
        self._condition_poll_min_seconds = condition_poll_min_seconds
        self._shutting_down = False

    def _emit_activity(
        self,
        state: _Monitor,
        *,
        note: str | None = None,
        terminal: bool = False,
    ) -> None:
        # TODO(migration): main emitted a ``BackgroundActivityEvent`` to the
        # presenter activity surface. Neither the event nor that surface exists
        # on this branch, so activity emission is a no-op.
        del state, note, terminal

    def _push_fire(self, monitor_id: str, payload: dict[str, Any]) -> None:
        # TODO(migration): main used ``api.post_inbox(..., dedup_key=...)`` whose
        # replace-undrained de-dup kept repeated fires from stacking. The v2
        # ``push_trigger`` path has no de-dup; ``MonitorFire.payload`` is a str,
        # so the structured payload is JSON-encoded.
        self._api.push_trigger(
            MonitorFire(monitor_id=monitor_id, payload=json.dumps(to_jsonable(payload))),
            origin="monitor",
        )

    def _gateway_scheduler(self) -> Any | None:
        try:
            return self._api.services.get(_GATEWAY_SCHEDULER_SERVICE)
        except ServiceNotFound:
            return None

    async def _create_cron_monitor(self, args: dict[str, Any], cron: Any) -> ToolResult:
        """Create a durable gateway schedule bound to this session."""

        if not isinstance(cron, str) or not cron.strip():
            return _tool_result(
                {"error": "cron must be a non-empty 5-field cron expression"},
                is_error=True,
            )
        if args.get("poll_interval") is not None:
            return _tool_result(
                {"error": "poll_interval only applies to condition monitors"},
                is_error=True,
            )
        note = args.get("note")
        if not isinstance(note, str) or not note.strip():
            return _tool_result(
                {"error": "cron monitors require a non-empty note"},
                is_error=True,
            )
        raw_recurring = args.get("recurring", True)
        if not isinstance(raw_recurring, bool):
            return _tool_result(
                {"error": "recurring must be a boolean"},
                is_error=True,
            )
        service = self._gateway_scheduler()
        create = getattr(service, "create", None) if service is not None else None
        if not callable(create):
            return _tool_result(
                {
                    "error": (
                        "persistent cron monitors require the gateway scheduler service"
                    )
                },
                is_error=True,
            )
        try:
            result = create(
                cron=cron.strip(),
                prompt=note.strip(),
                recurring=raw_recurring,
            )
            if isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001
            logger.debug("monitor: gateway schedule create failed: {}", exc)
            return _tool_result({"error": str(exc)}, is_error=True)
        if not isinstance(result, dict):
            return _tool_result(
                {"error": "gateway scheduler returned an invalid response"},
                is_error=True,
            )
        if result.get("error"):
            return _tool_result({"error": str(result["error"])}, is_error=True)
        job_id = str(result.get("id") or "")
        if not job_id:
            return _tool_result(
                {"error": "gateway scheduler returned a job without an id"},
                is_error=True,
            )
        state = _Monitor(
            monitor_id=_schedule_monitor_id(job_id),
            kind=_KIND_CRON,
            status=_ACTIVE,
            note=note.strip(),
        )
        self._emit_activity(state)
        view = _schedule_view(result)
        return _tool_result(view)

    async def _list_gateway_monitors(self) -> list[dict[str, Any]]:
        service = self._gateway_scheduler()
        list_jobs = getattr(service, "list", None) if service is not None else None
        if not callable(list_jobs):
            return []
        try:
            result = list_jobs()
            if isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001
            logger.debug("monitor: gateway schedule list failed: {}", exc)
            return []
        if not isinstance(result, list):
            return []
        views: list[dict[str, Any]] = []
        for row in result:
            if isinstance(row, dict):
                views.append(_schedule_view(row))
        return views

    async def _cancel_gateway_monitor(self, monitor_id: str) -> ToolResult | None:
        job_id = _schedule_job_id(monitor_id)
        if job_id is None:
            return None
        service = self._gateway_scheduler()
        delete = getattr(service, "delete", None) if service is not None else None
        if not callable(delete):
            return _tool_result(
                {
                    "error": (
                        "persistent cron monitors require the gateway scheduler service"
                    )
                },
                is_error=True,
            )
        try:
            result = delete(job_id)
            if isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001
            logger.debug("monitor: gateway schedule delete failed: {}", exc)
            return _tool_result({"error": str(exc)}, is_error=True)
        if bool(result):
            state = _Monitor(
                monitor_id=monitor_id,
                kind=_KIND_CRON,
                status=_CANCELLED,
            )
            self._emit_activity(state, terminal=True)
            return _tool_result(
                {
                    "monitor_id": monitor_id,
                    "kind": _KIND_CRON,
                    "status": _CANCELLED,
                    "persistent": True,
                }
            )
        return _tool_result(
            {"error": f"unknown monitor_id: {monitor_id}"},
            is_error=True,
        )

    # --- producers ---------------------------------------------------------

    async def schedule_wakeup(self, args: dict[str, Any]) -> ToolResult:
        """One-shot timer → trigger push at ``now + delay``."""

        raw_delay = args.get("delay")
        try:
            delay = float(raw_delay)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _tool_result(
                {"error": "delay must be a number (seconds)"}, is_error=True
            )
        if delay < 0:
            return _tool_result({"error": "delay must be non-negative"}, is_error=True)
        if self._shutting_down:
            return _tool_result(
                {"error": "session is shutting down; wakeup refused"},
                is_error=True,
            )
        note = args.get("note")
        note_str: str | None = str(note) if note is not None else None

        monitor_id = uuid.uuid4().hex
        state = _Monitor(
            monitor_id=monitor_id,
            kind=_KIND_WAKEUP,
            status=_PENDING,
            note=note_str,
            delay=delay,
        )
        state.task = asyncio.create_task(self._wakeup_runner(state))
        self._monitors[monitor_id] = state
        self._emit_activity(state)
        return _tool_result(
            {
                "monitor_id": monitor_id,
                "kind": _KIND_WAKEUP,
                "delay": delay,
                "status": _PENDING,
            }
        )

    async def _wakeup_runner(self, state: _Monitor) -> None:
        """Sleep ``state.delay`` seconds, then push a single trigger."""

        delay = state.delay or 0.0
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if state.status == _CANCELLED:
            return
        state.status = _FIRED
        payload = {
            "kind": _KIND_WAKEUP,
            "monitor_id": state.monitor_id,
            "note": state.note,
            "delay": delay,
        }
        self._emit_activity(state, terminal=True)
        try:
            self._push_fire(state.monitor_id, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("monitor: wakeup push failed for {}: {}", state.monitor_id, exc)

    async def _create_condition_monitor(
        self, args: dict[str, Any], condition: Any
    ) -> ToolResult:
        """Start a recurring condition-poll monitor."""

        if not isinstance(condition, str) or not condition.strip():
            return _tool_result(
                {"error": "condition must be a non-empty free-text predicate (str)"},
                is_error=True,
            )
        raw_interval = args.get("poll_interval", _DEFAULT_CONDITION_POLL)
        try:
            poll_interval = float(raw_interval)
        except (TypeError, ValueError):
            return _tool_result(
                {"error": "poll_interval must be a number (seconds)"},
                is_error=True,
            )
        if poll_interval <= 0:
            return _tool_result(
                {"error": "poll_interval must be positive"}, is_error=True
            )
        poll_interval = max(poll_interval, self._condition_poll_min_seconds)
        if self._shutting_down:
            return _tool_result(
                {"error": "session is shutting down; monitor refused"},
                is_error=True,
            )
        note = args.get("note")
        note_str: str | None = str(note) if note is not None else None

        monitor_id = uuid.uuid4().hex
        state = _Monitor(
            monitor_id=monitor_id,
            kind=_KIND_CONDITION,
            status=_ACTIVE,
            note=note_str,
            condition=condition,
            poll_interval=poll_interval,
        )
        state.task = asyncio.create_task(self._condition_runner(state))
        self._monitors[monitor_id] = state
        self._emit_activity(state)
        return _tool_result(
            {
                "monitor_id": monitor_id,
                "kind": _KIND_CONDITION,
                "condition": condition,
                "poll_interval": poll_interval,
                "status": _ACTIVE,
            }
        )

    async def _condition_runner(self, state: _Monitor) -> None:
        """Re-push the condition every ``poll_interval`` seconds until cancelled."""

        interval = state.poll_interval or _DEFAULT_CONDITION_POLL
        try:
            while True:
                await asyncio.sleep(interval)
                if state.status == _CANCELLED:
                    return
                payload = {
                    "kind": _KIND_CONDITION,
                    "monitor_id": state.monitor_id,
                    "condition": state.condition,
                    "note": state.note,
                    "poll_interval": interval,
                }
                self._emit_activity(state)
                self._push_fire(state.monitor_id, payload)
        except asyncio.CancelledError:
            return

    async def create_monitor(self, args: dict[str, Any]) -> ToolResult:
        """Subscribe to a bus channel, start a condition poll, OR create cron."""

        watch = args.get("watch")
        condition = args.get("condition")
        cron = args.get("cron")
        selected = sum(
            value is not None
            for value in (
                watch,
                condition,
                cron,
            )
        )
        if selected != 1:
            return _tool_result(
                {
                    "error": (
                        "create_monitor takes exactly one of 'watch' (bus "
                        "channel), 'condition' (recurring poll), or 'cron' "
                        "(persistent gateway schedule)"
                    )
                },
                is_error=True,
            )
        if condition is not None:
            return await self._create_condition_monitor(args, condition)
        if cron is not None:
            return await self._create_cron_monitor(args, cron)
        if not isinstance(watch, str) or not watch:
            return _tool_result(
                {"error": "watch must be a non-empty bus channel name (str)"},
                is_error=True,
            )
        if watch in _KERNEL_CONTROL_CHANNELS:
            return _tool_result(
                {
                    "error": (
                        f"channel {watch!r} is a kernel-internal control "
                        "channel and cannot be monitored (subscribing would "
                        "create a trigger / loop spin)"
                    )
                },
                is_error=True,
            )
        if self._shutting_down:
            return _tool_result(
                {"error": "session is shutting down; monitor refused"},
                is_error=True,
            )
        note = args.get("note")
        note_str: str | None = str(note) if note is not None else None

        monitor_id = uuid.uuid4().hex
        state = _Monitor(
            monitor_id=monitor_id,
            kind=_KIND_CHANNEL,
            status=_ACTIVE,
            watch=watch,
            note=note_str,
        )

        def _handler(event: Any) -> None:
            if state.status == _CANCELLED:
                return
            event_summary = _event_summary(
                event,
                max_tokens=self._event_summary_max_tokens,
                model_name=(
                    self._api.model.id if self._api.model is not None else None
                ),
            )
            payload = {
                "kind": _KIND_CHANNEL,
                "monitor_id": state.monitor_id,
                "channel": watch,
                "note": state.note,
                "event_summary": event_summary,
            }
            try:
                self._push_fire(state.monitor_id, payload)
                self._emit_activity(state, note=event_summary)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "monitor: channel push failed for {}: {}", state.monitor_id, exc
                )

        state.unsubscribe = self._api.on(watch, _handler)
        self._monitors[monitor_id] = state
        self._emit_activity(state)
        return _tool_result(
            {
                "monitor_id": monitor_id,
                "kind": _KIND_CHANNEL,
                "watch": watch,
                "status": _ACTIVE,
            }
        )

    async def list_monitors(self, _args: dict[str, Any]) -> ToolResult:
        monitors = [_monitor_view(state) for state in self._monitors.values()]
        monitors.extend(await self._list_gateway_monitors())
        return _tool_result({"monitors": monitors})

    async def cancel_monitor(self, args: dict[str, Any]) -> ToolResult:
        """Cancel one monitor — and ONLY that monitor."""

        monitor_id = str(args.get("monitor_id", ""))
        state = self._monitors.get(monitor_id)
        if state is None:
            gateway_result = await self._cancel_gateway_monitor(monitor_id)
            if gateway_result is not None:
                return gateway_result
            return _tool_result(
                {"error": f"unknown monitor_id: {monitor_id}"}, is_error=True
            )
        if state.status in (_CANCELLED, _FIRED):
            return _tool_result({"monitor_id": monitor_id, "status": state.status})
        state.status = _CANCELLED
        self._emit_activity(state, terminal=True)
        if state.kind in (_KIND_WAKEUP, _KIND_CONDITION) and state.task is not None:
            if not state.task.done():
                state.task.cancel()
        elif state.kind == _KIND_CHANNEL and state.unsubscribe is not None:
            try:
                state.unsubscribe()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "monitor: unsubscribe failed cancelling {}: {}", monitor_id, exc
                )
        return _tool_result({"monitor_id": monitor_id, "status": _CANCELLED})

    # --- lifecycle ---------------------------------------------------------

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        """Cancel every task (wakeup + condition) and clear every channel sub."""

        self._shutting_down = True
        wakeup_tasks: list[asyncio.Task[Any]] = []
        for state in list(self._monitors.values()):
            if state.status in (_CANCELLED, _FIRED):
                continue
            state.status = _CANCELLED
            if state.kind in (_KIND_WAKEUP, _KIND_CONDITION) and state.task is not None:
                if not state.task.done():
                    state.task.cancel()
                    wakeup_tasks.append(state.task)
            elif state.kind == _KIND_CHANNEL and state.unsubscribe is not None:
                try:
                    state.unsubscribe()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("monitor: unsubscribe failed during shutdown: {}", exc)
        if not wakeup_tasks:
            return
        await asyncio.wait(wakeup_tasks, timeout=self._shutdown_grace_seconds)
        await asyncio.gather(*wakeup_tasks, return_exceptions=True)


# Tool schemas (Pydantic -> JSON Schema via pydantic_to_tool_schema)
# ---------------------------------------------------------------------------


class _ScheduleWakeupParams(BaseModel):
    delay: float = Field(ge=0, description="Seconds to wait before firing.")
    note: str | None = Field(
        default=None,
        description="Free-form note delivered with the wake.",
    )


class _CreateMonitorParams(BaseModel):
    watch: str | None = Field(
        default=None,
        description=(
            "Bus channel name to subscribe to (e.g. 'tool_call', "
            "'tool_result'). Kernel control/lifecycle channels (before_run, "
            "run_end, context, turn_*, ...) are rejected."
        ),
    )
    condition: str | None = Field(
        default=None,
        description=(
            "Free-text predicate to re-surface every poll_interval seconds "
            "until the agent decides it is satisfied and cancels the monitor."
        ),
    )
    cron: str | None = Field(
        default=None,
        description=(
            "Standard 5-field cron expression for a persistent gateway "
            "schedule. Requires note; mutually exclusive with watch and "
            "condition."
        ),
    )
    poll_interval: float | None = Field(
        default=None,
        description=(
            "Seconds between condition monitor fires. Values below the "
            "configured minimum (default 5s) are clamped up to it."
        ),
    )
    recurring: bool = Field(
        default=True,
        description=(
            "For cron monitors, keep firing on every matching cron tick. If "
            "false, fire once at the next matching tick."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "Free-form note delivered with each fire. Required for cron monitors."
        ),
    )


class _ListMonitorsParams(BaseModel):
    pass


class _CancelMonitorParams(BaseModel):
    monitor_id: str


class _MonitorRuntime:
    def __init__(self, api: AtomAPI, config: MonitorConfig) -> None:
        self._api = api
        self._manager = _MonitorManager(
            api=api,
            shutdown_grace_seconds=config.shutdown_grace_seconds,
            event_summary_max_tokens=config.event_summary_max_tokens,
            condition_poll_min_seconds=config.condition_poll_min_seconds,
        )

    def install(self) -> None:
        self._api.on(SessionShutdownEvent.CHANNEL, self._manager.on_session_shutdown)
        self._register_tools()

    def _register_tools(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="schedule_wakeup",
                description=(
                    "Schedule a one-shot wakeup that pushes a trigger into the "
                    "session after `delay` seconds. Returns a monitor_id."
                ),
                parameters=pydantic_to_tool_schema(_ScheduleWakeupParams),
                fn=self._manager.schedule_wakeup,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="create_monitor",
                description=(
                    "Create a monitor that delivers events into your session. "
                    "Exactly one mode — watch (bus channel subscription), "
                    "condition + poll_interval (recurring poll), or cron + note "
                    "(persistent gateway-backed schedule)."
                ),
                parameters=pydantic_to_tool_schema(_CreateMonitorParams),
                fn=self._manager.create_monitor,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="list_monitors",
                description=(
                    "List every live monitor with its id, kind, status and "
                    "mode-specific fields (watch channel, condition + "
                    "poll_interval, or cron schedule details). Durable "
                    "gateway cron schedules are listed alongside in-memory "
                    "monitors."
                ),
                parameters=pydantic_to_tool_schema(_ListMonitorsParams),
                fn=self._manager.list_monitors,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="cancel_monitor",
                description=(
                    "Cancel one monitor by id. Cancels the wakeup task, "
                    "condition poller, or channel subscription; for a "
                    "persistent cron monitor (schedule:<id>) this DELETES the "
                    "durable gateway schedule. Never touches any shared "
                    "session signal. Idempotent."
                ),
                parameters=pydantic_to_tool_schema(_CancelMonitorParams),
                fn=self._manager.cancel_monitor,
            )
        )


def install(api: AtomAPI, config: MonitorConfig) -> None:
    _MonitorRuntime(api, config).install()


__all__ = (
    "MANIFEST",
    "MonitorConfig",
    "install",
)
