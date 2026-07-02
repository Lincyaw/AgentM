"""Builtin ``monitor`` atom: agent-defined subscriptions and wakeups.

Design: ``.claude/designs/session-inbox.md`` (the ``monitor`` section) — the
third producer atom on the inbox spine after ``background_exec`` (whose
lifecycle / cancel / stale wiring this atom mirrors so we don't relearn
step-3's hard lessons).

What it exposes (all four are tools the agent calls):

* :func:`schedule_wakeup(delay, note=None)` — one-shot timer. ``await
  asyncio.sleep(delay)`` then post a ``source="monitor"`` inbox item with
  ``payload.kind="wakeup"``. Returns the monitor's id and a pending status.
* :func:`create_monitor(watch, note=None)` — subscribe to a bus channel; on
  each fire post a ``source="monitor"`` item with ``payload.kind="channel"``
  under a stable ``dedup_key`` so repeated fires REPLACE the prior undrained
  item rather than stacking (same discipline as the step-3 ticker).
* :func:`create_monitor(condition, poll_interval=…, note=None)` — recurring
  condition poll (#178). Every ``poll_interval`` seconds it posts a
  ``source="monitor"`` item with ``payload.kind="condition"`` carrying the
  free-text ``condition`` so the AGENT re-evaluates it at the turn boundary
  and cancels the monitor once satisfied. The atom is not an evaluator (it has
  no LLM); it is the metronome that keeps re-surfacing the predicate. Same
  ``dedup_key`` replace + per-monitor cancel + stale-guard discipline as the
  channel form.
* :func:`list_monitors` — every live monitor's id / kind / watch / status.
* :func:`cancel_monitor(monitor_id)` — cancels the wakeup task or unsubscribes
  the channel handler. Idempotent (unknown id → tool-error result).

**Scope.** ``create_monitor`` supports two mutually-exclusive forms: a
bus-channel subscription (``watch``) and a recurring condition poll
(``condition``). Passing both, or neither, returns a tool-error ``ToolResult``
so the agent sees the misuse explicitly (no ``NotImplementedError`` reaches the
tool surface).

Architecture: a single :class:`_MonitorManager` owns the per-session in-memory
state (the registry of live monitors, the asyncio tasks, the channel
unsubscribe callables) so ``install`` stays a thin wire-up, mirroring
``background_exec``.

§11: single file; module-level ``MANIFEST`` + ``install(api, config)``; no
atom→atom imports; ``core.lib`` / ``core.abi`` only; no ``core.runtime.*`` /
``core._internal``. State is per-session and in-memory only (no persistence —
step-1 decision #5: a restart regenerates monitors via re-prompting).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from agentm.core.abi import (
    ExtensionAPI,
    ExtensionStaleError,
    FunctionTool,
    SessionShutdownEvent,
    TextContent,
    ToolResult,
    Unsubscribe,
)
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.lib import pydantic_to_tool_schema
from agentm.core.lib import (
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    to_jsonable,
    truncate_text_tokens,
)
from agentm.extensions import ExtensionManifest

_PENDING: Literal["pending"] = "pending"
_FIRED: Literal["fired"] = "fired"
_ACTIVE: Literal["active"] = "active"
_CANCELLED: Literal["cancelled"] = "cancelled"
_Status = Literal["pending", "fired", "active", "cancelled"]

_KIND_WAKEUP: Literal["wakeup"] = "wakeup"
_KIND_CHANNEL: Literal["channel"] = "channel"
_KIND_CONDITION: Literal["condition"] = "condition"
_Kind = Literal["wakeup", "channel", "condition"]

# Default seconds between condition-poll fires (#178). A scenario can widen the
# floor via ``condition_poll_min_seconds``; the per-call ``poll_interval`` is
# clamped to it so the agent cannot wedge the session into a tight re-poll spin.
_DEFAULT_CONDITION_POLL = 30.0
_DEFAULT_CONDITION_POLL_MIN = 5.0

# Channels the agent MUST NOT subscribe to via ``create_monitor`` — every fire
# would post to the session inbox, the inbox-non-empty floor would keep the
# loop alive, and the next turn's ``context`` drain would re-fire the same
# channels, producing an unbreakable spin (the canonical example is
# ``watch="context"``: every monitor fire triggers another context event).
#
# Hand-curated rather than derived from ``Event.__subclasses__()``: introspection
# would also include legitimately-watchable kernel events (``tool_call``,
# ``tool_result``, ``plan_submitted``, ``cost_budget_exceeded``, ...) that the
# brief explicitly leaves open to the agent. Grep ``CHANNEL: ClassVar[Literal[``
# in ``core.abi.events`` when reviewing — every channel listed here exists
# there; channels NOT listed here are open to monitor.
_KERNEL_CONTROL_CHANNELS: frozenset[str] = frozenset(
    {
        # Per-turn control channels — fire from inside the kernel loop and
        # any monitor on them creates an inbox/loop spin.
        "context",
        "decide_turn_action",
        "before_send_to_llm",
        "before_agent_start",
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "stream_delta",
        "llm_request_start",
        "llm_request_end",
        # Persistence + compaction — fire as part of the kernel's message
        # pipeline; treating them as agent-observable would mix substrate
        # plumbing into the agent's reasoning surface.
        "message_persisted",
        "message_appended",
        "entry_appended",
        "before_compact",
        "after_compact",
        # Session lifecycle — substrate-owned; the agent has its own way to
        # observe these (e.g. the dispatch tools) without a monitor.
        "session_header_emitted",
        "session_ready",
        "session_shutdown",
        "child_session_start",
        "child_session_end",
        "child_session_extending",
        # Extension / API meta-events — observability concerns, not agent
        # reasoning targets. ``api_register`` fires synchronously during
        # ``install`` and a monitor would race the install path itself.
        "extension_install",
        "extension_reload",
        "extension_unload",
        "api_register",
        "api_send_user_message",
    }
)

class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS
    event_summary_max_tokens: int = Field(gt=0)
    condition_poll_min_seconds: float = _DEFAULT_CONDITION_POLL_MIN

MANIFEST = ExtensionManifest(
    name="monitor",
    description=(
        "Agent-defined wakeups (schedule_wakeup) and bus-channel subscriptions "
        "(create_monitor) that push to the session inbox. Per-session, "
        "in-memory state only."
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
)

@dataclass(slots=True, kw_only=True)
class _Monitor:
    """One live monitor's bookkeeping.

    Each monitor owns its OWN handle — an ``asyncio.Task`` for a wakeup, an
    ``Unsubscribe`` callable for a channel subscription — so ``cancel_monitor``
    can stop just this one without ever touching a shared session signal
    (step-3 Major 2). The shared kernel/session abort signal is intentionally
    NOT referenced anywhere in this dataclass for that reason.
    """

    monitor_id: str
    kind: _Kind
    status: _Status
    watch: str | None = None  # channel name (kind="channel"); None for wakeup
    note: str | None = None
    delay: float | None = None  # wakeup only
    condition: str | None = None  # free-text predicate (kind="condition")
    poll_interval: float | None = None  # condition poll period (kind="condition")
    task: asyncio.Task[Any] | None = None  # wakeup sleep / condition poll task
    unsubscribe: Unsubscribe | None = None  # channel's bus unsubscribe

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
    """Short, bounded repr of a bus event for the inbox payload.

    Kept tiny on purpose — the inbox item is a poke ("channel X fired"), not a
    transport for the event itself. The agent inspects the live state through
    other tools if it needs detail. ``max_tokens`` is set per-manager from the
    atom config (``event_summary_max_tokens``).
    """

    text = repr(event)
    truncated = truncate_text_tokens(text, max_tokens, model=model_name)
    if not truncated.was_truncated:
        return text
    return truncated.text + "..."

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

class _MonitorManager:
    """Per-session registry + asyncio/bus handles for live monitors.

    Holds the in-memory dict of monitors plus the per-monitor task /
    unsubscribe handles. Mirrors the ``_BgManager`` shape so the install entry
    point stays a thin wire-up (and so the lifecycle / cancel / stale-guard
    discipline is structurally identical, not just intentionally similar).
    """

    def __init__(
        self,
        *,
        api: ExtensionAPI,
        shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
        event_summary_max_tokens: int,
        condition_poll_min_seconds: float = _DEFAULT_CONDITION_POLL_MIN,
    ) -> None:
        self._api = api
        self._monitors: dict[str, _Monitor] = {}
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._event_summary_max_tokens = event_summary_max_tokens
        self._condition_poll_min_seconds = condition_poll_min_seconds
        # Set in on_session_shutdown so any in-flight wakeup that wakes mid-
        # shutdown sees it and exits without trying to register / push.
        self._shutting_down = False

    # --- producers ---------------------------------------------------------

    async def schedule_wakeup(self, args: dict[str, Any]) -> ToolResult:
        """One-shot timer → inbox push at ``now + delay``."""

        raw_delay = args.get("delay")
        try:
            delay = float(raw_delay)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _tool_result(
                {"error": "delay must be a number (seconds)"}, is_error=True
            )
        if delay < 0:
            return _tool_result(
                {"error": "delay must be non-negative"}, is_error=True
            )
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
        return _tool_result(
            {
                "monitor_id": monitor_id,
                "kind": _KIND_WAKEUP,
                "delay": delay,
                "status": _PENDING,
            }
        )

    async def _wakeup_runner(self, state: _Monitor) -> None:
        """Sleep ``state.delay`` seconds, then post a single inbox item.

        Posts even if ``self._shutting_down`` is set: by then the bus may already
        be cleared, so ``post_inbox`` is wrapped in :class:`ExtensionStaleError`
        (step-3 Major 3) — a stale api stops gracefully without crashing this
        detached task.
        """

        delay = state.delay or 0.0
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            # cancel_monitor or shutdown asked us to stop before firing.
            return
        if state.status == _CANCELLED:
            # Won the race with cancel_monitor: cancel set the status but the
            # sleep had already completed — honour the cancel and do not fire.
            return
        state.status = _FIRED
        payload = {
            "kind": _KIND_WAKEUP,
            "monitor_id": state.monitor_id,
            "note": state.note,
            "delay": delay,
        }
        try:
            self._api.post_inbox(
                source="monitor",
                payload=payload,
                dedup_key=f"monitor-wake-{state.monitor_id}",
            )
        except ExtensionStaleError:
            # Atom reloaded while this detached wakeup was sleeping: the inbox
            # we hold is stale. Stop gracefully (no unretrieved exception).
            return

    async def _create_condition_monitor(
        self, args: dict[str, Any], condition: Any
    ) -> ToolResult:
        """Start a recurring condition-poll monitor (#178).

        The atom cannot evaluate free-text; the AGENT does, at the turn
        boundary. This task is the metronome: every ``poll_interval`` seconds it
        re-posts the predicate so the agent re-checks and cancels when satisfied.
        Same per-monitor cancel + stale-guard discipline as the wakeup/channel
        forms — it spawns its OWN task and NEVER touches a shared kernel signal.
        """

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
        # Clamp up to the configured floor so the agent can't wedge a spin.
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
        """Re-post the condition every ``poll_interval`` seconds until cancelled.

        Unlike the one-shot wakeup, a condition monitor stays ``_ACTIVE`` across
        fires — only ``cancel_monitor`` / shutdown stops it (the agent cancels
        once the predicate holds). Each fire reuses one ``dedup_key`` so a
        stuck-in-a-long-turn agent never finds a pile of stale poll lines. The
        ``post_inbox`` is wrapped in :class:`ExtensionStaleError` so a mid-flight
        atom reload stops this detached task gracefully.
        """

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
                self._api.post_inbox(
                    source="monitor",
                    payload=payload,
                    dedup_key=f"monitor-cond-{state.monitor_id}",
                )
        except asyncio.CancelledError:
            # cancel_monitor or shutdown asked us to stop.
            return
        except ExtensionStaleError:
            # Atom reloaded while this detached poll was sleeping: the inbox we
            # hold is stale. Stop gracefully (no unretrieved exception).
            return

    async def create_monitor(self, args: dict[str, Any]) -> ToolResult:
        """Subscribe to a bus channel, OR start a recurring condition poll.

        Two mutually-exclusive forms:

        * ``watch`` — subscribe to a bus channel; each fire posts to the inbox.
        * ``condition`` — recurring poll (#178): every ``poll_interval`` seconds
          re-post the free-text predicate so the AGENT re-evaluates it and
          cancels the monitor once satisfied. The atom is the metronome, not the
          evaluator.

        Passing both or neither is a misuse and returns a tool-error.
        """

        watch = args.get("watch")
        condition = args.get("condition")
        if watch is not None and condition is not None:
            return _tool_result(
                {
                    "error": (
                        "create_monitor takes EITHER 'watch' (bus channel) OR "
                        "'condition' (recurring poll), not both"
                    )
                },
                is_error=True,
            )
        if condition is not None:
            return await self._create_condition_monitor(args, condition)
        if watch is None:
            return _tool_result(
                {
                    "error": (
                        "create_monitor needs a 'watch' (bus channel name) or a "
                        "'condition' (free-text predicate to poll)"
                    )
                },
                is_error=True,
            )
        if not isinstance(watch, str) or not watch:
            return _tool_result(
                {"error": "watch must be a non-empty bus channel name (str)"},
                is_error=True,
            )
        if watch in _KERNEL_CONTROL_CHANNELS:
            # Subscribing to a kernel control channel via create_monitor would
            # create an inescapable loop: every fire posts to the inbox, the
            # inbox-non-empty floor keeps the loop alive, the next turn fires
            # the channel again. Refuse explicitly so the agent gets a tool-
            # error rather than spinning the session. See
            # ``_KERNEL_CONTROL_CHANNELS`` for the rationale per channel.
            return _tool_result(
                {
                    "error": (
                        f"channel {watch!r} is a kernel-internal control "
                        "channel and cannot be monitored (subscribing would "
                        "create an inbox / loop spin)"
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
            # Channel handlers from the kernel bus are called even after the
            # atom reloads — the bus has no reload awareness. Same guard as the
            # detached wakeup: a stale api raises ExtensionStaleError; swallow
            # so we do not propagate into the kernel dispatch path.
            if state.status == _CANCELLED:
                return
            payload = {
                "kind": _KIND_CHANNEL,
                "monitor_id": state.monitor_id,
                "channel": watch,
                "note": state.note,
                "event_summary": _event_summary(
                    event,
                    max_tokens=self._event_summary_max_tokens,
                    model_name=(
                        self._api.model.id if self._api.model is not None else None
                    ),
                ),
            }
            try:
                self._api.post_inbox(
                    source="monitor",
                    payload=payload,
                    dedup_key=f"monitor-chan-{state.monitor_id}",
                )
            except ExtensionStaleError:
                return

        state.unsubscribe = self._api.on(watch, _handler)
        self._monitors[monitor_id] = state
        return _tool_result(
            {
                "monitor_id": monitor_id,
                "kind": _KIND_CHANNEL,
                "watch": watch,
                "status": _ACTIVE,
            }
        )

    async def list_monitors(self, _args: dict[str, Any]) -> ToolResult:
        return _tool_result(
            {"monitors": [_monitor_view(state) for state in self._monitors.values()]}
        )

    async def cancel_monitor(self, args: dict[str, Any]) -> ToolResult:
        """Cancel one monitor — and ONLY that monitor.

        Wakeups: cancel the per-monitor ``asyncio.Task``. Channels: call the
        per-subscription ``Unsubscribe``. Neither path touches the shared
        session/kernel signal — that conflation was step-3 Major 2 and we do
        not repeat it here.
        """

        monitor_id = str(args.get("monitor_id", ""))
        state = self._monitors.get(monitor_id)
        if state is None:
            return _tool_result(
                {"error": f"unknown monitor_id: {monitor_id}"}, is_error=True
            )
        if state.status in (_CANCELLED, _FIRED):
            # Both ``_CANCELLED`` and ``_FIRED`` are terminal; cancelling
            # after a successful fire must NOT overwrite ``_FIRED`` (the
            # bookkeeping would then lie to the agent). Return the actual
            # current status — idempotent in both directions.
            return _tool_result(
                {"monitor_id": monitor_id, "status": state.status}
            )
        state.status = _CANCELLED
        # Wakeup + condition monitors own an asyncio.Task; channels own an
        # unsubscribe callable. Either way we stop ONLY this monitor's handle —
        # never a shared session/kernel signal (step-3 Major 2).
        if state.kind in (_KIND_WAKEUP, _KIND_CONDITION) and state.task is not None:
            if not state.task.done():
                state.task.cancel()
        elif state.kind == _KIND_CHANNEL and state.unsubscribe is not None:
            try:
                state.unsubscribe()
            except Exception as exc:  # noqa: BLE001
                # An unsubscribe failure must not crash the tool — the monitor
                # is now marked cancelled and its handler is the no-op branch.
                logger.debug("monitor: unsubscribe failed cancelling {}: {}", monitor_id, exc)
        return _tool_result({"monitor_id": monitor_id, "status": _CANCELLED})

    # --- lifecycle ---------------------------------------------------------

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        """Cancel every task (wakeup + condition) and clear every channel sub.

        Mirrors ``background_exec.on_session_shutdown``: set the shutdown flag
        first (so any in-flight ``schedule_wakeup`` / ``create_monitor`` racing
        in refuses cleanly), tear down handles, give pending wakeup tasks a
        bounded grace, then gather with ``return_exceptions=True`` so nothing
        leaks past the bus clearing.
        """

        self._shutting_down = True
        wakeup_tasks: list[asyncio.Task[Any]] = []
        for state in list(self._monitors.values()):
            if state.status in (_CANCELLED, _FIRED):
                # Already terminal — ``_FIRED`` stays ``_FIRED`` across
                # shutdown; do not rewrite the bookkeeping.
                continue
            state.status = _CANCELLED
            # Wakeup AND condition monitors own an asyncio.Task; cancel both so
            # no detached poll survives the session. Channels own an unsubscribe.
            if (
                state.kind in (_KIND_WAKEUP, _KIND_CONDITION)
                and state.task is not None
            ):
                if not state.task.done():
                    state.task.cancel()
                    wakeup_tasks.append(state.task)
            elif state.kind == _KIND_CHANNEL and state.unsubscribe is not None:
                try:
                    state.unsubscribe()
                except Exception as exc:  # noqa: BLE001
                    # Best-effort teardown during shutdown — keep draining.
                    logger.debug("monitor: unsubscribe failed during shutdown: {}", exc)
        if not wakeup_tasks:
            return
        await asyncio.wait(wakeup_tasks, timeout=self._shutdown_grace_seconds)
        # Gather so any CancelledError / late exception is retrieved cleanly
        # (no "Task exception was never retrieved" past shutdown).
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
            "Bus channel name to subscribe to "
            "(e.g. 'tool_call', 'agent_end')."
        ),
    )
    note: str | None = Field(
        default=None,
        description="Free-form note delivered with each fire.",
    )

class _ListMonitorsParams(BaseModel):
    pass

class _CancelMonitorParams(BaseModel):
    monitor_id: str

def install(api: ExtensionAPI, config: MonitorConfig) -> None:
    manager = _MonitorManager(
        api=api,
        shutdown_grace_seconds=config.shutdown_grace_seconds,
        event_summary_max_tokens=config.event_summary_max_tokens,
        condition_poll_min_seconds=config.condition_poll_min_seconds,
    )
    api.on(SessionShutdownEvent.CHANNEL, manager.on_session_shutdown)
    api.register_tool(
        FunctionTool(
            name="schedule_wakeup",
            description=(
                "Schedule a one-shot wakeup that posts to the session inbox "
                "after `delay` seconds. Returns a monitor_id."
            ),
            parameters=pydantic_to_tool_schema(_ScheduleWakeupParams),
            fn=manager.schedule_wakeup,
        )
    )
    api.register_tool(
        FunctionTool(
            name="create_monitor",
            description=(
                "Subscribe to a bus channel; each fire posts a "
                "source='monitor' item to the session inbox under a stable "
                "dedup_key (latest fire replaces the prior undrained one). "
                "Condition-polling is not yet supported."
            ),
            parameters=pydantic_to_tool_schema(_CreateMonitorParams),
            fn=manager.create_monitor,
        )
    )
    api.register_tool(
        FunctionTool(
            name="list_monitors",
            description="List every live monitor (id, kind, watch, status).",
            parameters=pydantic_to_tool_schema(_ListMonitorsParams),
            fn=manager.list_monitors,
        )
    )
    api.register_tool(
        FunctionTool(
            name="cancel_monitor",
            description=(
                "Cancel one monitor by id. Cancels the wakeup task or "
                "unsubscribes the channel handler — never touches any shared "
                "session signal. Idempotent."
            ),
            parameters=pydantic_to_tool_schema(_CancelMonitorParams),
            fn=manager.cancel_monitor,
        )
    )
