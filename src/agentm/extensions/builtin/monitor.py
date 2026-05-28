"""Builtin ``monitor`` atom: agent-defined subscriptions and wakeups.

Design: ``.claude/designs/session-inbox.md`` (the ``monitor`` section) +
``.claude/plans/2026-05-28-session-inbox.md`` (step 4). Step 4 of the Session
Inbox work — the third producer atom on the inbox spine after ``background_exec``
(whose lifecycle / cancel / stale wiring this atom mirrors so we don't relearn
step-3's hard lessons).

What it exposes (all four are tools the agent calls):

* :func:`schedule_wakeup(delay, note=None)` — one-shot timer. ``await
  asyncio.sleep(delay)`` then post a ``source="monitor"`` inbox item with
  ``payload.kind="wakeup"``. Returns the monitor's id and a pending status.
* :func:`create_monitor(watch, note=None)` — subscribe to a bus channel; on
  each fire post a ``source="monitor"`` item with ``payload.kind="channel"``
  under a stable ``dedup_key`` so repeated fires REPLACE the prior undrained
  item rather than stacking (same discipline as the step-3 ticker).
* :func:`list_monitors` — every live monitor's id / kind / watch / status.
* :func:`cancel_monitor(monitor_id)` — cancels the wakeup task or unsubscribes
  the channel handler. Idempotent (unknown id → tool-error result).

**MVP scope (per plan).** ``create_monitor`` supports bus-channel subscriptions
only — condition-polling is deferred and the ``condition`` argument returns a
tool-error ``ToolResult`` so the agent sees the refusal explicitly (no
``NotImplementedError`` is raised through the tool surface).

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

from agentm.core.abi import (
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.extension import (
    ExtensionAPI,
    ExtensionStaleError,
    Unsubscribe,
)
from agentm.core.lib import DEFAULT_SHUTDOWN_GRACE_SECONDS, to_jsonable
from agentm.extensions import ExtensionManifest

_PENDING: Literal["pending"] = "pending"
_FIRED: Literal["fired"] = "fired"
_ACTIVE: Literal["active"] = "active"
_CANCELLED: Literal["cancelled"] = "cancelled"
_Status = Literal["pending", "fired", "active", "cancelled"]

_KIND_WAKEUP: Literal["wakeup"] = "wakeup"
_KIND_CHANNEL: Literal["channel"] = "channel"
# ``_KIND_CONDITION`` constant deliberately not declared until the
# condition-polling form lands (see plan); the type alias keeps "condition"
# only as a forward-compatible literal for the defense branch.
_Kind = Literal["wakeup", "channel", "condition"]

# Default for the silenced channel-fire event summary cap. The atom exposes a
# ``event_summary_max_chars`` config knob so a scenario can widen it without
# touching the atom.
_DEFAULT_EVENT_SUMMARY_MAX = 200

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
    config_schema={
        "type": "object",
        "properties": {
            "shutdown_grace_seconds": {
                "type": "number",
                "minimum": 0,
                "default": DEFAULT_SHUTDOWN_GRACE_SECONDS,
                "description": (
                    "Seconds the session_shutdown drain waits for still-"
                    "pending wakeup tasks to be cancelled and gathered "
                    f"(default {DEFAULT_SHUTDOWN_GRACE_SECONDS:g})."
                ),
            },
            "event_summary_max_chars": {
                "type": "integer",
                "minimum": 1,
                "default": _DEFAULT_EVENT_SUMMARY_MAX,
                "description": (
                    "Maximum length of the channel-fire event_summary "
                    f"embedded in the inbox payload (default {_DEFAULT_EVENT_SUMMARY_MAX})."
                ),
            },
        },
        "additionalProperties": False,
    },
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
    task: asyncio.Task[Any] | None = None  # wakeup's asyncio.sleep task
    unsubscribe: Unsubscribe | None = None  # channel's bus unsubscribe


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload)))],
        is_error=is_error,
        extras=payload,
    )


def _event_summary(event: Any, *, max_chars: int) -> str:
    """Short, bounded repr of a bus event for the inbox payload.

    Kept tiny on purpose — the inbox item is a poke ("channel X fired"), not a
    transport for the event itself. The agent inspects the live state through
    other tools if it needs detail. ``max_chars`` is set per-manager from the
    atom config (``event_summary_max_chars``).
    """

    text = repr(event)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)] + "..."
    return text


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
        event_summary_max_chars: int = _DEFAULT_EVENT_SUMMARY_MAX,
    ) -> None:
        self._api = api
        self._monitors: dict[str, _Monitor] = {}
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._event_summary_max_chars = event_summary_max_chars
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

    async def create_monitor(self, args: dict[str, Any]) -> ToolResult:
        """Subscribe to a bus channel; on each fire post to the inbox.

        MVP supports bus-channel form only. The (deferred) condition-polling
        branch returns a tool-error result rather than silently doing nothing,
        so the agent gets explicit feedback that the form is not yet supported.
        Tracked as #178; re-introduce ``_KIND_CONDITION`` when it lands.
        """

        condition = args.get("condition")
        if condition is not None:
            return _tool_result(
                {
                    "error": (
                        "condition-polling form of create_monitor is not "
                        "implemented (MVP supports bus-channel subscription "
                        "only — use the 'watch' argument with a channel name)"
                    ),
                },
                is_error=True,
            )
        watch = args.get("watch")
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
                    event, max_chars=self._event_summary_max_chars
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
        if state.kind == _KIND_WAKEUP and state.task is not None:
            if not state.task.done():
                state.task.cancel()
        elif state.kind == _KIND_CHANNEL and state.unsubscribe is not None:
            try:
                state.unsubscribe()
            except Exception:  # noqa: BLE001
                # An unsubscribe failure must not crash the tool — the monitor
                # is now marked cancelled and its handler is the no-op branch.
                pass
        return _tool_result({"monitor_id": monitor_id, "status": _CANCELLED})

    # --- lifecycle ---------------------------------------------------------

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        """Cancel every wakeup task and clear every channel subscription.

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
            if state.kind == _KIND_WAKEUP and state.task is not None:
                if not state.task.done():
                    state.task.cancel()
                    wakeup_tasks.append(state.task)
            elif state.kind == _KIND_CHANNEL and state.unsubscribe is not None:
                try:
                    state.unsubscribe()
                except Exception:  # noqa: BLE001
                    pass
        if not wakeup_tasks:
            return
        await asyncio.wait(wakeup_tasks, timeout=self._shutdown_grace_seconds)
        # Gather so any CancelledError / late exception is retrieved cleanly
        # (no "Task exception was never retrieved" past shutdown).
        await asyncio.gather(*wakeup_tasks, return_exceptions=True)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    manager = _MonitorManager(
        api=api,
        shutdown_grace_seconds=float(
            config.get("shutdown_grace_seconds", DEFAULT_SHUTDOWN_GRACE_SECONDS)
        ),
        event_summary_max_chars=int(
            config.get("event_summary_max_chars", _DEFAULT_EVENT_SUMMARY_MAX)
        ),
    )
    api.on(SessionShutdownEvent.CHANNEL, manager.on_session_shutdown)
    api.register_tool(
        FunctionTool(
            name="schedule_wakeup",
            description=(
                "Schedule a one-shot wakeup that posts to the session inbox "
                "after `delay` seconds. Returns a monitor_id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "delay": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Seconds to wait before firing.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Free-form note delivered with the wake.",
                    },
                },
                "required": ["delay"],
                "additionalProperties": False,
            },
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
            parameters={
                "type": "object",
                "properties": {
                    "watch": {
                        "type": "string",
                        "description": (
                            "Bus channel name to subscribe to "
                            "(e.g. 'tool_call', 'agent_end')."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": "Free-form note delivered with each fire.",
                    },
                },
                "required": ["watch"],
                "additionalProperties": False,
            },
            fn=manager.create_monitor,
        )
    )
    api.register_tool(
        FunctionTool(
            name="list_monitors",
            description="List every live monitor (id, kind, watch, status).",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
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
            parameters={
                "type": "object",
                "properties": {"monitor_id": {"type": "string"}},
                "required": ["monitor_id"],
                "additionalProperties": False,
            },
            fn=manager.cancel_monitor,
        )
    )
