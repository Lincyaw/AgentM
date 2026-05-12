"""Adapter: AgentM bus -> two-phase cognitive audit (v3).

See `.claude/designs/llmharness-cognitive-audit.md` for the full design.

Phase 1 (extractor) on every TurnEndEvent:
* Slice the trajectory window since the last cursor.
* Build a per-firing :class:`ExtractionState`, populate ``turn_texts``
  with rendered turn content (used by the witness pipeline).
* Spawn an extractor child whose extensions list carries the state via
  the ``state`` config knob. The child registers ``submit_events``
  closed over that state (single-tool flow; one shot per firing).
* After the child loop terminates, snapshot
  :class:`RawExtractorOutput` from the state and write entries:
  ``audit_event`` per accepted event, ``audit_edge`` per accepted edge,
  ``extractor_partial`` once if any edges were dropped, and
  ``extractor_cursor`` to mark the window consumed.
* Failure modes use typed entries: ``extractor_no_call`` (terminator
  never called), ``extractor_empty`` (terminator called but window had
  no events on a non-trivial slice), ``extractor_error`` (spawn / prompt
  / coercion crash). Cursor advances ONLY on success or partial.

Phase 2 (auditor) every k turns (v3, commit 4):
* Walk the entry tree to assemble the live event + edge graph plus the
  most-recent verdict's continuation_notes.
* Resolve ``llmharness.audit_registry`` from the parent ``ExtensionAPI``;
  build a frozen ``CheckContext`` and run every registered check, folding
  the resulting findings + check_errors into the auditor system prompt.
* Apply the N=30 degradation rule (knob ``audit_summary_threshold``):
  full witness fields embedded inline ≤ threshold; degraded shape +
  ``get_event_detail([ids])`` drill-down tool above threshold.
* Spawn the auditor child with ``compose_auditor_extensions(...)`` —
  ``events`` / ``edges`` are bridged to the new ``get_event_detail``
  tool atom via the ``config`` dict (commit-3 pattern).
* Persist the verdict and (if ``surface_reminder``) queue the reminder.

Reminder delivery (unified path):
* All queued reminders are drained on :class:`DecideTurnActionEvent`.
* Mid-trajectory turn (kernel default ``Step``): handler returns
  ``Inject([reminder_msgs])`` — extends ``messages`` and continues, same
  visible effect as appending a synthetic user message before the next
  LLM call.
* Terminal turn (kernel default ``Stop`` with non-final cause, e.g.
  ``ToolTerminated`` from ``submit_final_report`` / ``ModelEndTurn``):
  handler returns ``Inject([reminder_msgs])`` which overrides the stop
  and re-opens the loop so the model sees the reminder + may revise.
* Final-cause stops (``MaxTurnsExhausted`` / ``SignalAborted`` / etc.)
  ignore overrides; the handler logs a warning and leaves the reminder
  pending — there is no safe way to re-open the loop in that case.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from agentm.core.abi import (
    DecideTurnActionEvent,
    Inject,
    LoopAction,
    Stop,
    TurnEndEvent,
)
from agentm.core.abi.events import DiagnosticEvent, SessionShutdownEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    ToolResultMessage,
    text_message,
)
from agentm.core.abi.session import SessionEntry
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest

from ..audit import entry_types as _et
from ..audit._session_helpers import (
    bind_extractor_state,
    find_terminal_tool_arguments,
    safe_shutdown,
)
from ..audit.auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
    compose_auditor_extensions,
)
from ..audit.extractor import (
    SUBMIT_EVENTS_TOOL_NAME,
    ExtractionState,
    RawExtractorOutput,
    compose_extractor_extensions,
)
from ..audit.phase import merge_to_phases
from ..audit.registry import SERVICE_KEY as AUDIT_REGISTRY_SERVICE_KEY
from ..audit.registry import AuditCheckRegistry, CheckContext
from ..replay.record import (
    ReplayRecord,
    now_ns,
    replay_log_path,
    write_record,
)
from ..schema import Edge, Event, Phase, Reminder, Verdict

_logger = logging.getLogger(__name__)

MANIFEST = ExtensionManifest(
    name="agentm",
    description=(
        "Two-phase cognitive-audit adapter (v3): per-turn extractor (Phase 1) "
        "with witness-based edge construction and an every-k-turns graph "
        "auditor (Phase 2). ``mode='async'`` (default) runs audit on a "
        "background worker so the main agent loop is never blocked; verdicts "
        "arrive as synthetic user messages via decide_turn_action.Inject "
        "(extending Step turns and overriding non-final Stop terminations so "
        "a terminal-turn reminder re-opens the loop), and the "
        "session_shutdown handler drains the queue. ``mode='sync'`` runs "
        "audit inline at turn_end — slower but guarantees every turn has "
        "paired audit data, suitable for dataset collection / offline "
        "distillation."
    ),
    registers=(
        "event:turn_end",
        "event:decide_turn_action",
        "event:session_shutdown",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["async", "sync"]},
            "audit_interval_turns": {"type": "integer", "minimum": 1},
            "audit_summary_threshold": {"type": "integer", "minimum": 0},
            "prompt_override_extractor": {"type": "string"},
            "prompt_override_auditor": {"type": "string"},
            "cards_tools_config": {"type": ["object", "null"]},
            "observability_config": {"type": ["object", "null"]},
            "shutdown_timeout_s": {"type": "number", "minimum": 0},
            "extractor_provider": {
                "type": ["object", "null"],
                "properties": {
                    "module": {"type": "string"},
                    "config": {"type": "object"},
                },
                "required": ["module"],
                "additionalProperties": False,
            },
            "auditor_provider": {
                "type": ["object", "null"],
                "properties": {
                    "module": {"type": "string"},
                    "config": {"type": "object"},
                },
                "required": ["module"],
                "additionalProperties": False,
            },
            "enable_auditor": {
                "type": "boolean",
                "description": (
                    "Run the auditor phase (default true). Set false to "
                    "collect a clean extractor-only event graph without "
                    "auditor LLM cost or reminder side-effects — useful for "
                    "dataset collection and isolating auditor-quality "
                    "regressions from extractor coverage."
                ),
            },
            "enable_reminders": {
                "type": "boolean",
                "description": (
                    "Inject auditor verdicts as harness advisories on the "
                    "main agent (default true). Set false to run the full "
                    "audit pipeline (extractor + auditor) for data "
                    "collection while leaving the main agent unaffected. "
                    "Ignored when ``enable_auditor: false``."
                ),
            },
            "enable_replay_log": {
                "type": "boolean",
                "description": (
                    "Append each phase invocation to "
                    "``<cwd>/.agentm/audit_replay/<root_session_id>.jsonl`` "
                    "for offline replay (default true). One record carries "
                    "the full compose-kwargs + payload + parsed output, so "
                    "``llmharness-replay {extractor|auditor} --record ...`` "
                    "can rebuild the exact extension list + payload and "
                    "swap provider / prompt for A/B."
                ),
            },
        },
        "additionalProperties": False,
    },
    affects=(
        "event:turn_end",
        "event:decide_turn_action",
        "event:session_shutdown",
    ),
    api_version=1,
    tier=1,
)


_DEFAULT_AUDIT_INTERVAL_TURNS = 3
_DEFAULT_RECENT_VERDICTS = _et.RECENT_VERDICTS_FOR_AUDITOR
_RECENT_GRAPH_SLICE_FOR_EXTRACTOR = _et.RECENT_GRAPH_SLICE_FOR_EXTRACTOR
_DEFAULT_SHUTDOWN_TIMEOUT_S = 60.0
_DEFAULT_MODE = "async"
_DEFAULT_AUDIT_SUMMARY_THRESHOLD = 30

_REMINDER_PREAMBLE = (
    "[harness advisory — meta-injection from cognitive audit, not from the human user]\n"
)

# Entry-type bindings (every literal must come from entry_types.py).
_AUDIT_EVENT_ENTRY_TYPE = _et.AUDIT_EVENT
_AUDIT_EDGE_ENTRY_TYPE = _et.AUDIT_EDGE
_AUDIT_PHASE_ENTRY_TYPE = _et.AUDIT_PHASE
_VERDICT_ENTRY_TYPE = _et.VERDICT
_EXTRACTOR_CURSOR_ENTRY_TYPE = _et.EXTRACTOR_CURSOR
_REMINDER_DELIVERED_ENTRY_TYPE = _et.REMINDER_DELIVERED

_EXTRACTOR_NO_CALL_ENTRY = _et.EXTRACTOR_NO_CALL
_EXTRACTOR_ERROR_ENTRY = _et.EXTRACTOR_ERROR
_EXTRACTOR_EMPTY_ENTRY = _et.EXTRACTOR_EMPTY
_EXTRACTOR_PARTIAL_ENTRY = _et.EXTRACTOR_PARTIAL
_AUDIT_NO_CALL_ENTRY = _et.AUDIT_NO_CALL
_AUDIT_ERROR_ENTRY = _et.AUDIT_ERROR


# --- branch state -----------------------------------------------------------


@dataclass(frozen=True)
class _BranchState:
    """Snapshot of audit-relevant entries pulled from a single branch walk."""

    cursor_last_turn_index: int
    graph: list[Event]
    edges: list[Edge]
    phases: list[Phase]
    recent_verdicts: list[dict[str, Any]]
    last_continuation_notes: list[str]


def _scan_branch(branch: list[SessionEntry], *, recent_verdicts_n: int) -> _BranchState:
    """Single-pass extraction of cursor + graph + edges + phases + verdicts."""
    cursor_last_turn_index = -1
    graph: list[Event] = []
    edges: list[Edge] = []
    phases: list[Phase] = []
    verdicts: list[dict[str, Any]] = []

    for entry in branch:
        payload = entry.payload
        if not isinstance(payload, dict):
            continue
        if entry.type == _AUDIT_EVENT_ENTRY_TYPE:
            try:
                graph.append(Event.from_dict(payload))
            except (KeyError, ValueError, TypeError):
                continue
        elif entry.type == _AUDIT_EDGE_ENTRY_TYPE:
            try:
                edges.append(Edge.from_dict(payload))
            except (KeyError, ValueError, TypeError):
                continue
        elif entry.type == _AUDIT_PHASE_ENTRY_TYPE:
            try:
                phases.append(Phase.from_dict(payload))
            except (KeyError, ValueError, TypeError):
                continue
        elif entry.type == _VERDICT_ENTRY_TYPE:
            verdicts.append(payload)
        elif entry.type == _EXTRACTOR_CURSOR_ENTRY_TYPE:
            raw = payload.get("last_turn_index")
            if isinstance(raw, int) and not isinstance(raw, bool):
                cursor_last_turn_index = raw

    last_continuation_notes: list[str] = []
    if verdicts:
        raw_notes = verdicts[-1].get("continuation_notes")
        if isinstance(raw_notes, list):
            last_continuation_notes = [n for n in raw_notes if isinstance(n, str)]

    return _BranchState(
        cursor_last_turn_index=cursor_last_turn_index,
        graph=graph,
        edges=edges,
        phases=phases,
        recent_verdicts=verdicts[-recent_verdicts_n:] if recent_verdicts_n > 0 else [],
        last_continuation_notes=last_continuation_notes,
    )


# --- failure recording ------------------------------------------------------


_FAILURE_DIAGNOSTIC_LEVEL: Literal["warning"] = "warning"


def _record_failure(api: ExtensionAPI, entry_type: str, payload: dict[str, Any]) -> None:
    """Single chokepoint for typed failure entries.

    Append-only on the session branch (consumed by ``_scan_branch`` and
    downstream eval), AND simultaneously emit a ``DiagnosticEvent`` so
    that the failure shows up in the OTel jsonl. Without the diagnostic
    leg an audit-child crash leaves zero evidence on disk — burning the
    extractor / auditor LLM quota and producing nothing — exactly the
    silent-no-op mode that masked the post-harness-collapse Operations
    fail-stop until a 50-case run was already spent. Never raises.
    """
    api.session.append_entry(entry_type, payload)
    reason_raw = payload.get("reason") if isinstance(payload, dict) else None
    reason = str(reason_raw) if reason_raw is not None else ""
    message = f"{entry_type}: {reason}" if reason else entry_type
    try:
        api.events.emit_sync(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level=_FAILURE_DIAGNOSTIC_LEVEL,
                source="llmharness.audit",
                message=message,
            ),
        )
    except Exception:
        # Diagnostics must never break the audit loop. Fall through.
        _logger.exception("llmharness audit diagnostic emit failed; suppressing.")


def _window_is_non_trivial(messages_slice: list[AgentMessage]) -> bool:
    """True iff the slice contains any AssistantMessage or ToolResultMessage."""
    return any(isinstance(msg, (AssistantMessage, ToolResultMessage)) for msg in messages_slice)


# --- jobs -------------------------------------------------------------------


@dataclass(frozen=True)
class _ExtractorJob:
    """Snapshot taken synchronously at TurnEndEvent; consumed by worker."""

    messages: tuple[AgentMessage, ...]


@dataclass(frozen=True)
class _AuditorJob:
    """Trajectory snapshot taken at TurnEndEvent time; consumed by the worker."""

    messages: tuple[AgentMessage, ...]


@dataclass(frozen=True)
class _AuditorSettings:
    """Per-install knobs for assembling a per-firing auditor extension list."""

    prompt_override: str | None
    cards_tools_config: dict[str, Any] | None
    observability_config: dict[str, Any] | None
    summary_threshold: int


@dataclass(frozen=True)
class _ShutdownJob:
    """Worker-stop sentinel; consumed only by ``_drain_queue``."""


_Job = _ExtractorJob | _AuditorJob | _ShutdownJob


# --- install ----------------------------------------------------------------


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    mode_raw = config.get("mode", _DEFAULT_MODE)
    mode = mode_raw if mode_raw in ("async", "sync") else _DEFAULT_MODE

    k = int(config.get("audit_interval_turns", _DEFAULT_AUDIT_INTERVAL_TURNS))
    if k < 1:
        k = _DEFAULT_AUDIT_INTERVAL_TURNS

    summary_threshold_raw = config.get("audit_summary_threshold", _DEFAULT_AUDIT_SUMMARY_THRESHOLD)
    try:
        summary_threshold = int(summary_threshold_raw)
    except (TypeError, ValueError):
        summary_threshold = _DEFAULT_AUDIT_SUMMARY_THRESHOLD
    if summary_threshold < 0:
        summary_threshold = _DEFAULT_AUDIT_SUMMARY_THRESHOLD

    shutdown_timeout = float(config.get("shutdown_timeout_s", _DEFAULT_SHUTDOWN_TIMEOUT_S))
    if shutdown_timeout < 0:
        shutdown_timeout = _DEFAULT_SHUTDOWN_TIMEOUT_S

    cards_cfg_raw = config.get("cards_tools_config", {})
    obs_cfg_raw = config.get("observability_config", {})
    cards_cfg = cards_cfg_raw if isinstance(cards_cfg_raw, dict) else None
    obs_cfg = obs_cfg_raw if isinstance(obs_cfg_raw, dict) else None

    prompt_extractor_raw = config.get("prompt_override_extractor")
    prompt_auditor_raw = config.get("prompt_override_auditor")
    prompt_extractor = prompt_extractor_raw if isinstance(prompt_extractor_raw, str) else None
    prompt_auditor = prompt_auditor_raw if isinstance(prompt_auditor_raw, str) else None

    extractor_extensions = compose_extractor_extensions(
        prompt_override=prompt_extractor,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
    )
    api._llmharness_extractor_compose_kwargs = {  # type: ignore[attr-defined]
        "prompt_override": prompt_extractor,
        "cards_tools_config": cards_cfg,
        "observability_config": obs_cfg,
    }
    # Auditor extensions are rebuilt per firing in _drain_auditor because
    # the v3 prompt is templated over the live event/edge graph + findings.
    auditor_settings = _AuditorSettings(
        prompt_override=prompt_auditor,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
        summary_threshold=summary_threshold,
    )

    extractor_provider = _parse_provider_spec(config.get("extractor_provider"))
    auditor_provider = _parse_provider_spec(config.get("auditor_provider"))

    enable_auditor = bool(config.get("enable_auditor", True))
    enable_reminders = bool(config.get("enable_reminders", True))
    enable_replay_log = bool(config.get("enable_replay_log", True))

    if enable_replay_log:
        api._llmharness_replay_log_path = replay_log_path(api.cwd, api.root_session_id)  # type: ignore[attr-defined]
    else:
        api._llmharness_replay_log_path = None  # type: ignore[attr-defined]

    # Publish the audit-check registry on the parent session. Atoms in
    # later commits (reference checks etc.) call
    # ``api.get_service(AUDIT_REGISTRY_SERVICE_KEY).register_check(...)``
    # from their own ``install``. Best-effort: a duplicate registration
    # (e.g. installing this adapter twice) is ignored.
    with contextlib.suppress(KeyError):
        api.set_service(AUDIT_REGISTRY_SERVICE_KEY, AuditCheckRegistry())

    pending_reminders: list[Reminder] = []
    turn_count = 0

    if mode == "sync":

        async def _on_turn_end_sync(event: TurnEndEvent) -> None:
            nonlocal turn_count
            turn_count += 1
            messages_snapshot = tuple(event.messages)
            extractor_ok = await _drain_extractor(
                api=api,
                job=_ExtractorJob(messages=messages_snapshot),
                extractor_extensions=extractor_extensions,
                extractor_provider=extractor_provider,
            )
            if enable_auditor and extractor_ok and (turn_count % k) == 0:
                await _drain_auditor(
                    api=api,
                    auditor_settings=auditor_settings,
                    auditor_provider=auditor_provider,
                    pending_reminders=pending_reminders,
                    messages=list(messages_snapshot),
                )

        api.on(TurnEndEvent.CHANNEL, _on_turn_end_sync)
        if enable_reminders:
            api.on(
                DecideTurnActionEvent.CHANNEL,
                _make_reminder_injector(api, pending_reminders),
            )
        return

    # Async path: queue + background worker.
    queue: asyncio.Queue[_Job] = asyncio.Queue()
    worker_task: asyncio.Task[None] | None = None

    def _ensure_worker() -> None:
        nonlocal worker_task
        if worker_task is None or worker_task.done():
            worker_task = asyncio.create_task(
                _drain_queue(
                    api=api,
                    queue=queue,
                    pending_reminders=pending_reminders,
                    extractor_extensions=extractor_extensions,
                    auditor_settings=auditor_settings,
                    extractor_provider=extractor_provider,
                    auditor_provider=auditor_provider,
                ),
                name="llmharness-audit-worker",
            )

    def _on_turn_end(event: TurnEndEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        _ensure_worker()
        messages_snapshot = tuple(event.messages)
        queue.put_nowait(_ExtractorJob(messages=messages_snapshot))
        if enable_auditor and (turn_count % k) == 0:
            queue.put_nowait(_AuditorJob(messages=messages_snapshot))

    async def _on_session_shutdown(_event: SessionShutdownEvent) -> None:
        if worker_task is None or worker_task.done():
            return
        queue.put_nowait(_ShutdownJob())
        try:
            await asyncio.wait_for(worker_task, timeout=shutdown_timeout)
        except asyncio.TimeoutError:
            _logger.warning(
                "llmharness audit drain exceeded %.1fs; cancelling worker; "
                "%d jobs may be unpersisted",
                shutdown_timeout,
                queue.qsize(),
            )
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker_task
        except Exception:
            _logger.exception("llmharness audit worker raised on shutdown")

    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
    if enable_reminders:
        api.on(
            DecideTurnActionEvent.CHANNEL,
            _make_reminder_injector(api, pending_reminders),
        )
    api.on(SessionShutdownEvent.CHANNEL, _on_session_shutdown)


def _parse_provider_spec(
    raw: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Coerce ``{"module": "...", "config": {...}}`` to ``AgentSessionConfig.provider``."""
    if not isinstance(raw, dict) or not raw:
        return None
    module = raw.get("module")
    if not isinstance(module, str) or not module.strip():
        return None
    cfg = raw.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    try:
        from agentm.ai.types import DEFAULT_PROVIDER_REGISTRY
    except ImportError:
        return module, dict(cfg)
    try:
        return DEFAULT_PROVIDER_REGISTRY.build(module, cfg)
    except KeyError:
        return module, dict(cfg)


def _make_reminder_injector(
    api: ExtensionAPI, pending_reminders: list[Reminder]
) -> Callable[[DecideTurnActionEvent], LoopAction | None]:
    """Drain pending reminders into the loop via DecideTurnAction.

    Unified path: a reminder produced mid-trajectory (default action would
    have been ``Step``) becomes ``Inject(reminder_msgs)`` and the loop
    extends ``messages`` and continues — identical visible behavior to
    appending the reminder before the next LLM call. A reminder produced
    on the terminal turn (default action is ``Stop(non_final_cause)``)
    likewise becomes ``Inject``, which overrides the kernel's stop and
    re-opens the loop so the model sees the reminder + can revise. Stops
    flagged ``cause.final`` (kernel-imposed terminations like
    MaxTurnsExhausted / SignalAborted) ignore overrides; we leave the
    reminder pending and warn — there is no safe way to re-open then.
    """

    def _on_decide(event: DecideTurnActionEvent) -> LoopAction | None:
        if not pending_reminders:
            return None
        default = event.observation.default_action
        if isinstance(default, Stop) and default.cause.final:
            _logger.warning(
                "llmharness audit reminder pending but loop default is "
                "final %s; reminder will not be delivered",
                type(default.cause).__name__,
            )
            return None
        injected: list[AgentMessage] = []
        while pending_reminders:
            reminder = pending_reminders.pop(0)
            injected.append(
                text_message(
                    _REMINDER_PREAMBLE + reminder.text, timestamp=time.time()
                )
            )
            try:
                api.session.append_entry(
                    _REMINDER_DELIVERED_ENTRY_TYPE,
                    {"text": reminder.text},
                )
            except Exception:
                _logger.exception("failed to persist reminder_delivered entry")
        return Inject(messages=injected)

    return _on_decide


async def _drain_queue(
    *,
    api: ExtensionAPI,
    queue: asyncio.Queue[_Job],
    pending_reminders: list[Reminder],
    extractor_extensions: list[tuple[str, dict[str, Any]]],
    auditor_settings: _AuditorSettings,
    extractor_provider: tuple[str, dict[str, Any]] | None,
    auditor_provider: tuple[str, dict[str, Any]] | None,
) -> None:
    """Serial worker. Owns all session-mutating audit writes."""
    _last_extractor_held_cursor: bool = False

    while True:
        try:
            job = await queue.get()
        except asyncio.CancelledError:
            raise
        try:
            if isinstance(job, _ShutdownJob):
                return
            if isinstance(job, _ExtractorJob):
                extractor_ok = await _drain_extractor(
                    api=api,
                    job=job,
                    extractor_extensions=extractor_extensions,
                    extractor_provider=extractor_provider,
                )
                _last_extractor_held_cursor = not extractor_ok
            elif isinstance(job, _AuditorJob):
                if _last_extractor_held_cursor:
                    _logger.debug(
                        "llmharness audit worker: skipping auditor — "
                        "preceding extractor firing held the cursor"
                    )
                else:
                    await _drain_auditor(
                        api=api,
                        auditor_settings=auditor_settings,
                        auditor_provider=auditor_provider,
                        pending_reminders=pending_reminders,
                        messages=list(job.messages),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.exception("llmharness audit worker job failed")
        finally:
            queue.task_done()


# --- extractor (Phase 1, v3) ------------------------------------------------


async def _drain_extractor(
    *,
    api: ExtensionAPI,
    job: _ExtractorJob,
    extractor_extensions: list[tuple[str, dict[str, Any]]],
    extractor_provider: tuple[str, dict[str, Any]] | None,
) -> bool:
    """Run one v3 extractor firing.

    Returns ``True`` on success or partial-success (cursor advanced).
    Returns ``False`` on no_call / empty / error paths (cursor held).
    """
    branch = api.session.get_branch()
    branch_state = _scan_branch(branch, recent_verdicts_n=_DEFAULT_RECENT_VERDICTS)
    messages = list(job.messages)

    window_lo = max(branch_state.cursor_last_turn_index + 1, 0)
    window_hi_inclusive = len(messages) - 1
    window_messages = messages[window_lo:]
    if not window_messages:
        # Nothing new to extract; do nothing (don't advance cursor).
        return True

    turn_window = [window_lo, window_hi_inclusive]

    # Build the per-firing ExtractionState. ``turn_texts`` keys are
    # absolute trajectory indices so the witness pipeline can resolve
    # ``src_turns`` / ``dst_turns`` against the rendered text.
    state = ExtractionState()
    for i, msg in enumerate(window_messages, start=window_lo):
        state.turn_texts[i] = _render_message_text(msg)

    # Build the new-turn JSON window for the prompt + payload.
    new_turn_window = [
        s
        for s in (
            _serialize_message_for_extractor(msg, index=i)
            for i, msg in enumerate(window_messages, start=window_lo)
        )
        if s is not None
    ]

    payload = {
        "new_turns": new_turn_window,
        "recent_graph": [
            e.to_dict() for e in branch_state.graph[-_RECENT_GRAPH_SLICE_FOR_EXTRACTOR:]
        ],
    }

    turn_window_json = json.dumps(new_turn_window, ensure_ascii=False, default=str)

    # Inject state + turn-window JSON substitution into the per-firing
    # extensions list. The base list returned by
    # ``compose_extractor_extensions`` is shared across firings; we
    # never mutate it.
    firing_extensions = bind_extractor_state(
        extractor_extensions,
        state=state,
        turn_window_json=turn_window_json,
    )

    # Only materialize the replay snapshot if the sidecar is enabled.
    # turn_texts can be large in long sessions; skip the copy otherwise.
    replay_path = _replay_log_path_for(api)
    if replay_path is not None:
        replay_compose_kwargs: dict[str, Any] | None = dict(
            getattr(api, "_llmharness_extractor_compose_kwargs", {})
        )
        replay_extras: dict[str, Any] | None = {
            "turn_window_json": turn_window_json,
            "turn_texts": {str(k): v for k, v in state.turn_texts.items()},
        }
    else:
        replay_compose_kwargs = None
        replay_extras = None

    def _record(status: str, output: dict[str, Any] | None, error: str | None = None) -> None:
        _record_replay_at(
            replay_path,
            phase="extractor",
            turn_index=window_hi_inclusive,
            root_session_id=api.root_session_id,
            compose_kwargs=replay_compose_kwargs,
            payload=payload,
            provider=extractor_provider,
            output=output,
            status=status,
            error=error,
            extras=replay_extras,
        )

    try:
        terminator_called = await _spawn_extractor_child(
            api=api,
            extensions=firing_extensions,
            provider=extractor_provider,
            payload=payload,
            turn_window=turn_window,
        )
    except _ExtractorSpawnError as exc:
        _record_failure(
            api,
            _EXTRACTOR_ERROR_ENTRY,
            {"reason": str(exc), "turn_window": turn_window},
        )
        _record("spawn_error", output=None, error=str(exc))
        return False

    if not terminator_called:
        _record_failure(
            api,
            _EXTRACTOR_NO_CALL_ENTRY,
            {
                "reason": (f"child returned without calling {SUBMIT_EVENTS_TOOL_NAME}"),
                "turn_window": turn_window,
            },
        )
        _record("no_call", output=None)
        return False

    output = RawExtractorOutput.from_state(state)
    _record(
        "ok",
        output={
            "events": [e.to_dict() for e in output.events],
            "edges": [ed.to_dict() for ed in output.edges],
            "dropped_edges": list(output.dropped_edges),
        },
    )

    # Empty submission on a non-trivial window is a typed failure; on a
    # trivial (user-only) window an empty output is normal.
    if not output.events and not output.edges and not output.dropped_edges:
        if _window_is_non_trivial(window_messages):
            _record_failure(api, _EXTRACTOR_EMPTY_ENTRY, {"turn_window": turn_window})
            return False
        # Truly trivial window: still advance the cursor so we don't
        # re-extract the same prefix forever.
        api.session.append_entry(
            _EXTRACTOR_CURSOR_ENTRY_TYPE,
            {
                "last_turn_index": window_hi_inclusive,
                "extraction_run_id": uuid.uuid4().hex,
            },
        )
        return True

    for ev in output.events:
        api.session.append_entry(_AUDIT_EVENT_ENTRY_TYPE, ev.to_dict())
    for ed in output.edges:
        api.session.append_entry(_AUDIT_EDGE_ENTRY_TYPE, ed.to_dict())
    # Mechanical phase merge over the just-extracted events. Phases
    # collapse consecutive ``act`` / ``evid`` runs into one block so the
    # auditor reads a "basic block" view; raw events stay on disk for
    # drill-down via ``get_event_detail``.
    for ph in merge_to_phases(output.events):
        api.session.append_entry(_AUDIT_PHASE_ENTRY_TYPE, ph.to_dict())
    if output.dropped_edges:
        api.session.append_entry(
            _EXTRACTOR_PARTIAL_ENTRY,
            {
                "dropped_edges": list(output.dropped_edges),
                "turn_window": turn_window,
            },
        )

    api.session.append_entry(
        _EXTRACTOR_CURSOR_ENTRY_TYPE,
        {
            "last_turn_index": window_hi_inclusive,
            "extraction_run_id": uuid.uuid4().hex,
        },
    )
    return True


def _replay_log_path_for(api: ExtensionAPI) -> Path | None:
    """Resolve the per-session replay sidecar path, or None if disabled."""
    return getattr(api, "_llmharness_replay_log_path", None)


def _record_replay_at(
    path: Path | None,
    *,
    phase: str,
    turn_index: int,
    root_session_id: str,
    compose_kwargs: dict[str, Any] | None,
    payload: dict[str, Any] | None,
    provider: tuple[str, dict[str, Any]] | None,
    output: dict[str, Any] | None,
    status: str,
    error: str | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Append one record; ``path is None`` short-circuits.

    ``write_record`` swallows OSError internally; programmer errors
    (malformed kwargs) should surface, not be silently dropped.
    """
    if path is None:
        return
    rec = ReplayRecord(
        phase=phase,  # type: ignore[arg-type]
        turn_index=turn_index,
        root_session_id=root_session_id,
        ts_ns=now_ns(),
        compose_kwargs=compose_kwargs or {},
        payload=payload or {},
        provider=[provider[0], provider[1]] if provider else None,
        output=output,
        status=status,  # type: ignore[arg-type]
        error=error,
        extras=extras or {},
    )
    write_record(path, rec)


class _ExtractorSpawnError(RuntimeError):
    """Wraps spawn / prompt failures so the caller can record them."""


async def _spawn_extractor_child(
    *,
    api: ExtensionAPI,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    payload: dict[str, Any],
    turn_window: list[int],
) -> bool:
    """Run the extractor child. Returns True iff submit_events was called.

    Raises :class:`_ExtractorSpawnError` for spawn / prompt failures so
    the caller can route them to the typed failure path.
    """
    del turn_window  # surfaced by caller via _record_failure context
    child_config = AgentSessionConfig(
        cwd=api.cwd,
        provider=provider,
        extensions=extensions,
        purpose="cognitive_audit_extractor",
    )
    try:
        child = await api.spawn_child_session(child_config)
    except Exception as exc:
        raise _ExtractorSpawnError(str(exc)) from exc

    try:
        messages = await child.prompt(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:
        await safe_shutdown(child)
        raise _ExtractorSpawnError(str(exc)) from exc

    await safe_shutdown(child)
    return _has_tool_call(messages, SUBMIT_EVENTS_TOOL_NAME)


def _has_tool_call(messages: list[AgentMessage], tool_name: str) -> bool:
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return True
    return False




def _render_message_text(msg: AgentMessage) -> str:
    """Render the text content of a message for witness substring checks.

    Concatenates every text-bearing block (assistant text, thinking,
    tool-result text, user text) into a single string. The witness
    layer normalises whitespace and case before substring matching so
    block boundaries don't matter.
    """
    parts: list[str] = []
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            inner = getattr(block, "content", None)
            if isinstance(inner, list):
                for sub in inner:
                    sub_text = getattr(sub, "text", None)
                    if isinstance(sub_text, str) and sub_text:
                        parts.append(sub_text)
            args = getattr(block, "arguments", None)
            if isinstance(args, dict):
                with contextlib.suppress(TypeError, ValueError):
                    parts.append(json.dumps(args, ensure_ascii=False, default=str))
    return " ".join(parts)


# --- auditor (Phase 2, unchanged from v2) -----------------------------------


async def _drain_auditor(
    *,
    api: ExtensionAPI,
    auditor_settings: _AuditorSettings,
    auditor_provider: tuple[str, dict[str, Any]] | None,
    pending_reminders: list[Reminder],
    messages: list[AgentMessage],
) -> None:
    branch = api.session.get_branch()
    branch_state = _scan_branch(branch, recent_verdicts_n=_DEFAULT_RECENT_VERDICTS)

    events_tuple: tuple[Event, ...] = tuple(branch_state.graph)
    edges_tuple: tuple[Edge, ...] = tuple(branch_state.edges)
    phases_tuple: tuple[Phase, ...] = tuple(branch_state.phases)

    # Run scenario-registered checks. Empty registry / unset service →
    # empty findings, no error (design §4.c).
    findings: list[Any] = []
    check_errors: dict[str, str] = {}
    try:
        registry = api.get_service(AUDIT_REGISTRY_SERVICE_KEY)
    except Exception:  # pragma: no cover - defensive, no service registry path
        registry = None
    if isinstance(registry, AuditCheckRegistry):
        try:
            ctx = CheckContext(events=events_tuple, edges=edges_tuple)
            findings, check_errors = registry.run_all(ctx)
        except Exception:
            _logger.exception("audit-check registry run_all failed; using empty findings")
            findings, check_errors = [], {}

    trajectory_snapshot = _serialize_full_trajectory(messages)
    firing_extensions = compose_auditor_extensions(
        prompt_override=auditor_settings.prompt_override,
        cards_tools_config=auditor_settings.cards_tools_config,
        observability_config=auditor_settings.observability_config,
        trajectory_snapshot=trajectory_snapshot,
        events=events_tuple,
        edges=edges_tuple,
        phases=phases_tuple,
        findings=list(findings),
        check_errors=dict(check_errors),
        continuation_notes=list(branch_state.last_continuation_notes),
        summary_threshold=auditor_settings.summary_threshold,
    )

    # Build the replay snapshot only when the sidecar is enabled — the
    # to_dict() comprehensions over events/edges/phases/findings dominate
    # per-firing CPU on large graphs.
    replay_path = _replay_log_path_for(api)
    replay_compose_kwargs: dict[str, Any] | None = None
    replay_payload: dict[str, Any] | None = None
    if replay_path is not None:
        replay_compose_kwargs = {
            "prompt_override": auditor_settings.prompt_override,
            "cards_tools_config": auditor_settings.cards_tools_config,
            "observability_config": auditor_settings.observability_config,
            "trajectory_snapshot": trajectory_snapshot,
            "events": [e.to_dict() for e in events_tuple],
            "edges": [ed.to_dict() for ed in edges_tuple],
            "phases": [ph.to_dict() for ph in phases_tuple],
            "findings": [f.to_dict() for f in findings],
            "check_errors": dict(check_errors),
            "continuation_notes": list(branch_state.last_continuation_notes),
            "summary_threshold": auditor_settings.summary_threshold,
        }
        replay_payload = {
            "graph": [e.to_dict() for e in branch_state.graph],
            "recent_verdicts": list(branch_state.recent_verdicts),
            "continuation_notes_from_prior_firing": list(
                branch_state.last_continuation_notes
            ),
        }

    verdict = await _run_auditor(
        api=api,
        auditor_extensions=firing_extensions,
        provider=auditor_provider,
        graph_events=branch_state.graph,
        recent_verdicts=branch_state.recent_verdicts,
        continuation_notes_from_prior_firing=branch_state.last_continuation_notes,
    )
    turn_index = len(messages) - 1 if messages else -1
    if verdict is None:
        _record_replay_at(
            replay_path,
            phase="auditor",
            turn_index=turn_index,
            root_session_id=api.root_session_id,
            compose_kwargs=replay_compose_kwargs,
            payload=replay_payload,
            provider=auditor_provider,
            output=None,
            status="no_call",
        )
        return

    api.session.append_entry(_VERDICT_ENTRY_TYPE, verdict.to_dict())
    _record_replay_at(
        replay_path,
        phase="auditor",
        turn_index=turn_index,
        root_session_id=api.root_session_id,
        compose_kwargs=replay_compose_kwargs,
        payload=replay_payload,
        provider=auditor_provider,
        output=verdict.to_dict(),
        status="ok",
    )
    if verdict.surface_reminder and verdict.reminder_text:
        pending_reminders.append(Reminder(text=verdict.reminder_text))


# --- trajectory slicing -----------------------------------------------------


def _serialize_message_for_extractor(msg: AgentMessage, *, index: int) -> dict[str, Any] | None:
    """Best-effort dict view of one message."""
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return None

    blocks = [b for b in (_serialize_block(blk) for blk in content) if b is not None]
    if not blocks:
        return None
    return {
        "index": index,
        "role": getattr(msg, "role", "unknown"),
        "content": blocks,
    }


def _serialize_block(block: Any) -> dict[str, Any] | None:
    """Serialize one content block; preserves thinking + tool-result structure."""
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        block_type = getattr(block, "type", None)
        return {
            "type": block_type if isinstance(block_type, str) and block_type else "text",
            "text": text,
        }

    name = getattr(block, "name", None)
    arguments = getattr(block, "arguments", None)
    if isinstance(name, str) and isinstance(arguments, dict):
        return {
            "type": "tool_call",
            "id": getattr(block, "id", None),
            "name": name,
            "arguments": dict(arguments),
        }

    tool_call_id = getattr(block, "tool_call_id", None)
    inner_content = getattr(block, "content", None)
    if isinstance(tool_call_id, str) and isinstance(inner_content, list):
        inner_blocks: list[dict[str, Any]] = []
        for inner in inner_content:
            inner_text = getattr(inner, "text", None)
            if isinstance(inner_text, str):
                inner_blocks.append({"type": "text", "text": inner_text})
            else:
                inner_blocks.append(
                    {
                        "type": getattr(inner, "type", inner.__class__.__name__),
                        "repr": repr(inner),
                    }
                )
        return {
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "content": inner_blocks,
            "is_error": bool(getattr(block, "is_error", False)),
        }

    return {"type": getattr(block, "type", block.__class__.__name__), "repr": repr(block)}


def _serialize_full_trajectory(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Serialize the entire trajectory for the auditor's get_turn tool."""
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        serialized = _serialize_message_for_extractor(msg, index=i)
        if serialized is not None:
            out.append(serialized)
    return out


# --- phase runner -----------------------------------------------------------


_T = TypeVar("_T")


async def _run_phase(
    *,
    api: ExtensionAPI,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    purpose: str,
    payload: dict[str, Any],
    terminal_tool: str,
    coerce: Callable[[dict[str, Any]], _T],
    coerce_error: type[Exception],
    on_spawn_or_prompt_error: Callable[[str], None],
    on_no_call: Callable[[], None],
    on_malformed: Callable[[str], None],
) -> _T | None:
    """Spawn child, drive to terminal_tool, coerce, return result."""
    child_config = AgentSessionConfig(
        cwd=api.cwd,
        provider=provider,
        extensions=extensions,
        purpose=purpose,
    )
    try:
        child = await api.spawn_child_session(child_config)
    except Exception as exc:
        on_spawn_or_prompt_error(str(exc))
        return None

    try:
        messages = await child.prompt(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:
        on_spawn_or_prompt_error(str(exc))
        await safe_shutdown(child)
        return None

    await safe_shutdown(child)

    arguments = find_terminal_tool_arguments(messages, terminal_tool)
    if arguments is None:
        on_no_call()
        return None

    try:
        return coerce(arguments)
    except coerce_error as exc:
        on_malformed(str(exc))
        return None


async def _run_auditor(
    *,
    api: ExtensionAPI,
    auditor_extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    graph_events: list[Event],
    recent_verdicts: list[dict[str, Any]],
    continuation_notes_from_prior_firing: list[str],
) -> Verdict | None:
    payload = {
        "graph": [e.to_dict() for e in graph_events],
        "recent_verdicts": list(recent_verdicts),
        "continuation_notes_from_prior_firing": list(continuation_notes_from_prior_firing),
    }
    raw = await _run_phase(
        api=api,
        extensions=auditor_extensions,
        provider=provider,
        purpose="cognitive_audit_auditor",
        payload=payload,
        terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
        coerce=RawVerdictOutput.from_dict,
        coerce_error=AuditorOutputError,
        on_spawn_or_prompt_error=lambda reason: _record_failure(
            api, _AUDIT_ERROR_ENTRY, {"reason": reason}
        ),
        on_no_call=lambda: _record_failure(
            api,
            _AUDIT_NO_CALL_ENTRY,
            {"reason": f"child returned without calling {SUBMIT_VERDICT_TOOL_NAME}"},
        ),
        on_malformed=lambda reason: _record_failure(
            api, _AUDIT_ERROR_ENTRY, {"reason": f"malformed: {reason}"}
        ),
    )
    if raw is None:
        return None
    try:
        return raw.to_verdict()
    except AuditorOutputError as exc:
        _record_failure(api, _AUDIT_ERROR_ENTRY, {"reason": f"malformed: {exc}"})
        return None


__all__ = [
    "MANIFEST",
    "install",
]
