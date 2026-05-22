"""Adapter: AgentM bus -> two-phase cognitive audit (v3).

See `.claude/designs/llmharness-cognitive-audit.md` for the full design.

Phase 1 (extractor) on the configured TurnEndEvent cadence:
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
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

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
)
from agentm.core.abi.session import SessionEntry
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest

from ..audit import entry_types as _et
from ..audit._reminder_format import REMINDER_PREAMBLE as _SHARED_REMINDER_PREAMBLE
from ..audit._reminder_format import build_reminder_message
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
from ..audit.auditor.profiles import resolve_tools as _resolve_auditor_tools
from ..audit.auditor.prompt import load_auditor_prompt
from ..audit.extractor import (
    SUBMIT_EVENTS_TOOL_NAME,
    ExtractionState,
    RawExtractorOutput,
    compose_extractor_extensions,
)
from ..audit.extractor.prompt import load_extractor_prompt
from ..audit.graph_fold import fold_graph
from ..audit.graph_ops import (
    EdgeUpsert,
    GraphOp,
    NodeUpsert,
    parse_op,
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
            "extractor_interval_turns": {"type": "integer", "minimum": 1},
            "audit_interval_turns": {"type": "integer", "minimum": 1},
            "audit_summary_threshold": {"type": "integer", "minimum": 0},
            "prompt_override_extractor": {"type": "string"},
            "prompt_override_auditor": {"type": "string"},
            "extractor_prompt": {
                "type": "string",
                "description": (
                    "Named extractor prompt variant (file under "
                    "audit/extractor/prompts/) or an absolute path. "
                    "Default: 'default'. Overridden by "
                    "prompt_override_extractor when both are set."
                ),
            },
            "auditor_prompt": {
                "type": "string",
                "description": (
                    "Named auditor prompt variant (file under "
                    "audit/auditor/prompts/) or an absolute path. "
                    "Default: 'minimal'. Available: 'minimal', 'full'. "
                    "Overridden by prompt_override_auditor when both are set."
                ),
            },
            "auditor_profile": {
                "type": "string",
                "description": (
                    "Named auditor tool profile. Default: 'minimal' "
                    "(submit_verdict only). Available: 'minimal', "
                    "'with_drill_down'. Overridden by auditor_tools."
                ),
            },
            "auditor_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Explicit list of auditor tool names to mount, "
                    "overriding auditor_profile. submit_verdict is "
                    "force-included."
                ),
            },
            "cards_tools_config": {"type": ["object", "null"]},
            "observability_config": {"type": ["object", "null"]},
            "shutdown_timeout_s": {
                "type": "number",
                "minimum": 0,
                "description": (
                    "Seconds to wait for the async audit worker to drain "
                    "queued jobs at session shutdown. Default 600 to "
                    "accommodate slow LLM endpoints (Warpgate-fronted "
                    "providers can take ~14s/call); lower it if your "
                    "provider is fast and you want quicker teardown."
                ),
            },
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
    # Capability dependency declaration for atoms used on the PARENT
    # session: the adapter emits DiagnosticEvent / OTel spans on the
    # parent's bus, so those substrates must be present. ``system_prompt``
    # is intentionally NOT listed — it's only consumed by spawned audit
    # children (the composer mounts it by module path directly), so
    # requiring it here would lock the adapter out of scenarios that use
    # a scenario-specific prompt atom (e.g. rca's ``prompt_loader``).
    # Missing audit-child deps surface at child-session freeze, which is
    # the right scope.
    requires=(
        "observability",
        "otel_tracing",
        "operations_local",
    ),
    api_version=1,
    tier=1,
)


_DEFAULT_AUDIT_INTERVAL_TURNS = 3
_DEFAULT_EXTRACTOR_INTERVAL_TURNS = 1
_DEFAULT_RECENT_VERDICTS = _et.RECENT_VERDICTS_FOR_AUDITOR
# 600s (10min) accommodates slow / proxied LLM endpoints (e.g. Warpgate-
# fronted OpenAI-compatible servers at ~14s/call); a low default drops
# audit jobs at session teardown. Override via ``shutdown_timeout_s``.
_DEFAULT_SHUTDOWN_TIMEOUT_S = 600.0
_DEFAULT_MODE = "async"
_DEFAULT_AUDIT_SUMMARY_THRESHOLD = 30

# Canonical preamble lives in ``audit/_reminder_format.py`` so the
# replay.reminder_seed atom can build byte-identical messages without
# importing from this adapter (§11 atom-to-atom-import ban). Re-exported
# here under the long-standing private name to keep existing tests
# (test_reminder_injector) and call sites stable.
_REMINDER_PREAMBLE = _SHARED_REMINDER_PREAMBLE

# Entry-type bindings (every literal must come from entry_types.py).
_AUDIT_EVENT_ENTRY_TYPE = _et.AUDIT_EVENT
_AUDIT_EDGE_ENTRY_TYPE = _et.AUDIT_EDGE
_AUDIT_GRAPH_OP_ENTRY_TYPE = _et.AUDIT_GRAPH_OP
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

# Adapter-internal ExtensionAPI service keys. The audit-check registry has
# its own public key in ``audit/registry.py`` because external atoms need to
# resolve it; these two are read only by helpers inside this module, so they
# stay private to the adapter. Routing through ``api.set_service`` /
# ``api.get_service`` keeps the per-install state off the public
# ``ExtensionAPI`` attribute surface (§11.4.D2).
_EXTRACTOR_COMPOSE_KWARGS_SERVICE_KEY: Final[str] = (
    "llmharness._extractor_compose_kwargs"
)
_REPLAY_LOG_PATH_SERVICE_KEY: Final[str] = "llmharness._replay_log_path"


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
    """Single-pass extraction of cursor + graph + edges + phases + verdicts.

    Event-sourcing refactor (2026-05-22): the live graph is the **fold**
    of an ordered op log. This scanner accumulates ops from three
    sources in branch order:

    * :data:`AUDIT_GRAPH_OP` entries — first-class ops produced by the
      event-sourcing extractor; parsed via :func:`parse_op`.
    * Legacy :data:`AUDIT_EVENT` entries — translated to
      :class:`NodeUpsert`. Used by sessions written before the refactor;
      mixed sessions Just Work because ops are folded in branch order.
    * Legacy :data:`AUDIT_EDGE` entries — translated to
      :class:`EdgeUpsert`.

    Phases keep their own list — they're auditor-side metadata, not part
    of the graph op log.
    """
    cursor_last_turn_index = -1
    ops: list[GraphOp] = []
    phases: list[Phase] = []
    verdicts: list[dict[str, Any]] = []

    for entry in branch:
        payload = entry.payload
        if not isinstance(payload, dict):
            continue
        if entry.type == _AUDIT_GRAPH_OP_ENTRY_TYPE:
            try:
                ops.append(parse_op(payload))
            except (KeyError, ValueError, TypeError):
                continue
        elif entry.type == _AUDIT_EVENT_ENTRY_TYPE:
            try:
                ev = Event.from_dict(payload)
            except (KeyError, ValueError, TypeError):
                continue
            ops.append(
                NodeUpsert(
                    id=ev.id,
                    kind=ev.kind.value,
                    summary=ev.summary,
                    source_turns=tuple(ev.source_turns),
                )
            )
        elif entry.type == _AUDIT_EDGE_ENTRY_TYPE:
            try:
                ed = Edge.from_dict(payload)
            except (KeyError, ValueError, TypeError):
                continue
            ops.append(
                EdgeUpsert(
                    src=ed.src,
                    dst=ed.dst,
                    kind=ed.kind.value,
                    reason=ed.reason,
                    cited_entities=ed.cited_entities,
                    cited_quote=ed.cited_quote,
                    src_turns=ed.src_turns,
                    dst_turns=ed.dst_turns,
                )
            )
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

    folded = fold_graph(ops)
    graph: list[Event] = folded.nodes_list()
    edges: list[Edge] = folded.edges_list()

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

    base_prompt: str
    cards_tools_config: dict[str, Any] | None
    observability_config: dict[str, Any] | None
    summary_threshold: int
    tools: tuple[str, ...]


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
    extractor_k = int(
        config.get("extractor_interval_turns", _DEFAULT_EXTRACTOR_INTERVAL_TURNS)
    )
    if extractor_k < 1:
        extractor_k = _DEFAULT_EXTRACTOR_INTERVAL_TURNS

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
    prompt_extractor_override = (
        prompt_extractor_raw if isinstance(prompt_extractor_raw, str) else None
    )
    prompt_auditor_override = (
        prompt_auditor_raw if isinstance(prompt_auditor_raw, str) else None
    )

    extractor_prompt_name_raw = config.get("extractor_prompt")
    auditor_prompt_name_raw = config.get("auditor_prompt")
    extractor_prompt_name = (
        extractor_prompt_name_raw
        if isinstance(extractor_prompt_name_raw, str) and extractor_prompt_name_raw
        else "default"
    )
    auditor_prompt_name = (
        auditor_prompt_name_raw
        if isinstance(auditor_prompt_name_raw, str) and auditor_prompt_name_raw
        else "minimal"
    )
    extractor_base_prompt = (
        prompt_extractor_override
        if prompt_extractor_override is not None
        else load_extractor_prompt(extractor_prompt_name)
    )
    auditor_base_prompt = (
        prompt_auditor_override
        if prompt_auditor_override is not None
        else load_auditor_prompt(auditor_prompt_name)
    )

    auditor_profile_raw = config.get("auditor_profile")
    auditor_tools_raw = config.get("auditor_tools")
    auditor_tools = _resolve_auditor_tools(
        profile=auditor_profile_raw
        if isinstance(auditor_profile_raw, str)
        else None,
        tools=auditor_tools_raw if isinstance(auditor_tools_raw, list) else None,
    )

    extractor_extensions = compose_extractor_extensions(
        base_prompt=extractor_base_prompt,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
    )
    api.set_service(
        _EXTRACTOR_COMPOSE_KWARGS_SERVICE_KEY,
        {
            "base_prompt": extractor_base_prompt,
            "cards_tools_config": cards_cfg,
            "observability_config": obs_cfg,
        },
    )
    # Auditor extensions are rebuilt per firing in _drain_auditor because
    # the prompt is templated over the live event/edge graph + findings.
    auditor_settings = _AuditorSettings(
        base_prompt=auditor_base_prompt,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
        summary_threshold=summary_threshold,
        tools=auditor_tools,
    )

    extractor_provider = _parse_provider_spec(config.get("extractor_provider"))
    auditor_provider = _parse_provider_spec(config.get("auditor_provider"))

    enable_auditor = bool(config.get("enable_auditor", True))
    enable_reminders = bool(config.get("enable_reminders", True))
    enable_replay_log = bool(config.get("enable_replay_log", True))

    api.set_service(
        _REPLAY_LOG_PATH_SERVICE_KEY,
        replay_log_path(api.cwd, _audit_session_id(api)) if enable_replay_log else None,
    )

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
            auditor_due = enable_auditor and (turn_count % k) == 0
            extractor_due = (turn_count % extractor_k) == 0 or auditor_due
            extractor_ok = True
            if extractor_due:
                extractor_ok = await _drain_extractor(
                    api=api,
                    job=_ExtractorJob(messages=messages_snapshot),
                    extractor_extensions=extractor_extensions,
                    extractor_provider=extractor_provider,
                )
            if auditor_due and extractor_ok:
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
        messages_snapshot = tuple(event.messages)
        auditor_due = enable_auditor and (turn_count % k) == 0
        extractor_due = (turn_count % extractor_k) == 0 or auditor_due
        if not extractor_due and not auditor_due:
            return
        _ensure_worker()
        if extractor_due:
            queue.put_nowait(_ExtractorJob(messages=messages_snapshot))
        if auditor_due:
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
            # Route through the shared builder so the seed atom in
            # ``llmharness.replay.reminder_seed`` and this live path stay
            # byte-identical — train/inference parity is the fail-stop
            # for the prefix-replay iteration loop.
            injected.append(build_reminder_message(reminder.text))
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

    # Extend turn_texts with the source turns of every recent_graph
    # event so external_refs can be witnessed (their source side lives
    # in prior firings' windows). recent_graph carries the full prior
    # graph — letting the LLM cite any earlier event, not just a tail
    # window — so conclusions can attach back to the original evidence
    # rather than whichever events happened to be in the last slice.
    recent_graph_events = tuple(branch_state.graph)
    referenced_turns: set[int] = {
        t for e in recent_graph_events for t in e.source_turns
    }
    for t in referenced_turns:
        if t in state.turn_texts:
            continue
        if 0 <= t < len(messages):
            state.turn_texts[t] = _render_message_text(messages[t])
    state.recent_graph = recent_graph_events
    # Event-sourcing seed: the apply_* surface needs the folded
    # prior-firings view to validate cross-firing edits. Populate from
    # the same branch scan that produced ``recent_graph_events``.
    state.recent_graph_dict = {e.id: e for e in recent_graph_events}
    state.recent_edges_dict = {
        (ed.src, ed.dst, ed.kind.value): ed for ed in branch_state.edges
    }
    # Refold so the apply_* surface sees the prior graph from op zero.
    state._refold()

    # Globally-unique event ids: this firing's events must continue the
    # global sequence rather than restart at 1. Pick the smallest id
    # not yet present in the running graph (treat ids as a contiguous
    # claim — a hole is fine, but reusing a live id is not).
    state.next_event_id = (
        max((g.id for g in branch_state.graph), default=0) + 1
    )

    # Build the new-turn JSON window for the prompt + payload.
    new_turn_window = [
        s
        for s in (
            _serialize_message_for_extractor(msg, index=i)
            for i, msg in enumerate(window_messages, start=window_lo)
        )
        if s is not None
    ]

    # Enrich each recent_graph entry with the literal text of its
    # source_turns so the extractor can pick external_refs witnesses
    # without guessing. Without this the model only sees summaries and
    # cannot tell which tokens are actually present in the prior turn
    # texts the witness validator will check against.
    recent_graph_payload: list[dict[str, Any]] = []
    for e in recent_graph_events:
        entry = e.to_dict()
        entry["source_turn_texts"] = [
            state.turn_texts.get(t, "") for t in e.source_turns
        ]
        recent_graph_payload.append(entry)

    payload = {
        "next_event_id": state.next_event_id,
        "new_turns": new_turn_window,
        "recent_graph": recent_graph_payload,
    }

    # Inject state into the per-firing extensions list. The base list
    # returned by ``compose_extractor_extensions`` is shared across
    # firings; we never mutate it. The new-turn window is shipped as
    # the child's user message (see ``payload`` above), not embedded
    # into the system prompt.
    firing_extensions = bind_extractor_state(
        extractor_extensions,
        state=state,
    )

    # Only materialize the replay snapshot if the sidecar is enabled.
    # turn_texts can be large in long sessions; skip the copy otherwise.
    replay_path = _replay_log_path_for(api)
    if replay_path is not None:
        replay_compose_kwargs: dict[str, Any] | None = dict(
            api.get_service(_EXTRACTOR_COMPOSE_KWARGS_SERVICE_KEY) or {}
        )
        replay_extras: dict[str, Any] | None = {
            "turn_texts": {str(k): v for k, v in state.turn_texts.items()},
        }
    else:
        replay_compose_kwargs = None
        replay_extras = None

    raw_assistant_messages: list[dict[str, Any]] = []

    def _record(status: str, output: dict[str, Any] | None, error: str | None = None) -> None:
        _record_replay_at(
            replay_path,
            phase="extractor",
            turn_index=window_hi_inclusive,
            root_session_id=_audit_session_id(api),
            compose_kwargs=replay_compose_kwargs,
            payload=payload,
            provider=extractor_provider,
            output=output,
            status=status,
            error=error,
            extras=replay_extras,
            raw_assistant_messages=raw_assistant_messages,
        )

    try:
        terminator_called, raw_assistant_messages = await _spawn_extractor_child(
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
            "block_plan": list(output.block_plan),
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

    # Event-sourcing: persist the firing's op log as one
    # ``AUDIT_GRAPH_OP`` entry per op so any future scan can replay the
    # exact state changes this firing made. ``firing_id`` is the index
    # of this firing in the session (derived from the existing op log
    # size on the branch so we don't have to track it across firings);
    # ``op_index`` is the position within the firing.
    firing_id = sum(
        1
        for e in branch
        if isinstance(e.payload, dict) and e.type == _AUDIT_GRAPH_OP_ENTRY_TYPE
        and e.payload.get("op_index") == 0
    )
    for op_index, op in enumerate(state.pending_ops):
        payload = op.to_dict()
        payload["firing_id"] = firing_id
        payload["op_index"] = op_index
        payload["caused_by_turn_window"] = list(turn_window)
        api.session.append_entry(_AUDIT_GRAPH_OP_ENTRY_TYPE, payload)

    # Legacy emit: keep writing AUDIT_EVENT / AUDIT_EDGE entries so the
    # auditor, aggregate, replay, and web-viewer paths continue to work
    # without modification while the event-sourcing migration is rolled
    # out. The scanner translates both stripes into the same fold view.
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
    path = api.get_service(_REPLAY_LOG_PATH_SERVICE_KEY)
    return path if isinstance(path, Path) else None


def _audit_session_id(api: ExtensionAPI) -> str:
    """The id used for the sidecar filename and ``ReplayRecord.root_session_id``.

    Prefers ``api.session.get_session_id()`` (the persisted
    ``SessionManager`` header id, which is also the on-disk JSONL file
    name) so ``agent-from-reminder`` can relocate the source session.
    Falls back to ``api.root_session_id`` (the OTel trace_id assigned at
    session-construction time) when the session is in-memory /
    unpersisted — that branch keeps the existing behaviour for embedded
    SDK callers who never write a session file at all.
    """
    try:
        sid = api.session.get_session_id()
    except AttributeError:
        # Older ReadonlySession impl without get_session_id (third-party
        # SDK consumer). Fall back to the trace id.
        sid = ""
    if sid:
        return sid
    return api.root_session_id


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
    raw_assistant_messages: list[dict[str, Any]] | None = None,
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
        raw_assistant_messages=list(raw_assistant_messages or []),
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
) -> tuple[bool, list[dict[str, Any]]]:
    """Run the extractor child. Returns ``(submit_events_called, raw_blocks)``.

    ``raw_blocks`` is the chronological flattened list of serialized
    assistant content blocks (thinking + tool_call + text) captured from
    the child's message stream. It is empty when spawn / prompt fail —
    those paths raise :class:`_ExtractorSpawnError` so the caller can
    route them to the typed failure entry.
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
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        recent_n = len(payload.get("recent_graph") or [])
        next_id = payload.get("next_event_id")
        directive = (
            "Below is the firing input. Workflow:\n"
            "(1) Call submit_plan ONCE with the block_plan partitioning "
            "the new-turn window. The plan is CoT scaffolding; structural "
            "enforcement happens on the events you emit, not on the plan.\n"
            "(2) Call submit_events_batch one or more times to append "
            "events. Each batch is validated standalone — a hard reject "
            "only invalidates THAT batch, previously accepted batches "
            "stay. Set done=true on the final batch to terminate. "
            "Internal events must be true branch points (in-degree>=2 "
            "or out-degree>=2); passthrough (in=1, out=1) events are "
            "rejected on done=true.\n"
            f"(3) Start event ids at {next_id} and increment strictly — "
            "do NOT restart at 1 and do NOT reuse any id from recent_graph.\n"
            f"(4) external_refs pass: recent_graph has {recent_n} entries "
            "with source_turn_texts; for each event you emit, scan those "
            "texts for any literal token that also appears in this event's "
            "source_turns text. When you find one and the connection is "
            "causally meaningful, emit an external_refs entry whose "
            "to_recent_event_id copies that recent_graph entry's .id field. "
            "In a typical multi-turn investigation most evid events in this "
            "firing answer a hyp/act from earlier firings.\n\n"
            + payload_json
        )
        messages = await child.prompt(directive)
    except Exception as exc:
        await safe_shutdown(child)
        raise _ExtractorSpawnError(str(exc)) from exc

    await safe_shutdown(child)
    return (
        _has_tool_call(messages, SUBMIT_EVENTS_TOOL_NAME),
        _flatten_assistant_blocks(messages),
    )


def _flatten_assistant_blocks(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Flatten every AssistantMessage's content into a chronological block list.

    Used by ReplayRecord.raw_assistant_messages to persist child thinking
    + tool_call traces for downstream SFT. User / tool_result messages
    are skipped — the payload + output fields already carry them.
    """
    blocks: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for blk in content:
            serialized = _serialize_block(blk)
            if serialized is not None:
                blocks.append(serialized)
    return blocks


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
        base_prompt=auditor_settings.base_prompt,
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
        tools=auditor_settings.tools,
    )

    # Build the replay snapshot only when the sidecar is enabled — the
    # to_dict() comprehensions over events/edges/phases/findings dominate
    # per-firing CPU on large graphs.
    replay_path = _replay_log_path_for(api)
    replay_compose_kwargs: dict[str, Any] | None = None
    replay_payload: dict[str, Any] | None = None
    if replay_path is not None:
        replay_compose_kwargs = {
            "base_prompt": auditor_settings.base_prompt,
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
            "tools": list(auditor_settings.tools),
        }
        replay_payload = {
            "graph": [e.to_dict() for e in branch_state.graph],
            "recent_verdicts": list(branch_state.recent_verdicts),
            "continuation_notes_from_prior_firing": list(
                branch_state.last_continuation_notes
            ),
        }

    verdict, raw_blocks = await _run_auditor(
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
            root_session_id=_audit_session_id(api),
            compose_kwargs=replay_compose_kwargs,
            payload=replay_payload,
            provider=auditor_provider,
            output=None,
            status="no_call",
            raw_assistant_messages=raw_blocks,
        )
        return

    api.session.append_entry(_VERDICT_ENTRY_TYPE, verdict.to_dict())
    _record_replay_at(
        replay_path,
        phase="auditor",
        turn_index=turn_index,
        root_session_id=_audit_session_id(api),
        compose_kwargs=replay_compose_kwargs,
        payload=replay_payload,
        provider=auditor_provider,
        output=verdict.to_dict(),
        status="ok",
        raw_assistant_messages=raw_blocks,
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
) -> tuple[_T | None, list[dict[str, Any]]]:
    """Spawn child, drive to terminal_tool, coerce, return ``(result, raw_blocks)``.

    ``raw_blocks`` is the chronological flattened assistant content
    block list captured from the child's reply (thinking + tool_calls +
    text). Empty list when the child never produced an assistant
    message (spawn / prompt failure paths).
    """
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
        return None, []

    try:
        messages = await child.prompt(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:
        on_spawn_or_prompt_error(str(exc))
        await safe_shutdown(child)
        return None, []

    await safe_shutdown(child)

    raw_blocks = _flatten_assistant_blocks(messages)

    arguments = find_terminal_tool_arguments(messages, terminal_tool)
    if arguments is None:
        on_no_call()
        return None, raw_blocks

    try:
        return coerce(arguments), raw_blocks
    except coerce_error as exc:
        on_malformed(str(exc))
        return None, raw_blocks


async def _run_auditor(
    *,
    api: ExtensionAPI,
    auditor_extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    graph_events: list[Event],
    recent_verdicts: list[dict[str, Any]],
    continuation_notes_from_prior_firing: list[str],
) -> tuple[Verdict | None, list[dict[str, Any]]]:
    """Run one auditor firing; return ``(verdict, raw_assistant_blocks)``.

    ``raw_assistant_blocks`` is forwarded for replay-sidecar persistence
    so downstream SFT exporters can recover the auditor's thinking
    trace. The verdict-coercion branch keeps it intact even on
    malformed-output / no-call paths so the sidecar still captures what
    the LLM produced.
    """
    payload = {
        "graph": [e.to_dict() for e in graph_events],
        "recent_verdicts": list(recent_verdicts),
        "continuation_notes_from_prior_firing": list(continuation_notes_from_prior_firing),
    }
    raw, raw_blocks = await _run_phase(
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
        return None, raw_blocks
    try:
        return raw.to_verdict(), raw_blocks
    except AuditorOutputError as exc:
        _record_failure(api, _AUDIT_ERROR_ENTRY, {"reason": f"malformed: {exc}"})
        return None, raw_blocks


# Public aliases for downstream eval orchestrators that need to reconstruct
# extractor/auditor inputs offline (e.g. agentm_rca baseline-fork mode).
flatten_assistant_blocks = _flatten_assistant_blocks
serialize_full_trajectory = _serialize_full_trajectory


__all__ = [
    "MANIFEST",
    "flatten_assistant_blocks",
    "install",
    "serialize_full_trajectory",
]
