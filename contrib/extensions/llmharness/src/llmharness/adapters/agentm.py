"""Adapter: AgentM bus -> two-phase cognitive audit.

See `.claude/designs/llmharness-cognitive-audit.md` for the full design.

Phase 1 (extractor) on the configured TurnEndEvent cadence:
* Slice the trajectory window since the last cursor.
* Build a per-firing :class:`ExtractionState`, populate ``turn_texts``
  with rendered turn content (used by the witness pipeline).
* Spawn an extractor child whose extensions list carries the state via
  the ``state`` config knob. The child incrementally maintains the graph
  with ``upsert_node`` / ``upsert_edge`` / ``delete_node`` /
  ``delete_edge`` / ``reset_extraction`` (each edit validated for witness
  + id rules on the spot) and ends the firing with
  ``finalize_extraction``.
* After the child loop terminates, persist the accepted graph as
  ``audit_graph_op`` entries plus an ``extractor_cursor`` marking the
  window consumed.
* Failure / edge-case entries: ``extractor_no_call`` (terminator never
  called), ``extractor_empty`` (terminator called but the non-trivial
  window produced no graph), ``extractor_partial`` (some ops rejected),
  ``extractor_error`` (spawn / prompt / coercion crash). Cursor advances
  ONLY on success or partial.

Phase 2 (auditor) every k turns:
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

P1 refactor (2026-05-23). All of cadence / windowing / cumulative-state
threading / payload composition / sidecar emission has moved to
:class:`llmharness.audit.runner.HarnessRunner`. The adapter constructs
exactly one runner per install (seeded by
:meth:`CumulativeAuditState.hydrate_from_session_log`) and routes
``TurnEndEvent`` through ``runner.on_trajectory_progress``. The async
worker now drains :class:`_RunnerStepJob` jobs; the body of the legacy
``_drain_extractor`` / ``_drain_auditor`` / ``_spawn_extractor_child`` /
``_run_auditor`` lives inside the runner and
:mod:`llmharness.audit.seams.live` (:class:`LiveChildRunner` +
:class:`LiveOpSink`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from agentm.core.abi import (
    DecideTurnActionEvent,
    Inject,
    LoopAction,
    Stop,
    TurnEndEvent,
)
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.messages import AgentMessage, ToolCallBlock
from agentm.extensions import ExtensionManifest

from ..audit import entry_types as _et
from ..audit.auditor.profiles import resolve_tools as _resolve_auditor_tools
from ..audit.auditor.prompt import load_auditor_prompt
from ..audit.extractor import compose_extractor_extensions
from ..audit.extractor.prompt import load_extractor_prompt
from ..audit.registry import SERVICE_KEY as AUDIT_REGISTRY_SERVICE_KEY
from ..audit.registry import AuditCheckRegistry
from ..audit.runner import (
    AuditorSettings,
    CumulativeAuditState,
    ExtractorSettings,
    HarnessRunner,
    SidecarWriter,
    _flatten_assistant_blocks,
    _serialize_full_trajectory,
)
from ..audit.seams.live import LiveChildRunner, LiveOpSink
from ..audit.toolkit.reminder_format import REMINDER_PREAMBLE as _SHARED_REMINDER_PREAMBLE
from ..audit.toolkit.reminder_format import build_reminder_message
from ..audit.triggers import SERVICE_KEY as TRIGGER_SERVICE_KEY
from ..audit.triggers import TriggerRegistry
from ..replay.record import audit_session_id, replay_log_path
from ..schema import Reminder

_logger = logging.getLogger(__name__)

MANIFEST = ExtensionManifest(
    name="agentm",
    description=(
        "Two-phase cognitive-audit adapter: per-turn extractor (Phase 1) "
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
            "extractor_tool_call_budget": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": (
                    "Optional max tool-call budget for each extractor child "
                    "firing. When set, the child mounts loop_budget and "
                    "turn_reminder with this value, and the extractor "
                    "directive tells the model to reserve one call for "
                    "finalize_extraction."
                ),
            },
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
                    "``<cwd>/.agentm/audit_replay/<session_id>.jsonl`` "
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
        "operations_local",
    ),
    api_version=1,
    tier=1,
)


_DEFAULT_AUDIT_INTERVAL_TURNS = 3
_DEFAULT_EXTRACTOR_INTERVAL_TURNS = 1
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

# Entry-type binding (the literal must come from entry_types.py). Failure /
# op / cursor / verdict persistence reaches ``entry_types`` directly through
# the sinks in ``audit/seams/``; only the reminder-delivered marker is written
# from this module.
_REMINDER_DELIVERED_ENTRY_TYPE = _et.REMINDER_DELIVERED

# Adapter-internal ExtensionAPI service key. The audit-check registry has its
# own public key in ``audit/registry.py`` because external atoms need to
# resolve it; this one carries the per-session replay-sidecar path so external
# replay orchestrators can locate the log.
_REPLAY_LOG_PATH_SERVICE_KEY: Final[str] = "llmharness._replay_log_path"


# --- jobs -------------------------------------------------------------------


@dataclass(frozen=True)
class _RunnerStepJob:
    """Per-TurnEndEvent snapshot. Worker hands it to ``runner.on_trajectory_progress``."""

    messages: tuple[AgentMessage, ...]
    turn_count: int
    tool_names_called: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _ShutdownJob:
    """Worker-stop sentinel; consumed only by ``_drain_queue``."""


_Job = _RunnerStepJob | _ShutdownJob


def _extract_tool_names(event: TurnEndEvent) -> frozenset[str]:
    """Return all tool names from the turn's AssistantMessage."""
    return frozenset(
        block.name
        for block in event.message.content
        if isinstance(block, ToolCallBlock)
    )


# --- install ----------------------------------------------------------------


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    mode_raw = config.get("mode", _DEFAULT_MODE)
    mode = mode_raw if mode_raw in ("async", "sync") else _DEFAULT_MODE

    k = int(config.get("audit_interval_turns", _DEFAULT_AUDIT_INTERVAL_TURNS))
    if k < 1:
        k = _DEFAULT_AUDIT_INTERVAL_TURNS
    extractor_k = int(config.get("extractor_interval_turns", _DEFAULT_EXTRACTOR_INTERVAL_TURNS))
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

    obs_cfg_raw = config.get("observability_config", {})
    obs_cfg = obs_cfg_raw if isinstance(obs_cfg_raw, dict) else None

    prompt_extractor_raw = config.get("prompt_override_extractor")
    prompt_auditor_raw = config.get("prompt_override_auditor")
    prompt_extractor_override = (
        prompt_extractor_raw if isinstance(prompt_extractor_raw, str) else None
    )
    prompt_auditor_override = prompt_auditor_raw if isinstance(prompt_auditor_raw, str) else None

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
        profile=auditor_profile_raw if isinstance(auditor_profile_raw, str) else None,
        tools=auditor_tools_raw if isinstance(auditor_tools_raw, list) else None,
    )

    extractor_extensions = compose_extractor_extensions(
        base_prompt=extractor_base_prompt,
        observability_config=obs_cfg,
        tool_call_budget=config.get("extractor_tool_call_budget"),
    )
    extractor_compose_kwargs = {
        "base_prompt": extractor_base_prompt,
        "observability_config": obs_cfg,
        "tool_call_budget": config.get("extractor_tool_call_budget"),
    }
    extractor_settings = ExtractorSettings(
        extensions=extractor_extensions,
        compose_kwargs=extractor_compose_kwargs,
    )
    auditor_settings = AuditorSettings(
        base_prompt=auditor_base_prompt,
        observability_config=obs_cfg,
        summary_threshold=summary_threshold,
        tools=auditor_tools,
    )

    extractor_provider = _parse_provider_spec(config.get("extractor_provider"))
    auditor_provider = _parse_provider_spec(config.get("auditor_provider"))

    enable_auditor = bool(config.get("enable_auditor", True))
    enable_reminders = bool(config.get("enable_reminders", True))
    enable_replay_log = bool(config.get("enable_replay_log", True))

    sidecar_path = replay_log_path(api.cwd, audit_session_id(api)) if enable_replay_log else None
    api.set_service(_REPLAY_LOG_PATH_SERVICE_KEY, sidecar_path)

    # Publish the audit-check registry on the parent session. Atoms in
    # later commits (reference checks etc.) call
    # ``api.get_service(AUDIT_REGISTRY_SERVICE_KEY).register_check(...)``
    # from their own ``install``. Best-effort: a duplicate registration
    # (e.g. installing this adapter twice) is ignored.
    with contextlib.suppress(KeyError):
        api.set_service(AUDIT_REGISTRY_SERVICE_KEY, AuditCheckRegistry())

    # Publish the trigger registry for pluggable audit cadence. Trigger
    # atoms (e.g. trigger_cadence, trigger_on_submission) call
    # ``api.get_service(TRIGGER_SERVICE_KEY).register_trigger(...)``
    # from their own ``install``.
    trigger_registry = TriggerRegistry()
    with contextlib.suppress(KeyError):
        api.set_service(TRIGGER_SERVICE_KEY, trigger_registry)

    # Construct the single runner for this install. Hydrate the
    # cumulative state once from the existing session log so process
    # restarts pick up where they left off; thereafter the in-memory
    # state is authoritative (no per-firing re-reads).
    cumulative = CumulativeAuditState.hydrate_from_session_log(api.session.get_branch())
    # The trigger_registry is passed to the runner unconditionally; the
    # runner checks whether any triggers have been registered at
    # evaluation time. When no trigger atoms are registered (backward
    # compat), the runner falls through to the legacy hardcoded cadence.
    runner = HarnessRunner(
        cumulative=cumulative,
        child=LiveChildRunner(api),
        sink=LiveOpSink(api),
        sidecar=SidecarWriter(sidecar_path) if sidecar_path is not None else None,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=extractor_k,
        audit_interval=k,
        enable_auditor=enable_auditor,
        session_id=audit_session_id(api),
        trace_id=api.root_session_id,
        provider_extractor=extractor_provider,
        provider_auditor=auditor_provider,
        audit_registry=_resolve_registry(api),
        trigger_registry=trigger_registry,
    )

    pending_reminders: list[Reminder] = []
    turn_count = 0

    def _drain_step_result(reminder: Reminder | None) -> None:
        if reminder is not None:
            pending_reminders.append(reminder)

    if mode == "sync":

        async def _on_turn_end_sync(event: TurnEndEvent) -> None:
            nonlocal turn_count
            turn_count += 1
            # Re-resolve the registry per firing — atoms in the adapter
            # chain may install themselves after this adapter's install
            # but before the first turn, so a None at install time
            # doesn't mean "no checks ever".
            runner.set_audit_registry(_resolve_registry(api))
            step = await runner.on_trajectory_progress(
                list(event.messages),
                turn_count=turn_count,
                tool_names_called=_extract_tool_names(event),
            )
            _drain_step_result(step.surfaced_reminder)

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
                    runner=runner,
                    pending_reminders=pending_reminders,
                ),
                name="llmharness-audit-worker",
            )

    def _on_turn_end(event: TurnEndEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        # When trigger atoms are registered, always enqueue: the runner
        # evaluates the trigger registry internally and the pre-check
        # cannot replicate arbitrary trigger logic without building a
        # full TriggerContext. When no trigger atoms are registered, fall
        # back to the legacy cadence gate so no-op turns don't wake the
        # worker.
        if trigger_registry.registered_triggers():
            enqueue = True
        else:
            auditor_due = enable_auditor and (turn_count % k) == 0
            extractor_due = (turn_count % extractor_k) == 0 or auditor_due
            enqueue = extractor_due or auditor_due
        if not enqueue:
            return
        _ensure_worker()
        queue.put_nowait(
            _RunnerStepJob(
                messages=tuple(event.messages),
                turn_count=turn_count,
                tool_names_called=_extract_tool_names(event),
            )
        )

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


def _resolve_registry(api: ExtensionAPI) -> AuditCheckRegistry | None:
    """Return the live audit-check registry, or ``None`` if absent."""
    try:
        registry = api.get_service(AUDIT_REGISTRY_SERVICE_KEY)
    except Exception:  # pragma: no cover - defensive
        return None
    return registry if isinstance(registry, AuditCheckRegistry) else None


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
    runner: HarnessRunner,
    pending_reminders: list[Reminder],
) -> None:
    """Serial worker. Owns all session-mutating audit writes via the runner."""
    while True:
        try:
            job = await queue.get()
        except asyncio.CancelledError:
            raise
        try:
            if isinstance(job, _ShutdownJob):
                return
            if isinstance(job, _RunnerStepJob):
                # Re-resolve the registry each step — see comment in
                # the sync handler.
                runner.set_audit_registry(_resolve_registry(api))
                step = await runner.on_trajectory_progress(
                    list(job.messages),
                    turn_count=job.turn_count,
                    tool_names_called=job.tool_names_called,
                )
                if step.surfaced_reminder is not None:
                    pending_reminders.append(step.surfaced_reminder)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.exception("llmharness audit worker job failed")
        finally:
            queue.task_done()


# Public re-exports of the trajectory-serialization helpers for external eval
# orchestrators that reconstruct extractor/auditor inputs offline (rca-autorl).
# Not used in-tree; kept stable as package API.
flatten_assistant_blocks = _flatten_assistant_blocks
serialize_full_trajectory = _serialize_full_trajectory


__all__ = [
    "MANIFEST",
    "flatten_assistant_blocks",
    "install",
    "serialize_full_trajectory",
]
