"""Adapter: AgentM bus -> two-phase cognitive audit.

See `.claude/designs/llmharness-cognitive-audit.md` for the full design.
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
from typing import Any, TypeVar

from agentm.core.abi import BeforeSendToLlmEvent, TurnEndEvent
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    ToolResultMessage,
    text_message,
)
from agentm.core.abi.session import SessionEntry
from agentm.extensions import ExtensionManifest
from agentm.harness.events import SessionShutdownEvent
from agentm.harness.extension import ExtensionAPI
from agentm.harness.session_config import AgentSessionConfig

from ..audit import entry_types as _et
from ..audit.auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
    compose_auditor_extensions,
)
from ..audit.extractor import (
    SUBMIT_EVENTS_TOOL_NAME,
    ExtractorOutputError,
    RawExtractorOutput,
    compose_extractor_extensions,
)
from ..schema import Event, Reminder, Verdict

_logger = logging.getLogger(__name__)

MANIFEST = ExtensionManifest(
    name="agentm",
    description=(
        "Two-phase cognitive-audit adapter: per-turn extractor (Phase 1) and "
        "every-k-turns graph auditor (Phase 2). ``mode='async'`` (default) "
        "runs audit on a background worker so the main agent loop is never "
        "blocked; verdicts arrive as synthetic user messages on the next "
        "before_send_to_llm and the session_shutdown handler drains the "
        "queue. ``mode='sync'`` runs audit inline at turn_end — slower but "
        "guarantees every turn has paired audit data, suitable for dataset "
        "collection / offline distillation."
    ),
    registers=(
        "event:turn_end",
        "event:before_send_to_llm",
        "event:session_shutdown",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["async", "sync"]},
            "audit_interval_turns": {"type": "integer", "minimum": 1},
            "prompt_override_extractor": {"type": "string"},
            "prompt_override_auditor": {"type": "string"},
            "cards_tools_config": {"type": ["object", "null"]},
            "observability_config": {"type": ["object", "null"]},
            "shutdown_timeout_s": {"type": "number", "minimum": 0},
            # Per-role provider override. ``module`` is the dotted path to
            # a StreamFn module (e.g. ``agentm.llm.anthropic``); ``config``
            # is forwarded as the provider's kwargs (e.g. ``{"model":
            # "claude-haiku-4-5"}``). Omit to inherit the parent session's
            # provider — the v0 behaviour. Use this to point extractor /
            # auditor at a smaller / cheaper model so the main agent and
            # the audit pipeline don't share a rate-limit pool.
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
        },
        "additionalProperties": False,
    },
    affects=(
        "event:turn_end",
        "event:before_send_to_llm",
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

# The reminder body the model sees. Explicitly framed as a meta-injection so
# the model treats it as out-of-band advisory rather than as a fresh user
# instruction that supersedes the current task.
_REMINDER_PREAMBLE = (
    "[harness advisory — meta-injection from cognitive audit, not from the "
    "human user]\n"
)

_AUDIT_EVENT_ENTRY_TYPE = _et.AUDIT_EVENT
_VERDICT_ENTRY_TYPE = _et.VERDICT
_EXTRACTOR_CURSOR_ENTRY_TYPE = _et.EXTRACTOR_CURSOR
_REMINDER_DELIVERED_ENTRY_TYPE = _et.REMINDER_DELIVERED

_EXTRACTOR_NO_CALL_ENTRY = _et.EXTRACTOR_NO_CALL
_EXTRACTOR_ERROR_ENTRY = _et.EXTRACTOR_ERROR
_EXTRACTOR_EMPTY_ENTRY = _et.EXTRACTOR_EMPTY
_AUDIT_NO_CALL_ENTRY = _et.AUDIT_NO_CALL
_AUDIT_ERROR_ENTRY = _et.AUDIT_ERROR


# --- branch state -----------------------------------------------------------


@dataclass(frozen=True)
class _BranchState:
    """Snapshot of audit-relevant entries pulled from a single branch walk."""

    cursor_last_turn_index: int
    graph: list[Event]
    recent_verdicts: list[dict[str, Any]]


def _scan_branch(branch: list[SessionEntry], *, recent_verdicts_n: int) -> _BranchState:
    """Single-pass extraction of cursor + graph + recent verdicts."""
    cursor_last_turn_index = -1
    graph: list[Event] = []
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
        elif entry.type == _VERDICT_ENTRY_TYPE:
            verdicts.append(payload)
        elif entry.type == _EXTRACTOR_CURSOR_ENTRY_TYPE:
            raw = payload.get("last_turn_index")
            if isinstance(raw, int) and not isinstance(raw, bool):
                cursor_last_turn_index = raw

    return _BranchState(
        cursor_last_turn_index=cursor_last_turn_index,
        graph=graph,
        recent_verdicts=verdicts[-recent_verdicts_n:] if recent_verdicts_n > 0 else [],
    )


# --- failure recording ------------------------------------------------------


def _record_failure(
    api: ExtensionAPI, entry_type: str, payload: dict[str, Any]
) -> None:
    """Single chokepoint for typed failure entries. Append-only; never raises."""
    api.session.append_entry(entry_type, payload)


def _window_is_non_trivial(messages_slice: list[AgentMessage]) -> bool:
    """True iff the slice contains any AssistantMessage or ToolResultMessage.

    A pure user-only window does NOT count as non-trivial — extractor is not
    expected to produce events from user messages alone, so an empty events
    array on such a window is normal, not a failure (design §4).
    """
    return any(
        isinstance(msg, (AssistantMessage, ToolResultMessage)) for msg in messages_slice
    )


# --- install ----------------------------------------------------------------


@dataclass(frozen=True)
class _ExtractorJob:
    """Snapshot taken synchronously at TurnEndEvent; consumed by worker."""

    messages: tuple[AgentMessage, ...]


@dataclass(frozen=True)
class _AuditorJob:
    """Marker that an auditor pass should run after the most recent extractor.

    The worker reads the up-to-date branch state when this job fires, so it
    sees every extractor entry already appended by prior jobs in this queue.
    """


@dataclass(frozen=True)
class _ShutdownJob:
    """Worker-stop sentinel; consumed only by ``_drain_queue``."""


_Job = _ExtractorJob | _AuditorJob | _ShutdownJob


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    mode_raw = config.get("mode", _DEFAULT_MODE)
    mode = mode_raw if mode_raw in ("async", "sync") else _DEFAULT_MODE

    k = int(config.get("audit_interval_turns", _DEFAULT_AUDIT_INTERVAL_TURNS))
    if k < 1:
        k = _DEFAULT_AUDIT_INTERVAL_TURNS

    shutdown_timeout = float(
        config.get("shutdown_timeout_s", _DEFAULT_SHUTDOWN_TIMEOUT_S)
    )
    if shutdown_timeout < 0:
        shutdown_timeout = _DEFAULT_SHUTDOWN_TIMEOUT_S

    cards_cfg_raw = config.get("cards_tools_config", {})
    obs_cfg_raw = config.get("observability_config", {})
    cards_cfg = cards_cfg_raw if isinstance(cards_cfg_raw, dict) else None
    obs_cfg = obs_cfg_raw if isinstance(obs_cfg_raw, dict) else None

    prompt_extractor_raw = config.get("prompt_override_extractor")
    prompt_auditor_raw = config.get("prompt_override_auditor")
    prompt_extractor = (
        prompt_extractor_raw if isinstance(prompt_extractor_raw, str) else None
    )
    prompt_auditor = (
        prompt_auditor_raw if isinstance(prompt_auditor_raw, str) else None
    )

    extractor_extensions = compose_extractor_extensions(
        prompt_override=prompt_extractor,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
    )
    auditor_extensions = compose_auditor_extensions(
        prompt_override=prompt_auditor,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
    )

    extractor_provider = _parse_provider_spec(config.get("extractor_provider"))
    auditor_provider = _parse_provider_spec(config.get("auditor_provider"))

    pending_reminders: list[Reminder] = []
    turn_count = 0

    if mode == "sync":
        async def _on_turn_end_sync(event: TurnEndEvent) -> None:
            nonlocal turn_count
            turn_count += 1
            await _drain_extractor(
                api=api,
                job=_ExtractorJob(messages=tuple(event.messages)),
                extractor_extensions=extractor_extensions,
                extractor_provider=extractor_provider,
            )
            if (turn_count % k) == 0:
                await _drain_auditor(
                    api=api,
                    auditor_extensions=auditor_extensions,
                    auditor_provider=auditor_provider,
                    pending_reminders=pending_reminders,
                )

        api.on(TurnEndEvent.CHANNEL, _on_turn_end_sync)
        api.on(
            BeforeSendToLlmEvent.CHANNEL,
            _make_reminder_injector(api, pending_reminders),
        )
        return

    # Async path: queue + background worker.
    queue: asyncio.Queue[_Job] = asyncio.Queue()
    worker_task: asyncio.Task[None] | None = None

    def _ensure_worker() -> None:
        # Lazy-spawn the worker on first turn_end so we capture the active
        # event loop. Doing it at install time would bind to whichever loop
        # the loader ran on, which may differ from the agent's loop.
        nonlocal worker_task
        if worker_task is None or worker_task.done():
            worker_task = asyncio.create_task(
                _drain_queue(
                    api=api,
                    queue=queue,
                    pending_reminders=pending_reminders,
                    extractor_extensions=extractor_extensions,
                    auditor_extensions=auditor_extensions,
                    extractor_provider=extractor_provider,
                    auditor_provider=auditor_provider,
                ),
                name="llmharness-audit-worker",
            )

    def _on_turn_end(event: TurnEndEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        _ensure_worker()
        queue.put_nowait(_ExtractorJob(messages=tuple(event.messages)))
        if (turn_count % k) == 0:
            queue.put_nowait(_AuditorJob())

    _on_before_send_to_llm = _make_reminder_injector(api, pending_reminders)

    async def _on_session_shutdown(_event: SessionShutdownEvent) -> None:
        # Never raise: session shutdown must complete even if audit drain hangs.
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
    api.on(BeforeSendToLlmEvent.CHANNEL, _on_before_send_to_llm)
    api.on(SessionShutdownEvent.CHANNEL, _on_session_shutdown)


def _parse_provider_spec(
    raw: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Coerce ``{"module": "...", "config": {...}}`` to ``AgentSessionConfig.provider``.

    When ``module`` resolves to an entry in AgentM's provider registry, route
    through ``DEFAULT_PROVIDER_REGISTRY.build`` so the child picks up the same
    env-derived enrichment (``base_url``, warpgate ticket, ``verify_ssl``)
    that the parent CLI honours. For dotted-path module names that aren't
    registered, fall back to the raw ``(module, config)`` tuple — the v0
    contract for hand-written StreamFn modules.

    Returns ``None`` if the spec is missing/empty so the child inherits the
    parent session's provider.
    """
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
) -> Callable[[BeforeSendToLlmEvent], None]:
    """Build the ``before_send_to_llm`` handler shared by both modes.

    Drains every reminder ready as of this turn boundary. Multiple may
    accumulate if the worker (async mode) stayed busy across several turns;
    the model sees them as a sequence of advisories before its next call.
    """

    def _inject(event: BeforeSendToLlmEvent) -> None:
        while pending_reminders:
            reminder = pending_reminders.pop(0)
            event.messages.append(
                text_message(
                    _REMINDER_PREAMBLE + reminder.text, timestamp=time.time()
                )
            )
            try:
                api.session.append_entry(
                    _REMINDER_DELIVERED_ENTRY_TYPE,
                    {"type": reminder.type.value, "text": reminder.text},
                )
            except Exception:
                _logger.exception("failed to persist reminder_delivered entry")

    return _inject


async def _drain_queue(
    *,
    api: ExtensionAPI,
    queue: asyncio.Queue[_Job],
    pending_reminders: list[Reminder],
    extractor_extensions: list[tuple[str, dict[str, Any]]],
    auditor_extensions: list[tuple[str, dict[str, Any]]],
    extractor_provider: tuple[str, dict[str, Any]] | None,
    auditor_provider: tuple[str, dict[str, Any]] | None,
) -> None:
    """Serial worker. Owns all session-mutating audit writes.

    Single-consumer keeps invariants simple: cursor advances monotonically,
    auditor jobs see every extractor entry queued before them, and there is
    never a race between two extractor passes over the same window.
    """
    while True:
        try:
            job = await queue.get()
        except asyncio.CancelledError:
            raise
        try:
            if isinstance(job, _ShutdownJob):
                return
            if isinstance(job, _ExtractorJob):
                await _drain_extractor(
                    api=api,
                    job=job,
                    extractor_extensions=extractor_extensions,
                    extractor_provider=extractor_provider,
                )
            elif isinstance(job, _AuditorJob):
                await _drain_auditor(
                    api=api,
                    auditor_extensions=auditor_extensions,
                    auditor_provider=auditor_provider,
                    pending_reminders=pending_reminders,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Audit failures must never poison the main agent. Worst-case
            # we lose one job's persistence; cursor is only advanced on
            # success so the next firing re-covers the same window.
            _logger.exception("llmharness audit worker job failed")
        finally:
            queue.task_done()


async def _drain_extractor(
    *,
    api: ExtensionAPI,
    job: _ExtractorJob,
    extractor_extensions: list[tuple[str, dict[str, Any]]],
    extractor_provider: tuple[str, dict[str, Any]] | None,
) -> None:
    branch = api.session.get_branch()
    state = _scan_branch(branch, recent_verdicts_n=_DEFAULT_RECENT_VERDICTS)
    messages = list(job.messages)

    new_events = await _run_extractor(
        api=api,
        extractor_extensions=extractor_extensions,
        provider=extractor_provider,
        messages=messages,
        cursor_last_turn_index=state.cursor_last_turn_index,
        recent_graph_slice=state.graph[-_RECENT_GRAPH_SLICE_FOR_EXTRACTOR:],
        next_event_id=_next_event_id(state.graph),
    )
    if new_events is None:
        return  # typed failure already recorded; cursor stays put

    for ev in new_events:
        api.session.append_entry(_AUDIT_EVENT_ENTRY_TYPE, ev.to_dict())

    api.session.append_entry(
        _EXTRACTOR_CURSOR_ENTRY_TYPE,
        {
            "last_turn_index": len(messages) - 1,
            "extraction_run_id": uuid.uuid4().hex,
        },
    )


async def _drain_auditor(
    *,
    api: ExtensionAPI,
    auditor_extensions: list[tuple[str, dict[str, Any]]],
    auditor_provider: tuple[str, dict[str, Any]] | None,
    pending_reminders: list[Reminder],
) -> None:
    branch = api.session.get_branch()
    state = _scan_branch(branch, recent_verdicts_n=_DEFAULT_RECENT_VERDICTS)

    verdict = await _run_auditor(
        api=api,
        auditor_extensions=auditor_extensions,
        provider=auditor_provider,
        graph_events=state.graph,
        recent_verdicts=state.recent_verdicts,
    )
    if verdict is None:
        return

    api.session.append_entry(_VERDICT_ENTRY_TYPE, verdict.to_dict())
    if verdict.drift and verdict.reminder and verdict.type is not None:
        pending_reminders.append(
            Reminder(type=verdict.type, text=verdict.reminder)
        )


def _next_event_id(prior_events: list[Event]) -> int:
    return max((e.id for e in prior_events), default=-1) + 1


# --- trajectory slicing -----------------------------------------------------


def _slice_new_turns(
    messages: list[AgentMessage], cursor_last_turn_index: int
) -> list[dict[str, Any]]:
    """Slice ``messages[cursor + 1 :]`` into a payload the extractor can consume.

    Keeps thinking blocks; tool-result content stays structured (list of inner
    blocks rather than a flattened string) so the extractor can distinguish
    error vs success and inspect tool extras (design §7.4).
    """
    start = max(cursor_last_turn_index + 1, 0)
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages[start:], start=start):
        serialized = _serialize_message_for_extractor(msg, index=i)
        if serialized is not None:
            out.append(serialized)
    return out


def _serialize_message_for_extractor(
    msg: AgentMessage, *, index: int
) -> dict[str, Any] | None:
    """Best-effort dict view of one message.

    Walks blocks via duck-typing on public attrs so the serializer survives
    SDK additions (new content-block kinds) without code changes here.
    Unknown blocks fall through to a generic ``{type, repr}`` form rather than
    being silently dropped — visibility over precision.
    """
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
        return {"type": block_type if isinstance(block_type, str) and block_type else "text", "text": text}

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
    """Spawn child, drive to terminal_tool, coerce, return result.

    Failure paths invoke the supplied callbacks (which write typed failure
    entries) and return None. The driver in ``_on_turn_end`` short-circuits
    on None without poisoning cursor / graph / pending state.
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
        return None

    try:
        messages = await child.prompt(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:
        on_spawn_or_prompt_error(str(exc))
        await _safe_shutdown(child)
        return None

    await _safe_shutdown(child)

    arguments = _find_terminal_tool_arguments(messages, terminal_tool)
    if arguments is None:
        on_no_call()
        return None

    try:
        return coerce(arguments)
    except coerce_error as exc:
        on_malformed(str(exc))
        return None


async def _run_extractor(
    *,
    api: ExtensionAPI,
    extractor_extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    messages: list[AgentMessage],
    cursor_last_turn_index: int,
    recent_graph_slice: list[Event],
    next_event_id: int,
) -> list[Event] | None:
    new_turn_window = _slice_new_turns(messages, cursor_last_turn_index)
    if not new_turn_window:
        return []  # nothing to extract; clean no-op

    window_lo = max(cursor_last_turn_index + 1, 0)
    window_hi = len(messages) - 1
    window = [window_lo, window_hi]
    raw_window_slice = messages[window_lo:]

    payload = {
        "new_turns": new_turn_window,
        "recent_graph": [e.to_dict() for e in recent_graph_slice],
    }

    parsed = await _run_phase(
        api=api,
        extensions=extractor_extensions,
        provider=provider,
        purpose="cognitive_audit_extractor",
        payload=payload,
        terminal_tool=SUBMIT_EVENTS_TOOL_NAME,
        coerce=RawExtractorOutput.from_dict,
        coerce_error=ExtractorOutputError,
        on_spawn_or_prompt_error=lambda reason: _record_failure(
            api, _EXTRACTOR_ERROR_ENTRY, {"reason": reason, "turn_window": window}
        ),
        on_no_call=lambda: _record_failure(
            api,
            _EXTRACTOR_NO_CALL_ENTRY,
            {
                "reason": f"child returned without calling {SUBMIT_EVENTS_TOOL_NAME}",
                "turn_window": window,
            },
        ),
        on_malformed=lambda reason: _record_failure(
            api,
            _EXTRACTOR_ERROR_ENTRY,
            {"reason": f"malformed: {reason}", "turn_window": window},
        ),
    )
    if parsed is None:
        return None

    events = parsed.to_events(next_id=next_event_id)
    if not events and _window_is_non_trivial(raw_window_slice):
        _record_failure(api, _EXTRACTOR_EMPTY_ENTRY, {"turn_window": window})
        return None
    return events


async def _run_auditor(
    *,
    api: ExtensionAPI,
    auditor_extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    graph_events: list[Event],
    recent_verdicts: list[dict[str, Any]],
) -> Verdict | None:
    payload = {
        "graph": [e.to_dict() for e in graph_events],
        "recent_verdicts": list(recent_verdicts),
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


def _find_terminal_tool_arguments(
    messages: list[AgentMessage], tool_name: str
) -> dict[str, Any] | None:
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


async def _safe_shutdown(child: Any) -> None:
    # Best-effort: audit pipeline must not crash the parent; child may already
    # be torn down by parent shutdown in some race conditions.
    try:
        await child.shutdown()
    except Exception:
        return


__all__ = [
    "MANIFEST",
    "install",
]
