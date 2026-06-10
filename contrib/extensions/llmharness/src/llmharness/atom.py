"""§11 atom: two-phase cognitive audit (extractor + auditor).

Hooks into an AgentM session via turn_end / decide_turn_action events.
Uses primitives to build inputs, spawns agent children via the SDK's
scenario-based interface, processes outputs, and injects reminders.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
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
from agentm.core.abi.messages import AgentMessage, UserMessage, text_message
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest

from . import entry_types as _et
from .agents import auditor_scenario, extractor_scenario
from .agents.auditor.profiles import resolve_tools as _resolve_auditor_tools
from .agents.auditor.prompt import load_auditor_prompt
from .agents.auditor.submit_verdict import SUBMIT_VERDICT_TOOL_NAME
from .agents.extractor.extractor_tools import FINALIZE_EXTRACTION_TOOL_NAME
from .agents.extractor.prompt import load_extractor_prompt
from .primitives import (
    AuditorSettings,
    CumulativeAuditState,
    build_auditor_input,
    build_extractor_input,
    process_auditor_output,
    process_extractor_output,
    serialize_full_trajectory,
)
from .registry import SERVICE_KEY as AUDIT_REGISTRY_SERVICE_KEY
from .registry import AuditCheckRegistry, CheckContext
from .schema import Reminder
from .triggers import SERVICE_KEY as TRIGGER_SERVICE_KEY
from .triggers import TriggerContext, TriggerRegistry, tool_names_from_message

_log = logging.getLogger(__name__)

REMINDER_PREAMBLE: Final = "[system reminder — automated review of your investigation so far]\n"

MANIFEST = ExtensionManifest(
    name="llmharness",
    description="Two-phase cognitive-audit: per-turn extractor + every-k-turns auditor.",
    registers=("event:turn_end", "event:decide_turn_action", "event:session_shutdown"),
    config_schema={
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["async", "sync"]},
            "extractor_interval_turns": {"type": "integer", "minimum": 1},
            "audit_interval_turns": {"type": "integer", "minimum": 1},
            "audit_summary_threshold": {"type": "integer", "minimum": 0},
            "extractor_tool_call_budget": {"type": ["integer", "null"], "minimum": 1},
            "prompt_override_extractor": {"type": "string"},
            "prompt_override_auditor": {"type": "string"},
            "extractor_prompt": {"type": "string"},
            "auditor_prompt": {"type": "string"},
            "auditor_profile": {"type": "string"},
            "auditor_tools": {"type": "array", "items": {"type": "string"}},
            "shutdown_timeout_s": {"type": "number", "minimum": 0},
            "extractor_provider": {
                "type": ["object", "null"],
                "properties": {"module": {"type": "string"}, "config": {"type": "object"}},
                "required": ["module"],
                "additionalProperties": False,
            },
            "auditor_provider": {
                "type": ["object", "null"],
                "properties": {"module": {"type": "string"}, "config": {"type": "object"}},
                "required": ["module"],
                "additionalProperties": False,
            },
            "enable_auditor": {"type": "boolean"},
            "enable_reminders": {"type": "boolean"},
        },
        "additionalProperties": False,
    },
    requires=("observability", "operations"),
    api_version=1,
    tier=1,
)


# ---------------------------------------------------------------------------
# Child session helper
# ---------------------------------------------------------------------------


async def _run_child(
    api: ExtensionAPI,
    *,
    scenario: str,
    prompt: str,
    purpose: str,
    atom_config_overrides: dict[str, dict[str, Any]] | None = None,
    extra_extensions: list[tuple[str, dict[str, Any]]] | None = None,
    provider: tuple[str, dict[str, Any]] | None = None,
    terminal_tool: str | None = None,
) -> _ChildResult:
    config = AgentSessionConfig(
        cwd=api.cwd,
        provider=provider,
        scenario=scenario,
        extra_extensions=extra_extensions or [],
        atom_config_overrides=atom_config_overrides or {},
        purpose=purpose,
    )
    t0 = time.monotonic()
    try:
        child = await api.spawn_child_session(config)
    except Exception as exc:
        return _ChildResult(error=str(exc))

    try:
        messages = await child.prompt(prompt)
    except Exception as exc:
        await _safe_shutdown(child)
        return _ChildResult(error=str(exc))

    if terminal_tool is not None:
        from .child_collect import nudge_until_tool_call, terminal_tool_arguments

        messages = await nudge_until_tool_call(child.prompt, messages, terminal_tool)
        await _safe_shutdown(child)
        args = terminal_tool_arguments(messages, terminal_tool)
        return _ChildResult(
            terminal_called=args is not None,
            terminal_args=args,
            messages=messages,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    await _safe_shutdown(child)
    return _ChildResult(
        messages=messages,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


@dataclass
class _ChildResult:
    messages: list[AgentMessage] | None = None
    terminal_called: bool = False
    terminal_args: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: int = 0


async def _safe_shutdown(session: Any) -> None:
    try:
        shutdown = getattr(session, "shutdown", None)
        if shutdown is not None:
            await shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    mode = config.get("mode", "async")
    if mode not in ("async", "sync"):
        mode = "async"

    extractor_k = max(1, int(config.get("extractor_interval_turns", 1)))
    auditor_k = max(1, int(config.get("audit_interval_turns", 3)))
    shutdown_timeout = max(0.0, float(config.get("shutdown_timeout_s", 600.0)))
    enable_auditor = bool(config.get("enable_auditor", True))
    enable_reminders = bool(config.get("enable_reminders", True))

    summary_threshold = int(config.get("audit_summary_threshold", 30))

    # Prompts
    ext_prompt_override = config.get("prompt_override_extractor")
    aud_prompt_override = config.get("prompt_override_auditor")
    extractor_prompt = (
        ext_prompt_override
        if isinstance(ext_prompt_override, str)
        else load_extractor_prompt(config.get("extractor_prompt") or "default")
    )
    auditor_prompt = (
        aud_prompt_override
        if isinstance(aud_prompt_override, str)
        else load_auditor_prompt(config.get("auditor_prompt") or "minimal")
    )

    # Auditor tools/profile
    auditor_tools = _resolve_auditor_tools(
        profile=config.get("auditor_profile") if isinstance(config.get("auditor_profile"), str) else None,
        tools=config.get("auditor_tools") if isinstance(config.get("auditor_tools"), list) else None,
    )

    # Tool call budget
    tcb_raw = config.get("extractor_tool_call_budget")
    tool_call_budget: int | None = (
        int(tcb_raw)
        if isinstance(tcb_raw, int) and not isinstance(tcb_raw, bool)
        else None
    )

    # Providers
    extractor_provider = _parse_provider(config.get("extractor_provider"))
    auditor_provider = _parse_provider(config.get("auditor_provider"))

    # Auditor settings (primitives version)
    aud_settings = AuditorSettings(
        base_prompt=auditor_prompt,
        summary_threshold=summary_threshold,
        tools=auditor_tools,
    )

    # Registries
    with contextlib.suppress(KeyError):
        api.set_service(AUDIT_REGISTRY_SERVICE_KEY, AuditCheckRegistry())
    trigger_registry = TriggerRegistry()
    with contextlib.suppress(KeyError):
        api.set_service(TRIGGER_SERVICE_KEY, trigger_registry)

    # State
    cumulative = CumulativeAuditState.hydrate_from_session_log(api.session.get_branch())
    pending_reminders: list[Reminder] = []
    turn_count = 0

    # Agent scenarios
    ext_scenario = extractor_scenario()
    aud_scenario = auditor_scenario()

    # ------------------------------------------------------------------
    # Core pipeline step — shared by sync and async paths
    # ------------------------------------------------------------------

    async def _step(messages: list[AgentMessage], tc: int, tool_names: frozenset[str]) -> None:
        nonlocal turn_count
        turn_count = tc

        # Cadence
        if trigger_registry:
            ctx = TriggerContext(turn_count=tc, tool_names_called=tool_names)
            auditor_due_raw, extractor_due_raw = trigger_registry.evaluate(ctx)
            auditor_due = enable_auditor and auditor_due_raw
            extractor_due = extractor_due_raw or auditor_due
        else:
            auditor_due = enable_auditor and (tc % auditor_k) == 0
            extractor_due = (tc % extractor_k) == 0 or auditor_due

        # --- Extractor ---
        if extractor_due:
            inp = build_extractor_input(
                list(messages), cumulative,
                prompt_text=extractor_prompt,
                tool_call_budget=tool_call_budget,
            )
            if inp is not None:
                overrides = {
                    "extractor_tools": {"state": inp.state},
                    "system_prompt": {"prompt": extractor_prompt},
                }
                extra: list[tuple[str, dict[str, Any]]] = []
                if tool_call_budget is not None:
                    extra.extend([
                        ("agentm.extensions.builtin.loop_budget", {"max_tool_calls": tool_call_budget}),
                        ("agentm.extensions.builtin.turn_reminder", {"warn_within": tool_call_budget}),
                    ])

                result = await _run_child(
                    api,
                    scenario=ext_scenario,
                    prompt=inp.prompt,
                    purpose="cognitive_audit_extractor",
                    atom_config_overrides=overrides,
                    extra_extensions=extra,
                    provider=extractor_provider,
                    terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
                )
                if result.error is None:
                    output = process_extractor_output(
                        inp.state, cumulative,
                        terminator_called=result.terminal_called,
                        window=inp.turn_window,
                    )
                    for op in output.ops:
                        payload = op.to_dict()
                        payload["firing_id"] = cumulative.firing_id_counter - 1
                        api.session.append_entry(_et.AUDIT_GRAPH_OP, payload)
                    api.session.append_entry(
                        _et.EXTRACTOR_CURSOR,
                        {"last_turn_index": inp.turn_window[1]},
                    )

        # --- Auditor ---
        if auditor_due:
            # Run checks
            findings, check_errors = [], {}
            registry = _get_registry(api)
            if registry is not None:
                try:
                    events, edges, _ = cumulative.graph_view()
                    ctx = CheckContext(events=events, edges=edges)
                    findings, check_errors = registry.run_all(ctx)
                except Exception:
                    _log.exception("audit-check run_all failed")

            trajectory = serialize_full_trajectory(list(messages))
            ai = build_auditor_input(
                cumulative, aud_settings,
                trajectory_snapshot=trajectory,
                findings=findings,
                check_errors=check_errors,
            )
            overrides = {
                "auditor_tools": ai.tools_config,
                "system_prompt": {"prompt": ai.prompt_text},
            }
            result = await _run_child(
                api,
                scenario=aud_scenario,
                prompt=json.dumps({
                    "graph": [e.to_dict() for e in (cumulative.graph_view()[0])],
                    "recent_verdicts": list(cumulative.recent_verdicts),
                    "continuation_notes_from_prior_firing": list(cumulative.last_continuation_notes),
                }, ensure_ascii=False, default=str),
                purpose="cognitive_audit_auditor",
                atom_config_overrides=overrides,
                provider=auditor_provider,
                terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            )
            if result.error is None:
                out = process_auditor_output(result.terminal_args, cumulative)
                if out.verdict is not None:
                    api.session.append_entry(_et.VERDICT, out.verdict.to_dict())
                    if out.verdict.surface_reminder and out.verdict.reminder_text:
                        pending_reminders.append(Reminder(text=out.verdict.reminder_text))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _build_reminder_msg(text: str) -> UserMessage:
        return text_message(REMINDER_PREAMBLE + text, timestamp=time.time())

    def _on_decide(event: DecideTurnActionEvent) -> LoopAction | None:
        if not pending_reminders:
            return None
        default = event.observation.default_action
        if isinstance(default, Stop) and default.cause.final:
            _log.warning("reminder pending but loop is final-stopped; not delivered")
            return None
        injected: list[AgentMessage] = []
        while pending_reminders:
            r = pending_reminders.pop(0)
            injected.append(_build_reminder_msg(r.text))
            try:
                api.session.append_entry(_et.REMINDER_DELIVERED, {"text": r.text})
            except Exception:
                _log.exception("failed to persist reminder_delivered")
        return Inject(messages=injected)

    if mode == "sync":
        async def _on_turn_end_sync(event: TurnEndEvent) -> None:
            nonlocal turn_count
            turn_count += 1
            await _step(list(event.messages), turn_count, tool_names_from_message(event.message))

        api.on(TurnEndEvent.CHANNEL, _on_turn_end_sync)
        if enable_reminders:
            api.on(DecideTurnActionEvent.CHANNEL, _on_decide)
        return

    # --- Async path ---
    queue: asyncio.Queue[tuple[list[AgentMessage], int, frozenset[str]] | None] = asyncio.Queue()
    worker_task: asyncio.Task[None] | None = None

    async def _worker() -> None:
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return
                msgs, tc, tools = job
                await _step(msgs, tc, tools)
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("audit worker job failed")
            finally:
                queue.task_done()

    def _ensure_worker() -> None:
        nonlocal worker_task
        if worker_task is None or worker_task.done():
            worker_task = asyncio.create_task(_worker(), name="llmharness-audit-worker")

    def _on_turn_end(event: TurnEndEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        if trigger_registry:
            enqueue = True
        else:
            auditor_due = enable_auditor and (turn_count % auditor_k) == 0
            extractor_due = (turn_count % extractor_k) == 0 or auditor_due
            enqueue = extractor_due or auditor_due
        if not enqueue:
            return
        _ensure_worker()
        queue.put_nowait((list(event.messages), turn_count, tool_names_from_message(event.message)))

    async def _on_shutdown(_event: SessionShutdownEvent) -> None:
        if worker_task is None or worker_task.done():
            return
        queue.put_nowait(None)
        try:
            await asyncio.wait_for(worker_task, timeout=shutdown_timeout)
        except asyncio.TimeoutError:
            _log.warning("audit drain exceeded %.1fs; cancelling", shutdown_timeout)
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker_task
        except Exception:
            _log.exception("audit worker raised on shutdown")

    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
    if enable_reminders:
        api.on(DecideTurnActionEvent.CHANNEL, _on_decide)
    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)


def _get_registry(api: ExtensionAPI) -> AuditCheckRegistry | None:
    try:
        r = api.get_service(AUDIT_REGISTRY_SERVICE_KEY)
    except Exception:
        return None
    return r if isinstance(r, AuditCheckRegistry) else None


def _parse_provider(raw: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    module = raw.get("module")
    if not isinstance(module, str) or not module.strip():
        return None
    cfg = raw.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return module, dict(cfg)
