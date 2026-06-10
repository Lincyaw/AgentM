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
from pathlib import Path
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
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    UserMessage,
    text_message,
)
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest

from . import schema as _et
from .agents import auditor_scenario, extractor_scenario
from .agents.auditor.profiles import resolve_tools as _resolve_auditor_tools
from .agents.auditor.prompt import load_auditor_prompt
from .agents.auditor.submit_verdict import SUBMIT_VERDICT_TOOL_NAME
from .agents.extractor.prompt import load_extractor_prompt
from .graph.ops import GraphOp, parse_op
from .primitives import (
    AuditorSettings,
    CumulativeAuditState,
    _render_message_text,
    _serialize_message,
    build_auditor_input,
    process_auditor_output,
    serialize_full_trajectory,
)
from .schema import Reminder

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
# Extractor data preparation
# ---------------------------------------------------------------------------


def _prepare_extractor_data(
    messages: list[AgentMessage],
    cumulative: CumulativeAuditState,
    tool_call_budget: int | None,
) -> dict[str, Any] | None:
    """Prepare raw data for the extractor child. Returns None if window is empty."""
    window_lo = max(cumulative.cursor_last_turn_index + 1, 0)
    window_hi = len(messages) - 1
    window_messages = messages[window_lo:]
    if not window_messages:
        return None

    events_cum, edges_cum, _ = cumulative.graph_view()

    turn_texts: dict[str, str] = {}
    for i, msg in enumerate(window_messages, start=window_lo):
        turn_texts[str(i)] = _render_message_text(msg)
    for t in {t for e in events_cum for t in e.source_turns}:
        if str(t) not in turn_texts and 0 <= t < len(messages):
            turn_texts[str(t)] = _render_message_text(messages[t])

    new_turns = [
        s for s in (
            _serialize_message(msg, index=i)
            for i, msg in enumerate(window_messages, start=window_lo)
        ) if s is not None
    ]

    return {
        "turn_texts": turn_texts,
        "recent_graph": [e.to_dict() for e in events_cum],
        "recent_edges": [ed.to_dict() for ed in edges_cum],
        "next_event_id": cumulative.next_event_id(),
        "new_turns": new_turns,
        "tool_call_budget": tool_call_budget,
        "window_hi": window_hi,
    }


def _read_ops_file(path: Path) -> list[GraphOp]:
    """Read persisted ops from the JSONL file the extractor tools wrote."""
    if not path.exists():
        return []
    ops: list[GraphOp] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ops.append(parse_op(json.loads(line)))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            _log.warning("skipping malformed op line in %s", path)
    return ops


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
) -> list[AgentMessage] | None:
    """Spawn a child agent session, drive to completion, return messages.

    Returns None on any failure (spawn or prompt). Shutdown is always
    executed via try/finally — no separate helper needed.
    """
    config = AgentSessionConfig(
        cwd=api.cwd,
        provider=provider,
        scenario=scenario,
        extra_extensions=extra_extensions or [],
        atom_config_overrides=atom_config_overrides or {},
        purpose=purpose,
    )
    try:
        child = await api.spawn_child_session(config)
        try:
            return await child.prompt(prompt)
        finally:
            with contextlib.suppress(Exception):
                await child.shutdown()
    except Exception:
        _log.exception("child session failed (purpose=%s)", purpose)
        return None


def _terminal_tool_args(
    messages: list[AgentMessage], tool_name: str,
) -> dict[str, Any] | None:
    """Extract the last call to ``tool_name`` from messages."""
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


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

    async def _step(messages: list[AgentMessage], tc: int) -> None:
        nonlocal turn_count
        turn_count = tc

        # Cadence
        auditor_due = enable_auditor and (tc % auditor_k) == 0
        extractor_due = (tc % extractor_k) == 0 or auditor_due

        # --- Extractor ---
        if extractor_due:
            data = _prepare_extractor_data(messages, cumulative, tool_call_budget)
            if data is not None:
                firing_id = cumulative.firing_id_counter
                ops_path = Path(api.cwd) / ".agentm" / "audit_ops" / f"{firing_id}.jsonl"

                extra: list[tuple[str, dict[str, Any]]] = []
                if tool_call_budget is not None:
                    extra.extend([
                        ("agentm.extensions.builtin.loop_budget", {"max_tool_calls": tool_call_budget}),
                        ("agentm.extensions.builtin.turn_reminder", {"warn_within": tool_call_budget}),
                    ])

                ctx_config = dict(data)
                ctx_config["prompt_text"] = extractor_prompt
                ctx_config["ops_file"] = str(ops_path)

                child_msgs = await _run_child(
                    api,
                    scenario=ext_scenario,
                    prompt=json.dumps(data, ensure_ascii=False, default=str),
                    purpose="cognitive_audit_extractor",
                    atom_config_overrides={"extractor_context": ctx_config},
                    extra_extensions=extra,
                    provider=extractor_provider,
                )
                if child_msgs is not None:
                    ops = _read_ops_file(ops_path)
                    for op in ops:
                        d = op.to_dict()
                        d["firing_id"] = firing_id
                        api.session.append_entry(_et.AUDIT_GRAPH_OP, d)
                    cumulative.absorb_extractor_firing(
                        firing_ops=ops,
                        firing_cursor=data["window_hi"],
                        firing_id=firing_id,
                    )
                    api.session.append_entry(
                        _et.EXTRACTOR_CURSOR,
                        {"last_turn_index": data["window_hi"]},
                    )

        # --- Auditor ---
        if auditor_due:
            trajectory = serialize_full_trajectory(list(messages))
            ai = build_auditor_input(
                cumulative, aud_settings,
                trajectory_snapshot=trajectory,
            )
            child_msgs = await _run_child(
                api,
                scenario=aud_scenario,
                prompt=json.dumps({
                    "graph": [e.to_dict() for e in (cumulative.graph_view()[0])],
                    "recent_verdicts": list(cumulative.recent_verdicts),
                    "continuation_notes_from_prior_firing": list(cumulative.last_continuation_notes),
                }, ensure_ascii=False, default=str),
                purpose="cognitive_audit_auditor",
                atom_config_overrides={
                    "auditor_tools": ai.tools_config,
                    "system_prompt": {"prompt": ai.prompt_text},
                },
                provider=auditor_provider,
            )
            if child_msgs is not None:
                args = _terminal_tool_args(child_msgs, SUBMIT_VERDICT_TOOL_NAME)
                out = process_auditor_output(args, cumulative)
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
            await _step(list(event.messages), turn_count)

        api.on(TurnEndEvent.CHANNEL, _on_turn_end_sync)
        if enable_reminders:
            api.on(DecideTurnActionEvent.CHANNEL, _on_decide)
        return

    # --- Async path ---
    queue: asyncio.Queue[tuple[list[AgentMessage], int] | None] = asyncio.Queue()
    worker_task: asyncio.Task[None] | None = None

    async def _worker() -> None:
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return
                msgs, tc = job
                await _step(msgs, tc)
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
        auditor_due = enable_auditor and (turn_count % auditor_k) == 0
        extractor_due = (turn_count % extractor_k) == 0 or auditor_due
        if not (extractor_due or auditor_due):
            return
        _ensure_worker()
        queue.put_nowait((list(event.messages), turn_count))

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
