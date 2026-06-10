from __future__ import annotations

import asyncio
import collections
import contextlib
import copy
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

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
from agentm.core.abi.session import SessionEntry
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from . import schema as _et
from .agents import auditor_scenario, extractor_scenario
from .agents.auditor.tools import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
)
from .agents.extractor.tools import GraphOp, parse_op
from .schema import Edge, Event, Phase, Reminder


class ProviderConfig(BaseModel):
    module: str
    config: dict[str, Any] = {}


class LLMHarnessConfig(BaseModel):
    mode: Literal["async", "sync"] = "async"
    extractor_interval_turns: int = 1
    audit_interval_turns: int = 3
    audit_summary_threshold: int = 30
    extractor_tool_call_budget: int | None = None
    prompt_override_extractor: str | None = None
    prompt_override_auditor: str | None = None
    extractor_prompt: str = "default"
    auditor_prompt: str = "minimal"
    auditor_profile: str = "minimal"
    auditor_tools: list[str] | None = None
    shutdown_timeout_s: float = 600.0
    extractor_provider: ProviderConfig | None = None
    auditor_provider: ProviderConfig | None = None
    enable_auditor: bool = True
    enable_reminders: bool = True

_log = logging.getLogger(__name__)

REMINDER_PREAMBLE: Final = "[system reminder — automated review of your investigation so far]\n"

_DEFAULT_RECENT_VERDICTS: Final[int] = _et.RECENT_VERDICTS_FOR_AUDITOR

MANIFEST = ExtensionManifest(
    name="llmharness",
    description="Two-phase cognitive-audit: per-turn extractor + every-k-turns auditor.",
    registers=("event:turn_end", "event:decide_turn_action", "event:session_shutdown"),
    config_schema=LLMHarnessConfig.model_json_schema(),
    requires=("observability", "operations"),
    api_version=1,
    tier=1,
)


# ---------------------------------------------------------------------------
# CumulativeAuditState
# ---------------------------------------------------------------------------


def _bool_safe_int(raw: Any) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


@dataclass
class CumulativeAuditState:
    """Event-sourced graph state + auditor side-channel state across firings."""

    ops: list[GraphOp] = field(default_factory=list)
    cursor_last_turn_index: int = -1
    recent_verdicts: collections.deque[dict[str, Any]] = field(
        default_factory=lambda: collections.deque(maxlen=_DEFAULT_RECENT_VERDICTS)
    )
    last_continuation_notes: list[str] = field(default_factory=list)
    firing_id_counter: int = 0
    _cached_len: int = -1
    _cached_view: tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]] | None = None
    _phases: list[Phase] = field(default_factory=list)

    def graph_view(self) -> tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]]:
        if self._cached_view is not None and self._cached_len == len(self.ops):
            return self._cached_view
        from .agents.extractor.tools import fold_graph
        folded = fold_graph(self.ops)
        events = tuple(folded.nodes_list())
        edges = tuple(folded.edges_list())
        phases = tuple(self._phases)
        self._cached_view = (events, edges, phases)
        self._cached_len = len(self.ops)
        return self._cached_view

    def next_event_id(self) -> int:
        events, _edges, _phases = self.graph_view()
        return max((e.id for e in events), default=0) + 1

    def _invalidate_cache(self) -> None:
        self._cached_view = None
        self._cached_len = -1

    def absorb_extractor_firing(
        self,
        *,
        firing_ops: Sequence[GraphOp],
        firing_cursor: int,
        firing_id: int,
        firing_phases: Sequence[Phase] = (),
    ) -> None:
        self.ops.extend(firing_ops)
        self.cursor_last_turn_index = firing_cursor
        self._phases.extend(firing_phases)
        if firing_id >= self.firing_id_counter:
            self.firing_id_counter = firing_id + 1
        self._invalidate_cache()

    def absorb_auditor_verdict(self, verdict: dict[str, Any]) -> None:
        self.recent_verdicts.append(verdict)
        raw_notes = verdict.get("continuation_notes")
        if isinstance(raw_notes, list):
            self.last_continuation_notes = [n for n in raw_notes if isinstance(n, str)]

    @classmethod
    def fresh(cls) -> CumulativeAuditState:
        return cls()

    def snapshot(self) -> CumulativeAuditState:
        return copy.deepcopy(self)

    @classmethod
    def hydrate_from_session_log(cls, branch: list[SessionEntry]) -> CumulativeAuditState:
        ops: list[GraphOp] = []
        verdicts_all: list[dict[str, Any]] = []
        cursor_last_turn_index = -1
        for entry in branch:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if entry.type == _et.AUDIT_GRAPH_OP:
                try:
                    ops.append(parse_op(payload))
                except (KeyError, ValueError, TypeError):
                    continue
            elif entry.type == _et.VERDICT:
                verdicts_all.append(payload)
            elif entry.type == _et.EXTRACTOR_CURSOR:
                raw = _bool_safe_int(payload.get("last_turn_index"))
                if raw is not None:
                    cursor_last_turn_index = raw
        last_notes: list[str] = []
        if verdicts_all:
            raw_notes = verdicts_all[-1].get("continuation_notes")
            if isinstance(raw_notes, list):
                last_notes = [n for n in raw_notes if isinstance(n, str)]
        recent: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_DEFAULT_RECENT_VERDICTS
        )
        for v in verdicts_all[-_DEFAULT_RECENT_VERDICTS:]:
            recent.append(v)
        return cls(
            ops=ops,
            cursor_last_turn_index=cursor_last_turn_index,
            recent_verdicts=recent,
            last_continuation_notes=last_notes,
            firing_id_counter=0,
        )


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------


def _render_message_text(msg: AgentMessage) -> str:
    """Extract all text from a message into one string (for witness validation)."""
    parts: list[str] = []
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return ""
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


def _serialize_trajectory(
    messages: list[AgentMessage], *, start_index: int = 0,
) -> list[dict[str, Any]]:
    from agentm.core.lib import to_jsonable
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages, start=start_index):
        d = to_jsonable(msg)
        if isinstance(d, dict):
            d["index"] = i
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Data preparation
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
    new_turns = _serialize_trajectory(window_messages, start_index=window_lo)
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
    """Spawn a child agent session and return its messages, or None on failure."""
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


def install(api: ExtensionAPI, config: LLMHarnessConfig) -> None:
    cfg = config

    extractor_k = max(1, cfg.extractor_interval_turns)
    auditor_k = max(1, cfg.audit_interval_turns)
    shutdown_timeout = max(0.0, cfg.shutdown_timeout_s)
    enable_auditor = cfg.enable_auditor
    enable_reminders = cfg.enable_reminders
    summary_threshold = cfg.audit_summary_threshold
    tool_call_budget = cfg.extractor_tool_call_budget

    extractor_provider = (cfg.extractor_provider.module, dict(cfg.extractor_provider.config)) if cfg.extractor_provider else None
    auditor_provider = (cfg.auditor_provider.module, dict(cfg.auditor_provider.config)) if cfg.auditor_provider else None

    # State
    cumulative = CumulativeAuditState.hydrate_from_session_log(api.session.get_branch())
    pending_reminders: list[Reminder] = []
    turn_count = 0

    ext_scenario = extractor_scenario()
    aud_scenario = auditor_scenario()

    # ------------------------------------------------------------------
    # Core pipeline step
    # ------------------------------------------------------------------

    async def _step(messages: list[AgentMessage], tc: int) -> None:
        nonlocal turn_count
        turn_count = tc

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
                ctx_config["ops_file"] = str(ops_path)
                ctx_config["prompt_name"] = cfg.extractor_prompt
                if cfg.prompt_override_extractor is not None:
                    ctx_config["prompt_text"] = cfg.prompt_override_extractor

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
            events, edges, phases = cumulative.graph_view()
            trajectory = _serialize_trajectory(list(messages))

            child_msgs = await _run_child(
                api,
                scenario=aud_scenario,
                prompt=json.dumps({
                    "graph": [e.to_dict() for e in events],
                    "recent_verdicts": list(cumulative.recent_verdicts),
                    "continuation_notes_from_prior_firing": list(cumulative.last_continuation_notes),
                }, ensure_ascii=False, default=str),
                purpose="cognitive_audit_auditor",
                atom_config_overrides={
                    "auditor_context": {
                        "events": [e.to_dict() for e in events],
                        "edges": [ed.to_dict() for ed in edges],
                        "phases": [p.to_dict() for p in phases],
                        "continuation_notes": list(cumulative.last_continuation_notes),
                        "summary_threshold": summary_threshold,
                        "prompt_name": cfg.auditor_prompt,
                        "trajectory_snapshot": trajectory,
                    },
                    "auditor_tools": {
                        "profile": cfg.auditor_profile,
                        **({"tools": list(cfg.auditor_tools)} if cfg.auditor_tools else {}),
                        "trajectory_snapshot": trajectory,
                        "events": [e.to_dict() for e in events],
                        "edges": [ed.to_dict() for ed in edges],
                    },
                },
                provider=auditor_provider,
            )
            if child_msgs is not None:
                args = _terminal_tool_args(child_msgs, SUBMIT_VERDICT_TOOL_NAME)
                if args is not None:
                    try:
                        raw = RawVerdictOutput.from_dict(args)
                        verdict = raw.to_verdict()
                    except AuditorOutputError:
                        _log.warning("auditor output malformed")
                    else:
                        cumulative.absorb_auditor_verdict(verdict.to_dict())
                        api.session.append_entry(_et.VERDICT, verdict.to_dict())
                        if verdict.surface_reminder and verdict.reminder_text:
                            pending_reminders.append(Reminder(text=verdict.reminder_text))

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

    if cfg.mode == "sync":
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


