from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any, Final, Literal

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    BeforeAgentStartEvent,
    DecideTurnActionEvent,
    ExtensionAPI,
    Inject,
    LoopAction,
    SessionShutdownEvent,
    Stop,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    TurnEndEvent,
    UserMessage,
    text_message,
)
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel

from . import schema as _et
from .agents import auditor_scenario, extractor_scenario
from .agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from .agents.extractor.index_store import IndexOp, parse_op
from .context_index import build_context_index
from .schema import Reminder, Verdict
from .state import CumulativeAuditState


class ProviderConfig(BaseModel):
    module: str
    config: dict[str, Any] = {}


class LLMHarnessConfig(BaseModel):
    mode: Literal["async", "sync"] = "async"
    extractor_interval_turns: int = 1
    audit_interval_turns: int = 3
    extractor_tool_call_budget: int | None = None
    prompt_override_extractor: str | None = None
    prompt_override_auditor: str | None = None
    extractor_prompt: str = "default"
    auditor_prompt: str = "minimal_index"
    shutdown_timeout_s: float = 600.0
    extractor_provider: ProviderConfig | None = None
    auditor_provider: ProviderConfig | None = None
    extractor_model: str | None = None
    auditor_model: str | None = None
    enable_auditor: bool = True
    enable_reminders: bool = True
    finalize_tool: str | None = None


REMINDER_OPEN: Final = "<system-reminder>\n"
REMINDER_CLOSE: Final = "\n</system-reminder>"

_SYSTEM_PROMPT_BLOCK: Final = (
    "## Automated review\n\n"
    "During your investigation you may receive `<system-reminder>` messages.\n"
    "These are from an independent reviewer monitoring your reasoning trajectory.\n\n"
    "When you receive one:\n"
    "- Treat it as a serious signal, not noise.\n"
    "- If it identifies a lead you haven't investigated, prioritize it.\n"
    "- If it flags a contradiction, re-examine the raw data rather than "
    "reasoning from memory.\n"
    "- Change your investigation direction based on the feedback — do not "
    "simply acknowledge and continue."
)

MANIFEST = ExtensionManifest(
    name="llmharness",
    description="Context-index audit harness: per-turn indexer + every-k-turns auditor.",
    registers=("event:turn_end", "event:decide_turn_action", "event:session_shutdown",
               "event:before_agent_start"),
    config_schema=LLMHarnessConfig,
    requires=("observability", "operations"),
    api_version=1,
    tier=1,
)


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def _resolve_provider(
    model_name: str | None,
    legacy: ProviderConfig | None,
) -> tuple[str, dict[str, Any]] | None:
    """Resolve a provider config from a config.toml profile name or legacy ProviderConfig."""
    if model_name is not None:
        from agentm.ai import DEFAULT_PROVIDER_DESCRIPTORS
        from agentm.core.lib import resolve_model_profile

        profile = resolve_model_profile(model_name)
        if profile is None:
            logger.warning(f"model profile {model_name!r} not found in config.toml")
            return None
        ext_module: str | None = None
        for desc in DEFAULT_PROVIDER_DESCRIPTORS:
            if desc.id == profile.provider:
                ext_module = desc.extension_module
                break
        if ext_module is None:
            logger.warning(f"provider {profile.provider!r} (from model {model_name!r}) has no extension module")
            return None
        return (ext_module, dict(profile.to_build_config()))
    if legacy is not None:
        return (legacy.module, dict(legacy.config))
    return None


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------


def _render_message_text(msg: AgentMessage) -> str:
    """Extract all text from a message into one string (for witness validation)."""
    return _render_message_text_for_index(msg, skip_tool_result_ids=set())


def _render_message_text_for_index(
    msg: AgentMessage,
    *,
    skip_tool_result_ids: set[str],
) -> str:
    """Extract message text for indexing, optionally omitting bulky tool results."""
    parts: list[str] = []
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return ""
    for block in content:
        tool_call_id = getattr(block, "tool_call_id", None)
        if (
            isinstance(tool_call_id, str)
            and tool_call_id in skip_tool_result_ids
        ):
            continue
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


def _tool_call_ids(messages: list[AgentMessage], tool_name: str) -> set[str]:
    call_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                call_ids.add(block.id)
    return call_ids


def _serialize_trajectory(
    messages: list[AgentMessage],
    *,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    from agentm.core.lib import to_jsonable

    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages, start=start_index):
        d = to_jsonable(msg)
        if isinstance(d, dict):
            d["index"] = i
            out.append(d)
    return out


def _extract_loaded_skills(messages: list[AgentMessage]) -> list[str]:
    """Extract text content from all load_skill tool results in the conversation."""
    skill_call_ids = _tool_call_ids(messages, "load_skill")
    if not skill_call_ids:
        return []
    skills: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            for block in msg.content:  # type: ignore[assignment]
                if not isinstance(block, ToolResultBlock):
                    continue
                if block.tool_call_id not in skill_call_ids or block.is_error:
                    continue
                text = " ".join(
                    inner.text
                    for inner in block.content
                    if isinstance(inner, TextContent) and inner.text
                )
                if text:
                    skills.append(text)
    return skills


def _has_successful_tool_result(
    messages: list[AgentMessage],
    tool_name: str | None,
) -> bool:
    if not tool_name:
        return False
    call_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                call_ids.add(block.id)
    if not call_ids:
        return False
    for msg in messages:
        if not isinstance(msg, ToolResultMessage):
            continue
        for block in msg.content:  # type: ignore[assignment]
            if (
                isinstance(block, ToolResultBlock)
                and block.tool_call_id in call_ids
                and not block.is_error
            ):
                return True
    return False


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
    events_cum, edges_cum, _ = cumulative.index_view()
    turn_texts: dict[str, str] = {}
    skill_call_ids = _tool_call_ids(messages, "load_skill")
    for i, msg in enumerate(window_messages, start=window_lo):
        turn_texts[str(i)] = _render_message_text_for_index(
            msg,
            skip_tool_result_ids=skill_call_ids,
        )
    for t in {t for e in events_cum for t in e.source_turns}:
        if str(t) not in turn_texts and 0 <= t < len(messages):
            turn_texts[str(t)] = _render_message_text_for_index(
                messages[t],
                skip_tool_result_ids=skill_call_ids,
            )
    new_turns = _serialize_trajectory(window_messages, start_index=window_lo)
    return {
        "turn_texts": turn_texts,
        "recent_records": [e.to_dict() for e in events_cum],
        "recent_links": [ed.to_dict() for ed in edges_cum],
        "next_event_id": cumulative.next_event_id(),
        "new_turns": new_turns,
        "tool_call_budget": tool_call_budget,
        "window_hi": window_hi,
    }


def _read_ops_file(path: Path) -> list[IndexOp]:
    if not path.exists():
        return []
    ops: list[IndexOp] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ops.append(parse_op(json.loads(line)))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            logger.warning(f"skipping malformed op line in {path}")
    return ops


# ---------------------------------------------------------------------------
# Child session helper
# ---------------------------------------------------------------------------


_CHILD_MAX_RETRIES: Final = 6
_CHILD_RETRY_BASE_DELAY: Final = 15.0
_CHILD_RETRY_MAX_DELAY: Final = 120.0


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
        lineage={
            "kind": "llmharness_child",
            "parent_session_id": api.session_id,
            "root_session_id": api.root_session_id,
            "purpose": purpose,
        },
        experiment=api.experiment,
    )
    for attempt in range(_CHILD_MAX_RETRIES):
        try:
            child = await api.spawn_child_session(config)
            try:
                result: list[AgentMessage] = await child.prompt(prompt)
                return result
            finally:
                with contextlib.suppress(Exception):
                    await child.shutdown()
        except Exception:
            if attempt < _CHILD_MAX_RETRIES - 1:
                delay = min(_CHILD_RETRY_BASE_DELAY * (2 ** attempt), _CHILD_RETRY_MAX_DELAY)
                logger.warning(
                    f"child session failed (purpose={purpose}), "
                    f"retry {attempt + 1}/{_CHILD_MAX_RETRIES - 1} in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.exception(f"child session failed after {_CHILD_MAX_RETRIES} attempts (purpose={purpose})")
    return None


def _terminal_tool_args(
    messages: list[AgentMessage],
    tool_name: str,
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
    finalize_tool = cfg.finalize_tool
    tool_call_budget = cfg.extractor_tool_call_budget

    extractor_provider = _resolve_provider(cfg.extractor_model, cfg.extractor_provider)
    auditor_provider = _resolve_provider(cfg.auditor_model, cfg.auditor_provider)

    # State
    cumulative = CumulativeAuditState.hydrate_from_session_log(api.session.get_branch())
    pending_reminders: list[Reminder] = []
    turn_count = 0

    ext_scenario = extractor_scenario()
    aud_scenario = auditor_scenario()

    # ------------------------------------------------------------------
    # Core pipeline step
    # ------------------------------------------------------------------

    async def _step(
        messages: list[AgentMessage],
        tc: int,
        *,
        force: bool = False,
    ) -> None:
        nonlocal turn_count
        turn_count = tc

        auditor_due = enable_auditor and (tc % auditor_k) == 0
        extractor_due = (tc % extractor_k) == 0 or auditor_due
        if force:
            extractor_due = True
            auditor_due = enable_auditor

        # --- Extractor ---
        if extractor_due:
            data = _prepare_extractor_data(messages, cumulative, tool_call_budget)
            if data is not None:
                firing_id = cumulative.firing_id_counter
                session_prefix = api.session.get_session_id()[:8]
                ops_path = Path(api.cwd) / ".agentm" / "audit_ops" / f"{session_prefix}_{firing_id}.jsonl"

                extra: list[tuple[str, dict[str, Any]]] = []
                if tool_call_budget is not None:
                    extra.extend(
                        [
                            (
                                "agentm.extensions.builtin.loop_budget",
                                {"max_tool_calls": tool_call_budget},
                            ),
                            (
                                "agentm.extensions.builtin.turn_reminder",
                                {"warn_within": tool_call_budget},
                            ),
                        ]
                    )

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
                        api.session.append_entry(_et.AUDIT_INDEX_OP, d)
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
            events, edges, _phases = cumulative.index_view()
            trajectory = _serialize_trajectory(list(messages))
            context_index = build_context_index(
                trajectory=trajectory,
                events=events,
                edges=edges,
            ).to_dict()
            loaded_skills = _extract_loaded_skills(messages)

            child_msgs = await _run_child(
                api,
                scenario=aud_scenario,
                prompt=json.dumps(
                    {
                        "records": [e.to_dict() for e in events],
                        "links": [ed.to_dict() for ed in edges],
                        "context_index": context_index,
                        "recent_verdicts": list(cumulative.recent_verdicts),
                        "continuation_notes_from_prior_firing": list(
                            cumulative.last_continuation_notes
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                purpose="cognitive_audit_auditor",
                atom_config_overrides={
                    "auditor_context": {
                        "events": [e.to_dict() for e in events],
                        "edges": [ed.to_dict() for ed in edges],
                        "continuation_notes": list(cumulative.last_continuation_notes),
                        "prompt_name": cfg.auditor_prompt,
                        "trajectory_snapshot": trajectory,
                        "context_index": context_index,
                        "methodology": loaded_skills,
                    },
                    "auditor_tools": {},
                },
                provider=auditor_provider,
            )
            if child_msgs is not None:
                args = _terminal_tool_args(child_msgs, SUBMIT_VERDICT_TOOL_NAME)
                if args is not None:
                    verdict_raw = args.get("verdict")
                    if isinstance(verdict_raw, dict):
                        verdict = Verdict.from_dict(verdict_raw)
                        cumulative.absorb_auditor_verdict(verdict.to_dict())
                        api.session.append_entry(_et.VERDICT, verdict.to_dict())
                        if verdict.surface_reminder and verdict.reminder_text:
                            pending_reminders.append(Reminder(text=verdict.reminder_text))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _build_reminder_msg(text: str) -> UserMessage:
        return text_message(f"{REMINDER_OPEN}{text}{REMINDER_CLOSE}", timestamp=time.time())

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
                logger.exception("failed to persist reminder_delivered")
        return Inject(messages=injected)

    if enable_reminders:

        def _on_before_start(event: BeforeAgentStartEvent) -> None:
            current = str(event.system or "")
            event.system = f"{current}\n\n{_SYSTEM_PROMPT_BLOCK}" if current else _SYSTEM_PROMPT_BLOCK

        api.on(BeforeAgentStartEvent.CHANNEL, _on_before_start)

    if cfg.mode == "sync":

        async def _on_turn_end_sync(event: TurnEndEvent) -> None:
            nonlocal turn_count
            turn_count += 1
            if _has_successful_tool_result(list(event.messages), finalize_tool):
                return
            await _step(list(event.messages), turn_count)

        async def _on_shutdown_sync(_event: SessionShutdownEvent) -> None:
            msgs = list(api.session.get_messages())
            if msgs and not _has_successful_tool_result(msgs, finalize_tool):
                await _step(msgs, turn_count, force=True)

        api.on(TurnEndEvent.CHANNEL, _on_turn_end_sync)
        api.on(SessionShutdownEvent.CHANNEL, _on_shutdown_sync)
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
                logger.exception("audit worker job failed")
            finally:
                queue.task_done()

    def _ensure_worker() -> None:
        nonlocal worker_task
        if worker_task is None or worker_task.done():
            worker_task = asyncio.create_task(_worker(), name="llmharness-audit-worker")

    def _on_turn_end(event: TurnEndEvent) -> None:
        nonlocal turn_count
        turn_count += 1
        if _has_successful_tool_result(list(event.messages), finalize_tool):
            return
        auditor_due = enable_auditor and (turn_count % auditor_k) == 0
        extractor_due = (turn_count % extractor_k) == 0 or auditor_due
        if not (extractor_due or auditor_due):
            return
        _ensure_worker()
        queue.put_nowait((list(event.messages), turn_count))

    async def _on_shutdown(_event: SessionShutdownEvent) -> None:
        if worker_task is not None and not worker_task.done():
            queue.put_nowait(None)
            try:
                await asyncio.wait_for(worker_task, timeout=shutdown_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"audit drain exceeded {shutdown_timeout:.1f}s; cancelling")
                worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await worker_task
            except Exception:
                logger.exception("audit worker raised on shutdown")
        msgs = list(api.session.get_messages())
        if msgs and not _has_successful_tool_result(msgs, finalize_tool):
            await _step(msgs, turn_count, force=True)

    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
    if enable_reminders:
        api.on(DecideTurnActionEvent.CHANNEL, _on_decide)
    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
