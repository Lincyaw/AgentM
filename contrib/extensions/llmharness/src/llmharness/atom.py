from __future__ import annotations

import asyncio
import contextlib
import json
import time
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
from .agents import auditor_scenario
from .agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from .context_index import build_context_index
from .schema import Reminder, Verdict
from .state import CumulativeAuditState


class ProviderConfig(BaseModel):
    module: str
    config: dict[str, Any] = {}


class LLMHarnessConfig(BaseModel):
    mode: Literal["async", "sync"] = "async"
    audit_interval_turns: int = 3
    prompt_override_auditor: str | None = None
    auditor_prompt: str = "minimal_index"
    shutdown_timeout_s: float = 600.0
    auditor_provider: ProviderConfig | None = None
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
    description="Context-index audit harness: trajectory indexer + every-k-turns auditor.",
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
            for block in msg.content:
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

    auditor_k = max(1, cfg.audit_interval_turns)
    shutdown_timeout = max(0.0, cfg.shutdown_timeout_s)
    enable_auditor = cfg.enable_auditor
    enable_reminders = cfg.enable_reminders
    finalize_tool = cfg.finalize_tool

    auditor_provider = _resolve_provider(cfg.auditor_model, cfg.auditor_provider)

    # State
    cumulative = CumulativeAuditState.hydrate_from_session_log(api.session.get_branch())
    pending_reminders: list[Reminder] = []
    turn_count = 0

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

        auditor_due = enable_auditor and ((tc % auditor_k) == 0 or force)

        # --- Auditor ---
        if auditor_due:
            trajectory = _serialize_trajectory(list(messages))
            context_index = build_context_index(
                trajectory=trajectory,
                symbols=[],
                references=[],
            ).to_dict()
            loaded_skills = _extract_loaded_skills(messages)

            child_msgs = await _run_child(
                api,
                scenario=aud_scenario,
                prompt=json.dumps(
                    {
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
                        "continuation_notes": list(cumulative.last_continuation_notes),
                        "prompt_name": cfg.auditor_prompt,
                        "trajectory_snapshot": trajectory,
                        "context_index": context_index,
                        "methodology": loaded_skills,
                    },
                    "auditor_tools": {},
                    "auditor_index_tools": {
                        "trajectory": trajectory,
                        "symbols": [],
                        "references": [],
                    },
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
        if not auditor_due:
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
