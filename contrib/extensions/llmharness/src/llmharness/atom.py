from __future__ import annotations

import asyncio
import contextlib
import difflib
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
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    TurnEndEvent,
    UserMessage,
    text_message,
)
from agentm.extensions import ChannelEffects, ExtensionManifest
from loguru import logger
from pydantic import BaseModel

from . import schema as _et
from .agents import auditor_scenario
from .agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from .schema import Reminder, Verdict
from .state import CumulativeAuditState
from .trajectory_utils import serialize_trajectory as _serialize_trajectory


class ProviderConfig(BaseModel):
    module: str
    config: dict[str, Any] = {}


class LLMHarnessConfig(BaseModel):
    mode: Literal["async", "sync"] = "async"
    audit_interval_turns: int = 3
    # When true, run the auditor once at the moment the agent tries to finish
    # (the loop is about to Stop) instead of every ``audit_interval_turns``.
    # A surfaced reminder overrides the Stop and the agent keeps working; a
    # clean verdict lets the finish proceed. Gates submission on verification.
    audit_on_finish: bool = False
    prompt_override_auditor: str | None = None
    auditor_prompt: str = "index"
    shutdown_timeout_s: float = 30.0
    auditor_provider: ProviderConfig | None = None
    auditor_model: str | None = None
    enable_auditor: bool = True
    enable_reminders: bool = True
    finalize_tool: str | None = None
    enable_methodology: bool = False
    methodology_model: str | None = None
    # Task-spec detection for methodology generation: a non-error tool result
    # is treated as spec material when its head contains one of these markers
    # (or opens with a markdown heading) and it is at least ``spec_min_chars``
    # long. Scenario manifests override the markers for other task layouts.
    spec_markers: tuple[str, ...] = ("instruction", "lab:", "task:")
    spec_min_chars: int = 200
    spec_max_chars: int = 5000
    enable_index: bool = False
    index_model: str | None = None
    index_vocabulary: str = "coding"


REMINDER_OPEN: Final = "<system-reminder>\n"
REMINDER_CLOSE: Final = "\n</system-reminder>"

# A reminder that is this similar to an already-delivered one counts as a
# repeat; after _MAX_SIMILAR_DELIVERIES repeats the issue goes silent (the
# goal checker enforces at acceptance — repeating unheeded warnings only
# devalues the channel).
_SIMILAR_REMINDER_THRESHOLD: Final = 0.6
_MAX_SIMILAR_DELIVERIES: Final = 2

_SYSTEM_PROMPT_BLOCK: Final = (
    "## Background monitor\n\n"
    "A read-only monitor periodically reviews your trajectory and may inject\n"
    "`<system-reminder>` observations. Treat them like environment feedback —\n"
    "a signal to verify, not an instruction to obey.\n\n"
    "- Each observation cites its evidence (a tool output you produced, or a\n"
    "  file the monitor read). Check that evidence against your own view of\n"
    "  the workspace before acting.\n"
    "- If it holds up, address it. If your direct observation contradicts it,\n"
    "  note the contradiction briefly and move on — the monitor sees only\n"
    "  your trajectory and may lag behind your latest changes.\n"
    "- The monitor does not decide anything; acceptance of your work is\n"
    "  judged separately."
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
    # _on_before_start appends the reminder-protocol block to event.system
    # (registered only when reminders are enabled; declared unconditionally).
    effects={"before_agent_start": ChannelEffects(appends=("system",))},
)


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------


def _render_message_text(msg: AgentMessage) -> str:
    """Extract all text from a message into one string (for witness validation)."""
    from agentm.core.abi import ThinkingBlock

    parts: list[str] = []
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return ""
    for block in content:
        if isinstance(block, ThinkingBlock):
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
    model: str | None = None,
) -> list[AgentMessage] | None:
    """Spawn a child agent session and return its messages, or None on failure."""
    config = AgentSessionConfig(
        cwd=api.cwd,
        model=model,
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
# Composable auditor API (standalone, no ExtensionAPI required)
# ---------------------------------------------------------------------------

_AUDITOR_MAX_RETRIES: Final = 3
_AUDITOR_RETRY_BASE_DELAY: Final[float] = 2.0


def build_auditor_config(
    *,
    cwd: str,
    model: str | None = None,
    provider: tuple[str, dict[str, Any]] | None = None,
    auditor_prompt: str = "index",
    continuation_notes: list[str] | None = None,
    messages: list[AgentMessage] | None = None,
    index_path: str = "",
    parent_session_id: str | None = None,
) -> AgentSessionConfig:
    """Build an ``AgentSessionConfig`` for a standalone auditor session.

    Accepts typed ``AgentMessage`` objects. Serialization happens here.
    ``index_path`` points at a persisted ``index.json``; the mounted
    ``trajectory_index_query`` atom loads it and exposes the query tools.
    """
    return AgentSessionConfig(
        cwd=cwd,
        model=model,
        provider=provider,
        scenario=auditor_scenario(),
        atom_config_overrides={
            "auditor_context": {
                "continuation_notes": list(continuation_notes or []),
                "prompt_name": auditor_prompt,
            },
            "auditor_tools": {},
            "trajectory_index_query": {
                "index_path": index_path,
            },
        },
        purpose="cognitive_audit_auditor_eval",
        parent_session_id=parent_session_id,
    )


def build_auditor_prompt(
    continuation_notes: list[str] | None = None,
) -> str:
    """Build the user-message payload for an auditor session."""
    return json.dumps(
        {"continuation_notes_from_prior_firing": list(continuation_notes or [])},
        ensure_ascii=False,
        default=str,
    )


async def run_auditor_session(
    config: AgentSessionConfig,
    prompt: str,
    *,
    followup_prompt: str | None = None,
    spawn: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Run the auditor child, return ``(verdict_raw_dict, session_id)``.

    ``spawn`` creates a session from a config. Defaults to
    ``AgentSession.create``; callers inside an atom pass
    ``api.spawn_child_session``.

    When ``followup_prompt`` is set the child is asked a second question in
    the same session (its tool reads and reasoning from the first turn carry
    over), and the verdict is taken from the follow-up turn. This drives
    progressive questioning — e.g. first decide the error mode, then localize
    conditioned on that decision.
    """
    if spawn is None:
        # ``spawn`` is required: atom callers pass ``api.spawn_child_session``;
        # non-atom callers (e.g. eval method adapters, which may import the
        # runtime) pass ``AgentSession.create``. This module must not import
        # ``agentm.core.runtime`` itself (§11.4.5).
        raise ValueError(
            "run_auditor_session requires a 'spawn' factory "
            "(api.spawn_child_session or AgentSession.create)"
        )

    for attempt in range(_AUDITOR_MAX_RETRIES):
        try:
            child = await spawn(config)
            sid = child.session_id
            try:
                child_msgs: list[AgentMessage] = await child.prompt(prompt)
                if followup_prompt is not None:
                    child_msgs = await child.prompt(followup_prompt)
                verdict_raw = _terminal_tool_args(child_msgs, SUBMIT_VERDICT_TOOL_NAME)
                if isinstance(verdict_raw, dict):
                    return verdict_raw.get("verdict", verdict_raw), sid
                return None, sid
            finally:
                with contextlib.suppress(Exception):
                    await child.shutdown()
        except Exception:
            if attempt < _AUDITOR_MAX_RETRIES - 1:
                delay = _AUDITOR_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("auditor session failed, retry {}/{} in {:.0f}s",
                               attempt + 1, _AUDITOR_MAX_RETRIES - 1, delay)
                await asyncio.sleep(delay)
            else:
                logger.exception("auditor session failed after {} attempts", _AUDITOR_MAX_RETRIES)
    return None, ""


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def install(api: ExtensionAPI, config: LLMHarnessConfig) -> None:
    cfg = config

    auditor_k = max(1, cfg.audit_interval_turns)
    shutdown_timeout = max(0.0, cfg.shutdown_timeout_s)
    enable_auditor = cfg.enable_auditor
    enable_reminders = cfg.enable_reminders
    audit_on_finish = cfg.audit_on_finish
    finalize_tool = cfg.finalize_tool

    auditor_model = cfg.auditor_model
    if auditor_model:
        logger.info("llmharness: auditor_model={!r}", auditor_model)
    elif cfg.auditor_provider:
        logger.info("llmharness: auditor uses legacy provider config, auditor inherits parent provider")
    else:
        logger.info("llmharness: auditor inherits parent provider")
    methodology_model = cfg.methodology_model or auditor_model

    # State
    cumulative = CumulativeAuditState.hydrate_from_session_log(api.session.get_branch())
    pending_reminders: list[Reminder] = []
    delivered_reminder_texts: list[str] = []
    turn_count = 0
    cached_methodology: list[str] = []

    aud_scenario = auditor_scenario()

    def _sandbox_attach_overrides() -> dict[str, Any]:
        """Attach the auditor child to the main agent's sandbox (if any).

        Without this the auditor's read tools see the host filesystem, not
        the workspace the agent is editing — its workspace claims would be
        unverifiable exactly where verification matters.
        """
        sid = api.get_service("agent_env.session_id")
        if not isinstance(sid, str) or not sid:
            return {}
        ov: dict[str, Any] = {"backend": "agent_env", "attach_session": sid}
        work_dir = api.get_service("agent_env.work_dir")
        if isinstance(work_dir, str) and work_dir:
            ov["work_dir"] = work_dir
        return {"operations": ov}

    def _similar_delivery_count(text: str) -> int:
        norm = " ".join(text.lower().split())
        return sum(
            1
            for prev in delivered_reminder_texts
            if difflib.SequenceMatcher(None, norm, prev).ratio()
            >= _SIMILAR_REMINDER_THRESHOLD
        )

    # ------------------------------------------------------------------
    # Methodology generation (spec-driven, runs once)
    # ------------------------------------------------------------------

    def _looks_like_spec(text: str) -> bool:
        head = text.lower()[:100]
        return any(marker in head for marker in cfg.spec_markers) or "# " in text[:20]

    async def _generate_methodology(messages: list[AgentMessage]) -> list[str]:
        """Extract task spec from messages and generate auditor methodology.

        Looks for the task spec in this order:
        1. Tool results matching ``cfg.spec_markers`` (most complete)
        2. First user message (fallback — may be just a pointer)
        """
        spec_parts: list[str] = []
        user_prompt = ""
        for msg in messages:
            if isinstance(msg, UserMessage) and not user_prompt:
                for block in msg.content:
                    text = getattr(block, "text", None)
                    if isinstance(text, str) and text:
                        user_prompt = text
                        break
            if isinstance(msg, ToolResultMessage):
                for result_block in msg.content:
                    if (
                        isinstance(result_block, ToolResultBlock)
                        and not result_block.is_error
                    ):
                        for sub in result_block.content:
                            text = getattr(sub, "text", None)
                            if (
                                isinstance(text, str)
                                and len(text) > cfg.spec_min_chars
                                and _looks_like_spec(text)
                            ):
                                spec_parts.append(text[: cfg.spec_max_chars])
        task_spec = "\n\n".join(spec_parts) if spec_parts else user_prompt
        if not task_spec:
            logger.warning("llmharness: no task spec found for methodology generation")
            return []

        child_msgs = await _run_child(
            api,
            scenario="llmharness:methodology_gen",
            prompt=f"Generate auditor methodology for the following task spec:\n\n{task_spec}",
            purpose="methodology_generation",
            model=methodology_model,
        )
        if child_msgs is None:
            logger.warning("llmharness: methodology generation child failed")
            return []
        methodology_text = _render_message_text(child_msgs[-1]) if child_msgs else ""
        if not methodology_text:
            for msg in reversed(child_msgs):
                methodology_text = _render_message_text(msg)
                if methodology_text:
                    break
        if methodology_text:
            logger.info(f"llmharness: generated methodology ({len(methodology_text)} chars)")
            return [methodology_text]
        return []

    # ------------------------------------------------------------------
    # Async index extraction
    # ------------------------------------------------------------------

    _latest_index_path: str | None = None

    if cfg.enable_index:
        try:
            import trajectory_index.atom  # noqa: F401 — verify import
        except ImportError:
            logger.exception(
                "llmharness: enable_index=true but trajectory-index is not "
                "installed — index extraction disabled"
            )

    async def _update_index(msgs: list[AgentMessage]) -> None:
        nonlocal _latest_index_path
        try:
            import os
            import tempfile

            from trajectory_index.atom import run_extraction
            from trajectory_index.ir.index import TrajectoryIndex

            # run_extraction expects raw AgentMessage objects (it serializes
            # them itself), not a pre-serialized trajectory snapshot.
            extraction = await run_extraction(
                api, msgs, vocabulary=cfg.index_vocabulary,
                model=cfg.index_model,
            )
            if extraction is not None:
                # Build the full index (symbols + references + grounding), not
                # just parsed symbols: populate references/claims, then
                # build_dependencies (Pass 3, deterministic) to compute the
                # grounding warnings the diagnostic relies on.
                idx = TrajectoryIndex()
                idx.populate_from_extraction(extraction, msgs, run_id=api.session_id)
                idx.build_dependencies()
                # Persist to disk. The auditor's index tools load and query it
                # from there -- a durable artifact, not an in-memory dict
                # shipped across the child-session config boundary.
                base = os.path.join(tempfile.gettempdir(), "agentm-trajectory-index")
                os.makedirs(base, exist_ok=True)
                path = os.path.join(base, f"{api.session_id or 'index'}.json")
                idx.dump(path)
                _latest_index_path = path
                stats = idx.stats(api.session_id)
                logger.info(
                    "llmharness: index built — {} symbols, {} refs, {} warnings -> {}",
                    stats.symbol_count,
                    stats.reference_count,
                    len(idx.warnings()),
                    path,
                )
        except Exception:
            logger.exception("llmharness: async index extraction failed")

    # ------------------------------------------------------------------
    # Stuck-loop compaction
    # ------------------------------------------------------------------

    _MIN_TURNS_FOR_COMPACTION = 50

    async def _request_compaction_if_stuck(n: int) -> None:
        request_compact = api.get_service("compaction.request")
        if not callable(request_compact):
            return
        msg_count = len(api.session.get_messages())
        if msg_count < _MIN_TURNS_FOR_COMPACTION:
            logger.debug(
                "llmharness: skipping compaction — only {} messages (min {})",
                msg_count, _MIN_TURNS_FOR_COMPACTION,
            )
            return
        try:
            await request_compact("stuck_loop")
            logger.info("llmharness: compaction triggered after {} consecutive reminders", n)
        except Exception:
            logger.warning("llmharness: compaction request failed after {} consecutive reminders", n)

    # ------------------------------------------------------------------
    # Core pipeline step
    # ------------------------------------------------------------------

    async def _step(
        messages: list[AgentMessage],
        tc: int,
        *,
        force: bool = False,
    ) -> None:
        nonlocal turn_count, cached_methodology
        turn_count = tc

        auditor_due = enable_auditor and ((tc % auditor_k) == 0 or force)

        # --- Methodology (once, on first auditor firing) ---
        if auditor_due and cfg.enable_methodology and not cached_methodology:
            cached_methodology = await _generate_methodology(messages)

        # --- Serialize trajectory once for index + auditor ---
        traj_snapshot: list[dict[str, Any]] | None = None
        if auditor_due:
            traj_snapshot = _serialize_trajectory(messages)

        # --- Index extraction (when auditor is due) ---
        if auditor_due and cfg.enable_index and traj_snapshot is not None:
            await _update_index(messages)

        index_path = _latest_index_path

        # --- Auditor ---
        if auditor_due:
            auditor_config: dict[str, Any] = {
                "continuation_notes": list(cumulative.last_continuation_notes),
                "prompt_name": cfg.auditor_prompt,
            }
            if cached_methodology:
                auditor_config["methodology"] = cached_methodology
            if traj_snapshot is not None:
                auditor_config["trajectory_snapshot"] = traj_snapshot
            if index_path is not None:
                auditor_config["index_path"] = index_path

            goal_condition = api.get_service("goal.condition")
            if isinstance(goal_condition, str) and goal_condition:
                auditor_config["goal_condition"] = goal_condition

            child_msgs = await _run_child(
                api,
                scenario=aud_scenario,
                prompt=json.dumps(
                    {
                        "recent_verdicts": list(cumulative.recent_verdicts),
                        "continuation_notes_from_prior_firing": list(
                            cumulative.last_continuation_notes
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                purpose="cognitive_audit_auditor",
                extra_extensions=[],
                atom_config_overrides={
                    "auditor_context": auditor_config,
                    "auditor_tools": {},
                    "trajectory_index_query": {
                        "index_path": index_path or "",
                    },
                    **_sandbox_attach_overrides(),
                },
                model=auditor_model,
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
                            text = verdict.reminder_text
                            if verdict.evidence:
                                evidence_lines = "\n".join(
                                    f"- {e}" for e in verdict.evidence if e.strip()
                                )
                                text = f"{text}\n\nEvidence:\n{evidence_lines}"
                            if _similar_delivery_count(text) >= _MAX_SIMILAR_DELIVERIES:
                                logger.info(
                                    "llmharness: suppressing repeat reminder "
                                    "(delivered {} similar already)",
                                    _MAX_SIMILAR_DELIVERIES,
                                )
                                api.session.append_entry(
                                    _et.REMINDER_SUPPRESSED, {"text": text}
                                )
                            else:
                                pending_reminders.append(Reminder(text=text))

                        n = cumulative.consecutive_reminders
                        if n >= 8 and n % 8 == 0:
                            await _request_compaction_if_stuck(n)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _build_reminder_msg(text: str) -> UserMessage:
        return text_message(f"{REMINDER_OPEN}{text}{REMINDER_CLOSE}", timestamp=time.time())

    async def _on_decide(event: DecideTurnActionEvent) -> LoopAction | None:
        default = event.observation.default_action
        # Audit-on-finish: the agent is about to stop (finish tool or a plain
        # end-turn). Run the auditor once, synchronously, against the full
        # index-grounded trajectory. If it surfaces a reminder, injecting it
        # below overrides the Stop and the loop continues; a clean verdict
        # leaves no reminder and the finish proceeds.
        if (
            audit_on_finish
            and isinstance(default, Stop)
            and not default.cause.final
        ):
            await _step(list(api.session.get_messages()), turn_count, force=True)
        if not pending_reminders:
            return None
        if isinstance(default, Stop) and default.cause.final:
            return None
        injected: list[AgentMessage] = []
        while pending_reminders:
            r = pending_reminders.pop(0)
            injected.append(_build_reminder_msg(r.text))
            delivered_reminder_texts.append(" ".join(r.text.lower().split()))
            try:
                api.session.append_entry(_et.REMINDER_DELIVERED, {"text": r.text})
            except Exception:
                logger.exception("failed to persist reminder_delivered")
        return Inject(messages=injected)

    if enable_reminders:

        def _on_before_start(event: BeforeAgentStartEvent) -> None:
            current = str(event.system or "")
            event.system = (
                f"{current}\n\n{_SYSTEM_PROMPT_BLOCK}" if current else _SYSTEM_PROMPT_BLOCK
            )

        api.on(BeforeAgentStartEvent.CHANNEL, _on_before_start)

    if cfg.mode == "sync":

        async def _on_turn_end_sync(event: TurnEndEvent) -> None:
            nonlocal turn_count
            turn_count += 1
            if audit_on_finish:
                return  # audit fires on finish (_on_decide), not per turn
            if _has_successful_tool_result(list(event.messages), finalize_tool):
                return
            await _step(list(event.messages), turn_count)

        async def _on_shutdown_sync(_event: SessionShutdownEvent) -> None:
            msgs = list(api.session.get_messages())
            if msgs and not _has_successful_tool_result(msgs, finalize_tool):
                await _step(msgs, turn_count, force=True)

        api.on(TurnEndEvent.CHANNEL, _on_turn_end_sync)
        api.on(SessionShutdownEvent.CHANNEL, _on_shutdown_sync)
        if enable_reminders or audit_on_finish:
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
        if audit_on_finish:
            return  # audit fires on finish (_on_decide), not per turn
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
            except TimeoutError:
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
    if enable_reminders or audit_on_finish:
        api.on(DecideTurnActionEvent.CHANNEL, _on_decide)
    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
