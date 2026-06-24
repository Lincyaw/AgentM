"""Offline cognitive audit over a recorded trajectory.

Public API for running the auditor pipeline offline against an existing
trajectory and returning surface points (where the auditor would
intervene). Designed to compose with ``SessionStore.fork()``:

    surfaces = await offline_audit(messages, cwd=cwd, provider=provider)
    for s in surfaces:
        forked = store.fork(baseline_sid, up_to=s.turn_index)
        ...
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    ToolCallBlock,
)
from loguru import logger

from .schema import Verdict
from .state import CumulativeAuditState

__all__ = ["SurfacePoint", "offline_audit"]


@dataclass(frozen=True)
class AuditFiring:
    """One auditor firing."""

    turn_number: int
    auditor_session_id: str | None
    surfaced: bool


@dataclass(frozen=True)
class SurfacePoint:
    """A position in the trajectory where the auditor would intervene."""

    turn_index: int
    reminder_text: str


@dataclass
class OfflineAuditResult:
    """Full result of an offline audit run."""

    surfaces: list[SurfacePoint]
    firings: list[AuditFiring]


# ---------------------------------------------------------------------------
# Internal: run a single child session (auditor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PhaseResult:
    output: dict[str, Any] | None
    error: str | None
    messages: list[AgentMessage]
    session_id: str | None = None


async def _run_phase(
    *,
    cwd: str,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    payload: str,
    terminal_tool: str,
    purpose: str,
) -> _PhaseResult:
    from agentm.core.runtime import AgentSession, create_agent_session

    config = AgentSessionConfig(
        cwd=cwd,
        provider=provider,
        extensions=extensions,
        purpose=purpose,
    )
    try:
        session = await create_agent_session(AgentSession, config)
    except Exception as exc:
        logger.error(f"[{purpose}] session creation FAILED: {exc}")
        raise RuntimeError(f"session creation failed ({purpose}): {exc}") from exc

    sid = session.session_id
    label = purpose.replace("cognitive_audit_", "").replace("_offline", "")
    logger.info(f"[{label}] agentm trace messages --session {sid} --format text")

    try:
        messages = await session.prompt(payload)
    except Exception as exc:
        with contextlib.suppress(Exception):
            await session.shutdown()
        logger.error(f"[{purpose}] session {sid} prompt FAILED: {exc}")
        return _PhaseResult(output=None, error=str(exc), messages=[], session_id=sid)

    with contextlib.suppress(Exception):
        await session.shutdown()

    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == terminal_tool:
                return _PhaseResult(
                    output=dict(block.arguments),
                    error=None,
                    messages=messages,
                    session_id=sid,
                )

    logger.warning(f"[{purpose}] session {sid}: terminal tool {terminal_tool!r} was NOT called")
    return _PhaseResult(
        output=None,
        error=f"{terminal_tool!r} was not called",
        messages=messages,
        session_id=sid,
    )


# ---------------------------------------------------------------------------
# Turn boundaries
# ---------------------------------------------------------------------------


def _turn_end_prefix_lengths(messages: list[AgentMessage]) -> list[int]:
    """Message-prefix length at the end of each agent turn."""
    idxs = [i for i, m in enumerate(messages) if isinstance(m, AssistantMessage)]
    if not idxs:
        return [len(messages)] if messages else []
    bounds: list[int] = []
    for j, _ in enumerate(idxs):
        nxt = idxs[j + 1] if j + 1 < len(idxs) else len(messages)
        bounds.append(nxt)
    return bounds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def offline_audit(
    messages: list[AgentMessage],
    *,
    cwd: str,
    provider: tuple[str, dict[str, Any]],
    audit_interval: int = 5,
    auditor_prompt: str = "minimal_index",
    stop_on_first_surface: bool = False,
    min_surface_turn_index: int = 0,
) -> OfflineAuditResult:
    """Run auditor offline over *messages*, return audit result.

    The result contains surface points (where the auditor would intervene)
    and per-firing session ids for trace inspection.  Set
    ``stop_on_first_surface`` for fork-replay flows that only use the first
    surfaced reminder.  ``min_surface_turn_index`` suppresses reminders at or
    before a message prefix length that has already been forked from; this keeps
    multi-generation replay moving forward instead of rediscovering an earlier
    intervention point.

    ``auditor_prompt`` selects the prompt variant (e.g. ``"minimal_index"``,
    ``"trajectory_coverage"``).
    """
    from .agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME

    assert len(messages) > 0, "offline_audit: empty message list"

    cumulative = CumulativeAuditState.fresh()
    surfaces: list[SurfacePoint] = []
    firings: list[AuditFiring] = []

    _AUD_TOOLS = "llmharness.agents.auditor.tools"
    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"

    turn_bounds = _turn_end_prefix_lengths(messages)
    logger.info(
        f"offline_audit: {len(messages)} messages, {len(turn_bounds)} turns, "
        f"firing at auditor={audit_interval}"
    )

    for turn_number, prefix_len in enumerate(turn_bounds, start=1):
        prefix = messages[:prefix_len]
        auditor_due = (turn_number % audit_interval) == 0

        # --- Auditor ---
        if auditor_due:
            stop_after_auditor = False
            from .atom import _extract_loaded_skills, _serialize_trajectory
            from .context_index import build_context_index

            trajectory = _serialize_trajectory(prefix)
            context_index = build_context_index(
                trajectory=trajectory,
                symbols=[],
                references=[],
            ).to_dict()
            loaded_skills = _extract_loaded_skills(prefix)

            _AUD_CTX = "llmharness.agents.auditor.context"
            aud_ctx_config: dict[str, Any] = {
                "continuation_notes": list(cumulative.last_continuation_notes),
                "prompt_name": auditor_prompt,
                "trajectory_snapshot": trajectory,
                "context_index": context_index,
                "methodology": loaded_skills,
            }
            aud_extensions: list[tuple[str, dict[str, Any]]] = [
                (_OBS, {}),
                (_OPS, {}),
                (_AUD_CTX, aud_ctx_config),
                (_AUD_TOOLS, {}),
            ]
            aud_payload = json.dumps(
                {
                    "context_index": context_index,
                    "recent_verdicts": list(cumulative.recent_verdicts),
                    "continuation_notes_from_prior_firing": list(
                        cumulative.last_continuation_notes
                    ),
                },
                ensure_ascii=False,
                default=str,
            )
            aud_result = await _run_phase(
                cwd=cwd,
                extensions=aud_extensions,
                provider=provider,
                payload=aud_payload,
                terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
                purpose="cognitive_audit_auditor_offline",
            )

            surfaced = False
            if aud_result.error:
                logger.warning(f"  auditor turn={turn_number} FAILED: {aud_result.error} (sid={aud_result.session_id})")
            elif aud_result.output is None:
                logger.warning(f"  auditor turn={turn_number}: no output (sid={aud_result.session_id})")
            else:
                verdict_raw = aud_result.output.get("verdict") or aud_result.output
                if not isinstance(verdict_raw, dict):
                    logger.warning(
                        f"  auditor turn={turn_number}: verdict is not a dict: "
                        f"{type(verdict_raw).__name__} (sid={aud_result.session_id})"
                    )
                else:
                    try:
                        verdict = Verdict.from_dict(verdict_raw)
                    except (KeyError, TypeError, ValueError) as exc:
                        logger.warning(f"  auditor turn={turn_number}: malformed verdict: {exc} (sid={aud_result.session_id})")
                    else:
                        cumulative.absorb_auditor_verdict(verdict.to_dict())
                        surface_allowed = prefix_len > min_surface_turn_index
                        if (
                            verdict.surface_reminder
                            and verdict.reminder_text
                            and surface_allowed
                        ):
                            surfaced = True
                            surfaces.append(
                                SurfacePoint(
                                    turn_index=prefix_len,
                                    reminder_text=verdict.reminder_text,
                                )
                            )
                            stop_after_auditor = stop_on_first_surface

            fire_mark = " ★ SURFACE" if surfaced else ""
            logger.info(f"  auditor  turn={turn_number} sid={aud_result.session_id}{fire_mark}")
            firings.append(
                AuditFiring(
                    turn_number=turn_number,
                    auditor_session_id=aud_result.session_id,
                    surfaced=surfaced,
                )
            )
            if stop_after_auditor:
                break

    logger.info(f"offline_audit done: {len(firings)} firings, {len(surfaces)} surfaces")
    return OfflineAuditResult(surfaces=surfaces, firings=firings)
