"""Offline cognitive audit over a recorded trajectory.

Public API for running the extractor + auditor pipeline offline against
an existing trajectory and returning surface points (where the auditor
would intervene). Designed to compose with ``SessionStore.fork()``:

    surfaces = await offline_audit(messages, cwd=cwd, provider=provider)
    for s in surfaces:
        forked = store.fork(baseline_sid, up_to=s.turn_index)
        ...
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    ToolCallBlock,
)

from .schema import Verdict
from .state import CumulativeAuditState

_log = logging.getLogger(__name__)

__all__ = ["SurfacePoint", "offline_audit"]


@dataclass(frozen=True)
class AuditFiring:
    """One extractor+auditor firing pair."""

    turn_number: int
    extractor_session_id: str | None
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
# Internal: run a single child session (extractor or auditor)
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
        return _PhaseResult(output=None, error=str(exc), messages=[])

    sid = session.session_id

    try:
        messages = await session.prompt(payload)
    except Exception as exc:
        with contextlib.suppress(Exception):
            await session.shutdown()
        return _PhaseResult(output=None, error=str(exc), messages=[], session_id=sid)

    with contextlib.suppress(Exception):
        await session.shutdown()

    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == terminal_tool:
                return _PhaseResult(
                    output=dict(block.arguments), error=None,
                    messages=messages, session_id=sid,
                )
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
    extractor_interval: int = 5,
    audit_interval: int = 5,
    auditor_prompt: str = "minimal",
) -> OfflineAuditResult:
    """Run extractor + auditor offline over *messages*, return audit result.

    The result contains surface points (where the auditor would intervene)
    and per-firing session ids for trace inspection.

    ``auditor_prompt`` selects the prompt variant (e.g. ``"minimal"``,
    ``"trajectory_coverage"``).
    """
    from .agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
    from .atom import _prepare_extractor_data

    cumulative = CumulativeAuditState.fresh()
    surfaces: list[SurfacePoint] = []
    firings: list[AuditFiring] = []

    _EXT_CTX = "llmharness.agents.extractor.context"
    _EXT_TOOLS = "llmharness.agents.extractor.tools"
    _AUD_TOOLS = "llmharness.agents.auditor.tools"
    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"

    for turn_number, prefix_len in enumerate(
        _turn_end_prefix_lengths(messages), start=1
    ):
        prefix = messages[:prefix_len]
        auditor_due = (turn_number % audit_interval) == 0
        extractor_due = (turn_number % extractor_interval) == 0 or auditor_due

        # --- Extractor ---
        if extractor_due:
            data = _prepare_extractor_data(prefix, cumulative, None)
            if data is not None:
                from pathlib import Path as _Path

                firing_id = cumulative.firing_id_counter
                ops_path = _Path(cwd) / ".agentm" / "audit_ops" / f"offline_{firing_id}.jsonl"

                ctx_config = dict(data)
                ctx_config["prompt_name"] = "default"
                ctx_config["ops_file"] = str(ops_path)

                extensions: list[tuple[str, dict[str, Any]]] = [
                    (_OBS, {}),
                    (_OPS, {}),
                    (_EXT_CTX, ctx_config),
                    (_EXT_TOOLS, {}),
                ]
                payload_json = json.dumps(
                    data, ensure_ascii=False, default=str
                )
                ext_sid: str | None = None
                try:
                    ext_result = await _run_phase(
                        cwd=cwd,
                        extensions=extensions,
                        provider=provider,
                        payload=payload_json,
                        terminal_tool="finalize_extraction",
                        purpose="cognitive_audit_extractor_offline",
                    )
                    ext_sid = ext_result.session_id
                    from .atom import _read_ops_file

                    ops = _read_ops_file(ops_path)
                    if ops:
                        cumulative.absorb_extractor_firing(
                            firing_ops=ops,
                            firing_cursor=data["window_hi"],
                            firing_id=firing_id,
                        )
                    _log.info(
                        "  extractor turn=%d ops=%d sid=%s  "
                        "→ agentm trace messages --session %s --format text",
                        turn_number, len(ops) if ops else 0, ext_sid, ext_sid,
                    )
                except Exception:
                    _log.exception("extractor firing failed at turn %d", turn_number)

        # --- Auditor ---
        if auditor_due:
            events, edges, phases = cumulative.graph_view()
            _AUD_CTX = "llmharness.agents.auditor.context"
            aud_ctx_config: dict[str, Any] = {
                "events": [e.to_dict() for e in events],
                "edges": [ed.to_dict() for ed in edges],
                "phases": [p.to_dict() for p in phases],
                "continuation_notes": list(cumulative.last_continuation_notes),
                "summary_threshold": 30,
                "prompt_name": auditor_prompt,
            }
            aud_extensions: list[tuple[str, dict[str, Any]]] = [
                (_OBS, {}),
                (_OPS, {}),
                (_AUD_CTX, aud_ctx_config),
                (_AUD_TOOLS, {}),
            ]
            aud_payload = json.dumps(
                {
                    "graph": [e.to_dict() for e in events],
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
            if aud_result.output is not None:
                verdict_raw = aud_result.output.get("verdict") or aud_result.output
                if isinstance(verdict_raw, dict):
                    with contextlib.suppress(KeyError, TypeError, ValueError):
                        verdict = Verdict.from_dict(verdict_raw)
                        cumulative.absorb_auditor_verdict(verdict.to_dict())
                        if verdict.surface_reminder and verdict.reminder_text:
                            surfaced = True
                            surfaces.append(
                                SurfacePoint(
                                    turn_index=prefix_len,
                                    reminder_text=verdict.reminder_text,
                                )
                            )
            fire_mark = " ★ SURFACE" if surfaced else ""
            _log.info(
                "  auditor  turn=%d graph=%d sid=%s%s  "
                "→ agentm trace messages --session %s --format text",
                turn_number, len(events), aud_result.session_id,
                fire_mark, aud_result.session_id,
            )
            firings.append(AuditFiring(
                turn_number=turn_number,
                extractor_session_id=ext_sid if extractor_due else None,
                auditor_session_id=aud_result.session_id,
                surfaced=surfaced,
            ))

    return OfflineAuditResult(surfaces=surfaces, firings=firings)
