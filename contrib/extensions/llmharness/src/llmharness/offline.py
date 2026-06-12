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
        _log.error("[%s] session creation FAILED: %s", purpose, exc)
        raise RuntimeError(f"session creation failed ({purpose}): {exc}") from exc

    sid = session.session_id
    _log.info("agentm trace messages --session %s --format text", sid)

    try:
        messages = await session.prompt(payload)
    except Exception as exc:
        with contextlib.suppress(Exception):
            await session.shutdown()
        _log.error("[%s] session %s prompt FAILED: %s", purpose, sid, exc)
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

    _log.warning("[%s] session %s: terminal tool %r was NOT called", purpose, sid, terminal_tool)
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

    assert len(messages) > 0, "offline_audit: empty message list"

    cumulative = CumulativeAuditState.fresh()
    surfaces: list[SurfacePoint] = []
    firings: list[AuditFiring] = []

    _EXT_CTX = "llmharness.agents.extractor.context"
    _EXT_TOOLS = "llmharness.agents.extractor.tools"
    _AUD_TOOLS = "llmharness.agents.auditor.tools"
    _OBS = "agentm.extensions.builtin.observability"
    _OPS = "agentm.extensions.builtin.operations"

    turn_bounds = _turn_end_prefix_lengths(messages)
    _log.info(
        "offline_audit: %d messages, %d turns, firing at extractor=%d auditor=%d",
        len(messages), len(turn_bounds), extractor_interval, audit_interval,
    )

    total_ext_ops = 0

    for turn_number, prefix_len in enumerate(turn_bounds, start=1):
        prefix = messages[:prefix_len]
        auditor_due = (turn_number % audit_interval) == 0
        extractor_due = (turn_number % extractor_interval) == 0 or auditor_due

        # --- Extractor ---
        ext_sid: str | None = None
        if extractor_due:
            data = _prepare_extractor_data(prefix, cumulative, None)
            assert data is not None, (
                f"extractor turn {turn_number}: _prepare_extractor_data returned None "
                f"(prefix_len={prefix_len}, cursor={cumulative.cursor_last_turn_index})"
            )

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
            payload_json = json.dumps(data, ensure_ascii=False, default=str)
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

                if ext_result.error:
                    _log.warning(
                        "  extractor turn=%d FAILED: %s (sid=%s)",
                        turn_number, ext_result.error, ext_sid,
                    )
                else:
                    from .atom import _read_ops_file

                    ops = _read_ops_file(ops_path)
                    n_ops = len(ops) if ops else 0
                    total_ext_ops += n_ops

                    if n_ops == 0:
                        _log.warning(
                            "  extractor turn=%d: 0 ops (LLM did not produce graph edits) sid=%s",
                            turn_number, ext_sid,
                        )
                    else:
                        cumulative.absorb_extractor_firing(
                            firing_ops=ops,
                            firing_cursor=data["window_hi"],
                            firing_id=firing_id,
                        )
                        _log.info("  extractor turn=%d ops=%d sid=%s", turn_number, n_ops, ext_sid)
            except Exception:
                _log.exception("extractor firing CRASHED at turn %d", turn_number)

        # --- Auditor ---
        if auditor_due:
            events, edges, phases = cumulative.graph_view()

            if not events:
                _log.warning(
                    "  auditor turn=%d: graph is EMPTY (all extractors failed or produced 0 ops), "
                    "auditor will have nothing to judge. total_ext_ops so far=%d",
                    turn_number, total_ext_ops,
                )

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
            if aud_result.error:
                _log.warning(
                    "  auditor turn=%d FAILED: %s (sid=%s)",
                    turn_number, aud_result.error, aud_result.session_id,
                )
            elif aud_result.output is None:
                _log.warning(
                    "  auditor turn=%d: no output (sid=%s)", turn_number, aud_result.session_id,
                )
            else:
                verdict_raw = aud_result.output.get("verdict") or aud_result.output
                if not isinstance(verdict_raw, dict):
                    _log.warning(
                        "  auditor turn=%d: verdict is not a dict: %s (sid=%s)",
                        turn_number, type(verdict_raw).__name__, aud_result.session_id,
                    )
                else:
                    try:
                        verdict = Verdict.from_dict(verdict_raw)
                    except (KeyError, TypeError, ValueError) as exc:
                        _log.warning(
                            "  auditor turn=%d: malformed verdict: %s (sid=%s)",
                            turn_number, exc, aud_result.session_id,
                        )
                    else:
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
                "  auditor  turn=%d graph=%d sid=%s%s",
                turn_number, len(events), aud_result.session_id, fire_mark,
            )
            firings.append(
                AuditFiring(
                    turn_number=turn_number,
                    extractor_session_id=ext_sid if extractor_due else None,
                    auditor_session_id=aud_result.session_id,
                    surfaced=surfaced,
                )
            )

    _log.info(
        "offline_audit done: %d firings, %d surfaces, %d total graph ops",
        len(firings), len(surfaces), total_ext_ops,
    )
    return OfflineAuditResult(surfaces=surfaces, firings=firings)
