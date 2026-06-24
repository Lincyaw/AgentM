"""Offline driver: replay the audit pipeline over a bare trajectory.

Takes a captured baseline run's ``final_messages`` and re-runs the
auditor pipeline against it *offline*, without re-executing the parent
agent.

The driver directly orchestrates AgentSession-based child spawns via
StandaloneChildRunner, managing cadence/cumulative-state threading
internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi import AgentMessage, AssistantMessage

from llmharness.schema import Reminder
from llmharness.state import CumulativeAuditState

from .offline import InMemorySink, StandaloneChildRunner
from .runner import AuditorSettings

__all__ = [
    "AuditorSettings",
    "OfflineRunResult",
    "SurfaceFiring",
    "replay_pipeline_over_trajectory",
]

@dataclass(frozen=True)
class SurfaceFiring:
    """One auditor firing that surfaced a reminder during an offline replay."""

    turn_index: int
    reminder_text: str
    cumulative_snapshot: CumulativeAuditState

@dataclass
class OfflineRunResult:
    """Outcome of one :func:`replay_pipeline_over_trajectory` invocation."""

    reminder: Reminder | None
    state: CumulativeAuditState
    sidecar_path: Path | None
    all_step_results: list[dict[str, Any]] = field(default_factory=list)
    surfaces: list[SurfaceFiring] = field(default_factory=list)

def _turn_end_prefix_lengths(messages: list[AgentMessage]) -> list[int]:
    """Message-prefix length at the end of each agent *turn*."""
    assistant_idxs = [i for i, m in enumerate(messages) if isinstance(m, AssistantMessage)]
    if not assistant_idxs:
        return [len(messages)] if messages else []
    bounds: list[int] = []
    for j, _ in enumerate(assistant_idxs):
        nxt = assistant_idxs[j + 1] if j + 1 < len(assistant_idxs) else len(messages)
        bounds.append(nxt)
    return bounds

async def replay_pipeline_over_trajectory(
    *,
    messages: list[AgentMessage],
    cwd: str,
    session_id: str,
    provider: tuple[str, dict[str, Any]] | None,
    auditor_settings: AuditorSettings,
    audit_interval: int = 5,
    enable_auditor: bool = True,
    stop_on_first_surface: bool = True,
    sidecar_path: Path | None = None,
    sink: InMemorySink | None = None,
    child: StandaloneChildRunner | None = None,
    seed_cumulative: CumulativeAuditState | None = None,
    start_turn: int = 1,
    symbols: list[dict[str, Any]] | None = None,
    references: list[dict[str, Any]] | None = None,
    trigger_registry: Any | None = None,
    trace_id: str | None = None,
    # Deprecated no-ops kept for backward compat
    extractor_settings: Any | None = None,
    extractor_interval: int = 5,
    skip_extractor: bool = False,
) -> OfflineRunResult:
    """Replay the cognitive-audit pipeline over a captured trajectory.

    This drives the auditor over the trajectory, managing cadence and
    cumulative state. The extractor has been replaced by the
    trajectory_index symbol table; ``symbols`` and ``references`` can be
    passed from a pre-built index.
    """
    _ = trigger_registry
    _ = sidecar_path
    _ = sink
    _ = extractor_settings
    _ = extractor_interval
    _ = skip_extractor

    resolved_trace_id = trace_id if trace_id is not None else session_id
    cumulative = seed_cumulative if seed_cumulative is not None else CumulativeAuditState.fresh()
    child_used = child if child is not None else StandaloneChildRunner(
        cwd,
        parent_session_id=session_id,
        trace_id=resolved_trace_id,
    )

    all_steps: list[dict[str, Any]] = []
    surfaces: list[SurfaceFiring] = []
    reminder: Reminder | None = None
    resolved_symbols = symbols or []
    resolved_references = references or []

    from llmharness.agents.auditor.context import build_auditor_system_prompt
    from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME

    for turn_number, prefix_len in enumerate(_turn_end_prefix_lengths(messages), start=1):
        if prefix_len < start_turn:
            continue

        prefix = messages[:prefix_len]
        turn_count = turn_number
        step_result: dict[str, Any] = {
            "turn_count": turn_count,
            "prefix_len": prefix_len,
            "surfaced_reminder": None,
            "auditor_record": None,
        }

        auditor_due = enable_auditor and (turn_count % audit_interval) == 0

        # --- Auditor ---
        if auditor_due:
            from llmharness.atom import _extract_loaded_skills, _serialize_trajectory
            from llmharness.context_index import build_context_index

            trajectory = _serialize_trajectory(prefix)
            context_index = build_context_index(
                trajectory=trajectory,
                symbols=resolved_symbols,
                references=resolved_references,
            ).to_dict()
            loaded_skills = _extract_loaded_skills(prefix)
            prompt_text = auditor_settings.base_prompt or "You are the cognitive-audit auditor."
            aud_prompt = build_auditor_system_prompt(
                check_errors={},
                continuation_notes=list(cumulative.last_continuation_notes),
                base_prompt=prompt_text,
                methodology=loaded_skills,
                context_index=context_index,
            )
            tools_config: dict[str, Any] = {
                "tools": list(auditor_settings.tools or (SUBMIT_VERDICT_TOOL_NAME,))
            }
            aud_result = await child_used.run_auditor(
                prompt_text=aud_prompt,
                tools_config=tools_config,
                provider=provider,
                context_index=context_index,
                recent_verdicts=list(cumulative.recent_verdicts),
                continuation_notes_from_prior_firing=list(cumulative.last_continuation_notes),
            )
            verdict = aud_result.get("verdict")
            if verdict is not None:
                cumulative.absorb_auditor_verdict(verdict.to_dict())
                if verdict.surface_reminder and verdict.reminder_text:
                    if stop_on_first_surface:
                        reminder = Reminder(text=verdict.reminder_text)
                        step_result["surfaced_reminder"] = reminder
                        all_steps.append(step_result)
                        break
                    surfaces.append(
                        SurfaceFiring(
                            turn_index=prefix_len - 1,
                            reminder_text=verdict.reminder_text,
                            cumulative_snapshot=cumulative.snapshot(),
                        )
                    )

        all_steps.append(step_result)

    return OfflineRunResult(
        reminder=reminder,
        state=cumulative,
        sidecar_path=sidecar_path,
        all_step_results=all_steps,
        surfaces=surfaces,
    )
