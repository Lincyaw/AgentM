"""Offline driver: replay the audit pipeline over a bare trajectory.

Takes a captured baseline run's ``final_messages`` and re-runs the
extractor + auditor pipeline against it *offline*, without re-executing
the parent agent.

The driver directly orchestrates AgentSession-based child spawns via
StandaloneChildRunner, managing cadence/windowing/cumulative-state
threading internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import AgentMessage, AssistantMessage

from llmharness.atom import CumulativeAuditState
from llmharness.schema import Reminder

from .offline import InMemorySink, StandaloneChildRunner
from .runner import AuditorSettings, ExtractorSettings

__all__ = [
    "AuditorSettings",
    "ExtractorSettings",
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
    extractor_settings: ExtractorSettings,
    auditor_settings: AuditorSettings,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    enable_auditor: bool = True,
    stop_on_first_surface: bool = True,
    sidecar_path: Path | None = None,
    sink: InMemorySink | None = None,
    child: StandaloneChildRunner | None = None,
    seed_cumulative: CumulativeAuditState | None = None,
    start_turn: int = 1,
    skip_extractor: bool = False,
    trigger_registry: Any | None = None,
    trace_id: str | None = None,
) -> OfflineRunResult:
    """Replay the cognitive-audit pipeline over a captured trajectory.

    This is a simplified version that drives extractor + auditor children
    over the trajectory, managing cadence and cumulative state.

    Note: This is a compatibility shim. The full HarnessRunner machinery
    was removed; this function provides the same external contract with
    simpler internals. Complex features (trigger_registry, sidecar writing)
    are reduced to their essential behavior.
    """
    _ = trigger_registry  # Not used in simplified version
    _ = sidecar_path  # Sidecar writing removed from offline path

    resolved_trace_id = trace_id if trace_id is not None else session_id
    cumulative = seed_cumulative if seed_cumulative is not None else CumulativeAuditState.fresh()
    sink_used = sink if sink is not None else InMemorySink()
    child_used = child if child is not None else StandaloneChildRunner(
        cwd,
        parent_session_id=session_id,
        trace_id=resolved_trace_id,
    )

    all_steps: list[dict[str, Any]] = []
    surfaces: list[SurfaceFiring] = []
    reminder: Reminder | None = None

    from llmharness.agents.extractor.tools import ExtractionState
    from llmharness.agents.extractor.prompt import load_extractor_prompt
    from llmharness.agents.auditor.prompt import build_auditor_system_prompt
    from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
    from llmharness.schema import Verdict

    for turn_number, prefix_len in enumerate(_turn_end_prefix_lengths(messages), start=1):
        if prefix_len < start_turn:
            continue

        prefix = messages[:prefix_len]
        turn_count = turn_number
        step_result: dict[str, Any] = {
            "turn_count": turn_count,
            "prefix_len": prefix_len,
            "surfaced_reminder": None,
            "extractor_record": None,
            "auditor_record": None,
        }

        auditor_due = enable_auditor and (turn_count % audit_interval) == 0
        extractor_due = (not skip_extractor) and ((turn_count % extractor_interval) == 0 or auditor_due)

        # --- Extractor ---
        if extractor_due:
            from llmharness.atom import _prepare_extractor_data, _render_message_text, _serialize_trajectory

            data = _prepare_extractor_data(prefix, cumulative, None)
            if data is not None:
                state = ExtractionState()
                nxt = data.get("next_event_id")
                if isinstance(nxt, int) and nxt >= 1:
                    state.next_event_id = nxt
                for k, v in (data.get("turn_texts") or {}).items():
                    try:
                        state.turn_texts[int(k)] = str(v)
                    except (TypeError, ValueError):
                        pass

                recent_graph_raw = data.get("recent_graph") or []
                from llmharness.schema import Event, Edge
                recent_events: list[Event] = []
                for entry in recent_graph_raw:
                    if isinstance(entry, dict):
                        try:
                            recent_events.append(Event.from_dict(entry))
                        except (KeyError, ValueError, TypeError):
                            pass
                state.recent_graph = tuple(recent_events)
                state.recent_graph_dict = {e.id: e for e in recent_events}

                recent_edges_raw = data.get("recent_edges") or []
                recent_edges: list[Edge] = []
                for entry in recent_edges_raw:
                    if isinstance(entry, dict):
                        try:
                            recent_edges.append(Edge.from_dict(entry))
                        except (KeyError, ValueError, TypeError):
                            pass
                state.recent_edges_dict = {(ed.src, ed.dst, ed.kind.value): ed for ed in recent_edges}
                state._refold()

                prompt_text = extractor_settings.base_prompt or load_extractor_prompt("default")
                try:
                    ok, raw_blocks = await child_used.run_extractor(
                        state=state,
                        prompt_text=prompt_text,
                        provider=provider,
                        payload=data,
                        turn_window=list(range(max(cumulative.cursor_last_turn_index + 1, 0), prefix_len)),
                        tool_call_budget=extractor_settings.tool_call_budget,
                    )
                    state.salvage()
                    if state.pending_ops:
                        firing_id = cumulative.firing_id_counter
                        cumulative.absorb_extractor_firing(
                            firing_ops=state.pending_ops,
                            firing_cursor=data["window_hi"],
                            firing_id=firing_id,
                        )
                except Exception:
                    pass

        # --- Auditor ---
        if auditor_due:
            events, edges, phases = cumulative.graph_view()
            prompt_text = auditor_settings.base_prompt or "You are the cognitive-audit auditor."
            aud_prompt = build_auditor_system_prompt(
                events=events,
                edges=edges,
                phases=phases,
                findings=[],
                check_errors={},
                continuation_notes=list(cumulative.last_continuation_notes),
                summary_threshold=auditor_settings.summary_threshold,
                base_prompt=prompt_text,
            )
            tools_config: dict[str, Any] = {
                "tools": list(auditor_settings.tools or (SUBMIT_VERDICT_TOOL_NAME,))
            }
            aud_result = await child_used.run_auditor(
                prompt_text=aud_prompt,
                tools_config=tools_config,
                provider=provider,
                graph_events=list(events),
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
