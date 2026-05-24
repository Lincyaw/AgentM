"""Offline driver: replay the audit pipeline over a bare trajectory.

The motivating use case (cf. ``.claude/designs/harness-runner.md`` §3):
take a captured baseline run's ``final_messages`` and re-run the
extractor + auditor pipeline against it *offline*, without re-executing
the parent agent. Produces a fresh sidecar identical-in-shape to the
one a live ``rca:harness.sync.extractor5`` run would have emitted on
the same trajectory.

The whole driver is just ``HarnessRunner`` plugged with the offline
seams; the cadence + windowing + cumulative-state threading all live
inside the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import AgentMessage

from ..audit._offline_seams import InMemorySink, StandaloneChildRunner
from ..audit._runner import (
    AuditorSettings,
    CumulativeAuditState,
    ExtractorSettings,
    HarnessRunner,
    SidecarWriter,
    StepResult,
)
from ..schema import Reminder

__all__ = [
    "AuditorSettings",
    "ExtractorSettings",
    "OfflineRunResult",
    "replay_pipeline_over_trajectory",
]


@dataclass
class OfflineRunResult:
    """Outcome of one :func:`replay_pipeline_over_trajectory` invocation."""

    reminder: Reminder | None
    state: CumulativeAuditState
    sidecar_path: Path | None
    all_step_results: list[StepResult] = field(default_factory=list)


async def replay_pipeline_over_trajectory(
    *,
    messages: list[AgentMessage],
    cwd: str,
    root_session_id: str,
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
) -> OfflineRunResult:
    """Replay the cognitive-audit pipeline over a captured trajectory.

    Drives :meth:`HarnessRunner.on_trajectory_progress` from
    ``turn_count=start_turn`` up through ``len(messages)`` so the
    runner's cadence (``turn_count % extractor_interval == 0`` etc.)
    fires identically to the live ``_on_turn_end`` path. Single source
    of truth for the cadence decision -- this function never duplicates
    it.

    ``stop_on_first_surface`` halts the run on the first auditor firing
    that surfaces a reminder so a chained-fork driver can re-seed.

    ``seed_cumulative`` lets a chained-fork driver thread state from a
    previous segment into the next so the auditor sees prior verdicts
    and the cumulative graph instead of restarting blank. When ``None``
    (the default), a fresh :class:`CumulativeAuditState` is used.

    ``start_turn`` shifts the inclusive lower bound of the cadence
    walk. Used by chained-fork replays where segment ``k+1`` begins
    one turn past segment ``k``'s surfaced-reminder boundary; previous
    turns have already been replayed under the parent segment's
    sidecar.

    ``sink`` / ``child`` are exposed for tests that want to inject
    stubs while still exercising the real :class:`HarnessRunner`.
    """
    cumulative = (
        seed_cumulative if seed_cumulative is not None else CumulativeAuditState.fresh()
    )
    sink_used = sink if sink is not None else InMemorySink()
    child_used = child if child is not None else StandaloneChildRunner(cwd)
    sidecar = SidecarWriter(sidecar_path) if sidecar_path is not None else None

    runner = HarnessRunner(
        cumulative=cumulative,
        child=child_used,
        sink=sink_used,
        sidecar=sidecar,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=extractor_interval,
        audit_interval=audit_interval,
        enable_auditor=enable_auditor,
        root_session_id=root_session_id,
        provider_extractor=provider,
        provider_auditor=provider,
        audit_registry=None,
    )

    all_steps: list[StepResult] = []
    reminder: Reminder | None = None
    for turn_count in range(max(1, start_turn), len(messages) + 1):
        step = await runner.on_trajectory_progress(
            messages[:turn_count], turn_count=turn_count
        )
        all_steps.append(step)
        if stop_on_first_surface and step.surfaced_reminder is not None:
            reminder = step.surfaced_reminder
            break

    return OfflineRunResult(
        reminder=reminder,
        state=cumulative,
        sidecar_path=sidecar_path,
        all_step_results=all_steps,
    )
