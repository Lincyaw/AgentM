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

from agentm.core.abi.messages import AgentMessage, AssistantMessage

from ..audit.runner import (
    AuditorSettings,
    CumulativeAuditState,
    ExtractorSettings,
    HarnessRunner,
    SidecarWriter,
    StepResult,
)
from ..audit.seams.offline import InMemorySink, StandaloneChildRunner
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


def _turn_end_prefix_lengths(messages: list[AgentMessage]) -> list[int]:
    """Message-prefix length at the end of each agent *turn*.

    A turn == one assistant generation. The live ``TurnEndEvent`` fires
    exactly once per turn, so the offline cadence must count turns too —
    not raw messages — to fire identically (design acceptance invariant
    #1, live ≡ offline). Turn ``k`` starts at the ``k``-th
    :class:`AssistantMessage` and ends just before the ``(k+1)``-th (or
    at the end of the list); the returned ``list[k-1]`` is
    ``len(messages[:end])`` for turn ``k`` — i.e. the message count
    through that turn's tool results.

    Returning prefix *lengths* (not turn numbers) lets the caller keep
    feeding the runner the full message prefix, so the recorded
    ``turn_index = len(prefix) - 1`` stays a message index exactly as in
    the live path (the chained-fork fork-slicing + every sidecar consumer
    depend on that being a message index, not a turn ordinal).
    """
    assistant_idxs = [
        i for i, m in enumerate(messages) if isinstance(m, AssistantMessage)
    ]
    if not assistant_idxs:
        # No assistant turn (e.g. a bare seeded prefix). Step once over the
        # whole list rather than zero times, so a degenerate trajectory
        # still threads cumulative state.
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

    Drives :meth:`HarnessRunner.on_trajectory_progress` once per agent
    *turn* (= per :class:`AssistantMessage`, mirroring the live
    one-``TurnEndEvent``-per-turn cadence), passing ``turn_count`` = the
    turn ordinal so the runner's cadence (``turn_count %
    extractor_interval == 0`` etc.) fires identically to the live
    ``_on_turn_end`` path. The runner stays the single source of truth
    for the cadence decision -- this function only supplies the
    per-turn ``turn_count`` and the matching message prefix.

    ``start_turn`` is a **message-index** lower bound (chained-fork
    passes the parent fork message-index plus a cadence interval): turns
    whose trajectory has not yet reached that message are skipped, their
    cumulative state arriving via ``seed_cumulative`` instead.

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
    for turn_number, prefix_len in enumerate(
        _turn_end_prefix_lengths(messages), start=1
    ):
        # ``start_turn`` is a message-index floor; skip turns whose
        # trajectory ends before it (already replayed under the parent
        # segment / arriving via ``seed_cumulative``).
        if prefix_len < start_turn:
            continue
        step = await runner.on_trajectory_progress(
            messages[:prefix_len], turn_count=turn_number
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
