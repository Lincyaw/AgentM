"""Replay-fork driver: re-audit a recorded baseline, fork on surface, judge.

Per case:

1. The recorded baseline trajectory is the control backbone -- the main
   agent is *not* re-run for control (its answer is already known).
2. The fork-tree engine re-runs extractor + auditor over that backbone with
   the harness model. ``max_surfaces_per_node=1`` makes it a greedy spine:
   the first reminder forks a continuation, which is itself re-audited, and
   so on up to ``max_depth`` -- the offline analogue of the live "inject
   when the auditor fires, then keep going" behaviour.
3. Each fork continuation is a real main-agent rollout (the agent model)
   seeded with the surfaced reminder, started from the parent prefix.
4. The deepest continuation's submission is judged against ground truth.
   When the auditor never fires, there is no intervention and the case's
   intervene outcome is its control outcome.

The harness model and the agent model are independent: the harness provider
is passed to the engine, the agent provider lives on the ``AgentMAgent``.
The driver depends only on :class:`~.case_source.ReplayCase`, the engine,
and the :class:`~.judge.LeafJudge` -- never on where the baseline was
stored.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import AgentMessage

from .case_source import CaseSource, ReplayCase
from .judge import LeafJudge

_logger = logging.getLogger(__name__)

__all__ = [
    "JsonlResultSink",
    "ReplayCaseResult",
    "ReplayForkDriver",
    "ReplaySummary",
    "ResultSink",
]


@dataclass(frozen=True)
class _RecordedBackbone:
    """A control backbone served from a recording (no agent run).

    Structurally satisfies the engine's ``SessionPayload`` protocol
    (``session_log_id`` + ``final_messages``), so the fork-tree audits the
    recorded trajectory exactly as it would a freshly produced one.
    """

    session_log_id: str
    final_messages: list[AgentMessage]


@dataclass
class ReplayCaseResult:
    """Per-case outcome of the replay-fork experiment."""

    case_id: str
    fired: bool
    n_interventions: int
    control_correct: bool | None
    intervene_correct: bool | None
    intervene_response: str | None = None
    intervention_path: list[str] = field(default_factory=list)
    leaf_session_log_id: str | None = None
    judge_detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def flipped_to_correct(self) -> bool:
        return self.control_correct is False and self.intervene_correct is True

    @property
    def flipped_to_wrong(self) -> bool:
        return self.control_correct is True and self.intervene_correct is False


@dataclass
class ReplaySummary:
    """Aggregate over a replay-fork run."""

    total: int = 0
    fired: int = 0
    errored: int = 0
    control_correct: int = 0
    intervene_correct: int = 0
    flips_to_correct: int = 0
    flips_to_wrong: int = 0

    def add(self, r: ReplayCaseResult) -> None:
        self.total += 1
        if r.error:
            self.errored += 1
        if r.fired:
            self.fired += 1
        if r.control_correct:
            self.control_correct += 1
        if r.intervene_correct:
            self.intervene_correct += 1
        if r.flipped_to_correct:
            self.flips_to_correct += 1
        if r.flipped_to_wrong:
            self.flips_to_wrong += 1

    def format(self) -> str:
        def pct(n: int) -> str:
            return f"{100.0 * n / self.total:.1f}%" if self.total else "n/a"

        return (
            f"cases={self.total} errored={self.errored} auditor_fired={self.fired}\n"
            f"control  correct: {self.control_correct} ({pct(self.control_correct)})\n"
            f"intervene correct: {self.intervene_correct} ({pct(self.intervene_correct)})\n"
            f"flips  W->R(helped): {self.flips_to_correct}   "
            f"R->W(harmed): {self.flips_to_wrong}"
        )


class ResultSink:
    """Sink for per-case results. Default base is a no-op; see subclasses."""

    def write(self, result: ReplayCaseResult) -> None:  # pragma: no cover - interface
        ...

    def close(self) -> None:  # pragma: no cover - interface
        ...


class JsonlResultSink(ResultSink):
    """Append one JSON line per case result."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")

    def write(self, result: ReplayCaseResult) -> None:
        import json

        self._fh.write(json.dumps(asdict(result), ensure_ascii=False, default=str))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class ReplayForkDriver:
    """Drives the replay-fork experiment over a stream of recorded cases."""

    def __init__(
        self,
        *,
        agent: Any,
        harness_provider: tuple[str, dict[str, Any]],
        judge: LeafJudge,
        scenario: str = "rca:baseline",
        max_depth: int = 3,
        extractor_interval: int = 5,
        audit_interval: int = 5,
        cwd: str | None = None,
        sidecar_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._agent = agent
        self._harness_provider = harness_provider
        self._judge = judge
        self._scenario = scenario
        self._max_depth = max_depth
        self._extractor_interval = extractor_interval
        self._audit_interval = audit_interval
        self._cwd = cwd or os.getcwd()
        self._sidecar_dir = Path(sidecar_dir) if sidecar_dir is not None else None

    async def run_case(self, case: ReplayCase) -> ReplayCaseResult:
        from llmharness import (
            AuditorSettings,
            ExtractorSettings,
            SessionPayload,
            run_fork_tree_experiment,
        )

        session_runs: dict[str, Any] = {}
        control_id = f"{case.case_id}-control"

        async def factory(
            *,
            initial_messages: list[Any] | None,
            seed_reminder_text: str | None,
        ) -> SessionPayload:
            if initial_messages is None:
                # Control backbone: serve the recording, never re-run the agent.
                return _RecordedBackbone(  # type: ignore[return-value]
                    session_log_id=control_id,
                    final_messages=case.backbone_messages,
                )
            run = await self._agent._execute_session(
                incident=None,
                data_dir=case.data_dir,
                scenario=self._scenario,
                initial_messages=initial_messages,
                seed_reminder_text=seed_reminder_text,
            )
            session_runs[run.session_log_id] = run
            return run  # type: ignore[return-value]  # structural SessionPayload match

        out_path = None
        if self._sidecar_dir is not None:
            out_path = self._sidecar_dir / f"{case.case_id}.chained.jsonl"

        experiment = await run_fork_tree_experiment(
            session_factory=factory,
            cwd=self._cwd,
            provider=self._harness_provider,
            extractor_settings=ExtractorSettings.default(),
            auditor_settings=AuditorSettings.default(),
            extractor_interval=self._extractor_interval,
            audit_interval=self._audit_interval,
            max_depth=self._max_depth,
            max_surfaces_per_node=1,
            out_path=out_path,
        )

        fork_nodes = [n for n in experiment.nodes if n.parent_id is not None]
        if not fork_nodes:
            # Auditor never surfaced: no intervention possible -> intervene
            # outcome is the control outcome (no re-judge needed).
            return ReplayCaseResult(
                case_id=case.case_id,
                fired=False,
                n_interventions=0,
                control_correct=case.control_correct,
                intervene_correct=case.control_correct,
                intervene_response=case.control_response,
                intervention_path=[],
                leaf_session_log_id=None,
            )

        leaf = max(fork_nodes, key=lambda n: n.depth)
        leaf_run = session_runs.get(leaf.backbone_session_id)
        response = getattr(leaf_run, "response", None)
        outcome = await self._judge.judge(
            agent_output_json=response,
            data_dir=case.data_dir,
            case_id=case.case_id,
        )
        return ReplayCaseResult(
            case_id=case.case_id,
            fired=True,
            n_interventions=leaf.depth,
            control_correct=case.control_correct,
            intervene_correct=outcome.correct,
            intervene_response=response,
            intervention_path=list(leaf.path),
            leaf_session_log_id=leaf.backbone_session_id,
            judge_detail=outcome.detail,
            error=outcome.error,
        )

    async def _run_case_guarded(self, case: ReplayCase) -> ReplayCaseResult:
        try:
            return await self.run_case(case)
        except Exception as exc:  # noqa: BLE001 -- isolate per-case failures
            _logger.exception("replay-fork: case %s failed", case.case_id)
            return ReplayCaseResult(
                case_id=case.case_id,
                fired=False,
                n_interventions=0,
                control_correct=case.control_correct,
                intervene_correct=None,
                error=f"{type(exc).__name__}: {exc!s:.300}",
            )

    async def run(
        self,
        source: CaseSource,
        sink: ResultSink | None = None,
        *,
        max_concurrency: int = 1,
    ) -> ReplaySummary:
        """Run every case from ``source``, up to ``max_concurrency`` at once.

        Per-case work (offline audit + continuations) is independent, so the
        only shared state is the sink and the summary, both mutated on the
        single event loop after each case completes -- no locking needed.
        Results are written / counted in completion order; a per-case failure
        is isolated and recorded, never aborting the batch.
        """
        summary = ReplaySummary()
        cases: list[ReplayCase] = list(source.cases())
        total = len(cases)
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def _bounded(case: ReplayCase) -> ReplayCaseResult:
            async with sem:
                return await self._run_case_guarded(case)

        tasks = [asyncio.create_task(_bounded(c)) for c in cases]
        done = 0
        for fut in asyncio.as_completed(tasks):
            result = await fut
            done += 1
            if sink is not None:
                sink.write(result)
            summary.add(result)
            _logger.info(
                "replay-fork [%d/%d] case=%s fired=%s n_iv=%d control=%s intervene=%s%s",
                done,
                total,
                result.case_id,
                result.fired,
                result.n_interventions,
                result.control_correct,
                result.intervene_correct,
                f" ERROR={result.error}" if result.error else "",
            )
        return summary
