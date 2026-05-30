"""Replay-fork driver: run a fork strategy over recorded baselines, judge.

The driver is strategy-agnostic: it delegates fork logic to a
:class:`~.strategy.ForkStrategy`, then judges the result and formats it
into a :class:`ReplayCaseResult`.  Adding a new ablation means writing a
new strategy, not editing this file.

Per case:

1. The strategy receives the recorded backbone and produces a
   :class:`~.strategy.ForkResult` -- the leaf continuation's response plus
   metadata about what was injected and where.
2. When the strategy fires (``ForkResult.fired``), the driver judges the
   leaf response against ground truth.  When it does not fire, the
   intervention outcome equals the control outcome (no re-judge needed).
3. Results are written to a sink in completion order.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .case_source import CaseSource, ReplayCase
from .judge import LeafJudge
from .strategy import ForkStrategy

_logger = logging.getLogger(__name__)

__all__ = [
    "JsonlResultSink",
    "ReplayCaseResult",
    "ReplayForkDriver",
    "ReplaySummary",
    "ResultSink",
]


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

    def __init__(self, path: str | os.PathLike[str], *, append: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a" if append else "w", encoding="utf-8")

    def write(self, result: ReplayCaseResult) -> None:
        import json

        self._fh.write(json.dumps(asdict(result), ensure_ascii=False, default=str))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class ReplayForkDriver:
    """Drives the replay-fork experiment over a stream of recorded cases.

    The driver is strategy-agnostic: it delegates fork logic to the
    :class:`~.strategy.ForkStrategy` and handles judging, error isolation,
    concurrency, and result formatting.
    """

    def __init__(
        self,
        *,
        agent: Any,
        strategy: ForkStrategy,
        judge: LeafJudge,
        scenario: str = "rca:baseline",
    ) -> None:
        self._agent = agent
        self._strategy = strategy
        self._judge = judge
        self._scenario = scenario

    async def run_case(self, case: ReplayCase) -> ReplayCaseResult:
        fork = await self._strategy.execute(
            case,
            agent=self._agent,
            scenario=self._scenario,
        )

        if not fork.fired:
            # Strategy did not intervene: intervention outcome = control.
            return ReplayCaseResult(
                case_id=case.case_id,
                fired=False,
                n_interventions=0,
                control_correct=case.control_correct,
                intervene_correct=case.control_correct,
                intervene_response=fork.response,
                intervention_path=[],
                leaf_session_log_id=None,
            )

        outcome = await self._judge.judge(
            agent_output_json=fork.response,
            data_dir=case.data_dir,
            case_id=case.case_id,
        )
        return ReplayCaseResult(
            case_id=case.case_id,
            fired=True,
            n_interventions=fork.n_interventions,
            control_correct=case.control_correct,
            intervene_correct=outcome.correct,
            intervene_response=fork.response,
            intervention_path=fork.intervention_path,
            leaf_session_log_id=fork.leaf_session_log_id,
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
