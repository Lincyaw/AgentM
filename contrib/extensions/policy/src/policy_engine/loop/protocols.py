"""The two seams between the loop and a benchmark.

The loop is the same everywhere: find graded failures, work out why each one
diverged, generalise a check from it, gate it, measure it counterfactually, keep
what survives. None of that knows what a verifier writes to disk, how a task is
re-run, or what a metric is called.

Three things do differ per benchmark, and they are the whole of these
protocols:

* **discovery** -- where attempts live, which of them failed, and what evidence
  exists for why. A benchmark knows its own file layout; the loop does not.
* **the environment** -- whether the machine an attempt ran on can be brought
  back for inspection, and how.
* **re-running** -- how to put an agent back at its decision point with a
  message injected, and how to grade the result.

A benchmark adapter implements these. Everything else in this package depends
on the protocols and the records, never on an adapter.

The environment seam is separate from the benchmark seam on purpose: they are
orthogonal. One benchmark can run on a sandbox cluster or in a local container,
and one cluster can host several benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agentm.core.abi import EnvironmentOperations, ResourceWriter

from .contracts import FailureCase, Metrics


@dataclass(slots=True)
class RestoredEnvironment:
    """A live machine, and an honest account of what is on it.

    ``description`` is written by the adapter and rendered into the prompt
    verbatim. It is not decoration. An attempt's own session and a fresh session
    from the same image look identical through ``EnvironmentOperations`` and
    support different conclusions: on the second, anything the attempt did
    outside its patch is absent, so "the service was not running" means nothing.
    Whoever reads the environment has to be told which one they are on, and the
    loop cannot tell them because only the adapter knows.

    ``writer`` travels with ``operations`` because they must describe the same
    filesystem. The file tools bind to the writer and the shell to the
    operations; supplying only one gives an agent two filesystems and no way to
    notice.
    """

    operations: EnvironmentOperations
    writer: ResourceWriter | None = field(default=None, kw_only=True)
    work_dir: str = field(default="/", kw_only=True)
    description: str = field(default="", kw_only=True)


class CaseSource(Protocol):
    """A benchmark's read side."""

    @property
    def name(self) -> str:
        """Short identifier, as given to ``--bench``."""
        ...

    def discover(self, batch: str, *, only: str = "") -> list[FailureCase]:
        """Every graded attempt in ``batch`` that did not pass.

        The returned cases must be self-describing: whatever a diagnoser needs
        to read is listed in ``evidence``, and whatever a replay backend needs to
        re-run the attempt is in ``backend_ref``. A stage is allowed to be handed
        a hand-edited ``cases.json`` and nothing else, so nothing may be left
        implicit for the adapter to supply later.
        """
        ...

    async def open_environment(self, case: FailureCase) -> RestoredEnvironment | None:
        """A machine with this task's repository on it, or ``None``.

        ``None`` is a normal answer, and so is an environment that is not the
        attempt's own -- see ``RestoredEnvironment.description``, which is why
        that field exists.
        """
        ...


class ReplayBackend(Protocol):
    """A benchmark's write side: the fitness measurement."""

    @property
    def name(self) -> str: ...

    def rerun(
        self,
        case: FailureCase,
        *,
        injected: str,
        label: str,
        resume_at: int = -1,
    ) -> ReplayOutcome:
        """Re-run one attempt from ``resume_at`` with ``injected`` waiting, then
        grade it.

        ``resume_at`` is a position in the recorded conversation; ``-1`` means
        the end. Resuming from the middle is the point of the exercise: the
        decisions that decide these outcomes are taken in the first half of a
        run, and a message delivered at the end arrives after the work that
        would have to be undone. A backend that can only resume at the end
        should say so by ignoring this and reporting what it did.

        ``label`` is a stable name for the run, used for whatever the backend
        writes to disk; the loop guarantees it is unique per measurement.

        Must not raise -- including while writing its own scratch files, which
        is where this was got wrong once: a full disk killed a whole batch
        mid-run. A measurement that did not happen is reported as a
        ``lost_reason``, never as an unchanged score: an infrastructure failure
        and a check that did nothing look identical in the numbers and mean
        opposite things.

        A backend that cannot resume where it was asked reports where it did,
        through ``ReplayOutcome.resumed_at``.
        """
        ...


@dataclass(slots=True)
class ReplayOutcome:
    """What one re-run produced.

    ``lost_reason`` non-empty means no measurement exists. The metrics are then
    meaningless and must not be compared.

    ``resumed_at`` is where the backend actually restarted the attempt, which is
    not always where it was asked to. A backend that cannot reach the requested
    turn may fall back to the end of the recording, and that is a different
    experiment: an interruption delivered after the work is finished measures
    something else entirely. Recording it here keeps the two from sitting side
    by side in one artifact looking identical.
    """

    metrics: Metrics
    lost_reason: str = field(default="", kw_only=True)
    artifact_dir: str = field(default="", kw_only=True)
    resumed_at: int = field(default=-1, kw_only=True)
    #: What an independent reviewer said during the run, when the scenario ran
    #: one. Carried with the measurement because the two are only meaningful
    #: together: whether a review found what grading punished is a question
    #: about this attempt, and asking it later means finding a session again
    #: from a job directory nobody kept.
    review_report: str = field(default="", kw_only=True)


__all__ = [
    "CaseSource",
    "ReplayBackend",
    "ReplayOutcome",
    "RestoredEnvironment",
]
