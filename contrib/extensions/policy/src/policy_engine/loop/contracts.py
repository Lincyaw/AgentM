# code-health: ignore-file[AM025] -- JSON arriving from disk and from model
# tool calls is untyped by construction; every isinstance here is at that
# boundary, converting it into the typed records below exactly once.
"""The protocol between loop stages.

Each stage reads one artifact and writes the next, so these records are the
whole interface: a stage can be re-run against a hand-edited file, and a stage
can be replaced without touching its neighbours. They are also the agents'
output schemas -- ``Diagnosis`` and ``Candidate`` are what the diagnoser and
abstractor return by tool call, so the model has no way to finish except by
filling one in.

Everything is plain JSON on disk. ``AGENTM_HOME/policy-loop/<run-id>/`` holds
one file per stage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from agentm.core.abi import JsonValue

# -- json helpers -------------------------------------------------------------


def json_str(raw: Mapping[str, JsonValue], key: str, default: str = "") -> str:
    """One string out of untyped JSON, or ``default``.

    Public because benchmark adapters read the same untyped JSON at the same
    boundary, and a second copy of this is where the ``bool`` subtlety below
    gets fixed in one place only.
    """
    value = raw.get(key, default)
    return value if isinstance(value, str) else default


def json_int(raw: Mapping[str, JsonValue], key: str, default: int = 0) -> int:
    """One int out of untyped JSON. ``bool`` is an ``int`` in Python and is not
    one here: ``True`` arriving where a turn index belongs is bad data."""
    value = raw.get(key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


_str = json_str
_int = json_int


def _bool(raw: Mapping[str, JsonValue], key: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    return value if isinstance(value, bool) else default


def _objects(raw: Mapping[str, JsonValue], key: str) -> list[Mapping[str, JsonValue]]:
    value = raw.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mapping(raw: Mapping[str, JsonValue], key: str) -> Mapping[str, JsonValue]:
    value = raw.get(key)
    return value if isinstance(value, Mapping) else {}


def _strings(raw: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


# -- shared -------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Metrics:
    """What a benchmark scored, by its own names.

    Not three fixed fields: benchmarks disagree about what they measure, and
    even within one, tasks differ in which measurements exist. A missing metric
    means the task has none of that kind -- one dev-set task has no functional
    suite, another no user stories -- which is a different thing from scoring
    zero, and collapsing the two turns "not applicable" into "failed".

    ``primary`` names the metric that decides whether the task was solved. The
    others still matter: a check can move a task from a third of its checks
    passing to all of them without flipping the primary, and that is progress
    worth keeping.
    """

    values: Mapping[str, float | None] = field(default_factory=dict)
    primary: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {"values": dict(self.values), "primary": self.primary}

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> Metrics:
        values = _mapping(raw, "values")
        parsed: dict[str, float | None] = {}
        for key, value in values.items():
            if value is None or isinstance(value, bool):
                parsed[key] = None
            elif isinstance(value, (int, float)):
                parsed[key] = float(value)
        return Metrics(values=parsed, primary=_str(raw, "primary"))

    def get(self, name: str) -> float | None:
        return self.values.get(name)

    @property
    def present(self) -> dict[str, float]:
        """Only the metrics this task actually has."""
        return {k: v for k, v in self.values.items() if v is not None}

    @property
    def is_pass(self) -> bool:
        """A full pass on every metric that exists. With no metric at all the
        attempt was lost to the harness, which is not a pass."""
        seen = self.present
        return bool(seen) and all(v >= 1.0 for v in seen.values())

    def minus(self, other: Metrics) -> Metrics:
        """Deltas, for metrics both sides have. A metric present on one side
        only cannot be compared and is dropped rather than guessed."""
        delta: dict[str, float | None] = {}
        for key, value in self.values.items():
            baseline = other.values.get(key)
            delta[key] = None if value is None or baseline is None else value - baseline
        return Metrics(values=delta, primary=self.primary or other.primary)

    @property
    def any_positive(self) -> bool:
        return any(v > 0 for v in self.present.values())

    @property
    def any_negative(self) -> bool:
        return any(v < 0 for v in self.present.values())

    def render(self) -> str:
        if not self.values:
            return "(no metrics)"
        return " ".join(
            f"{name}={'-' if value is None else format(value, 'g')}"
            for name, value in sorted(self.values.items())
        )


@dataclass(slots=True, frozen=True)
class Evidence:
    """A quote with where it came from, so a claim can be re-checked."""

    source: str = ""
    quote: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {"source": self.source, "quote": self.quote}

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> Evidence:
        return Evidence(source=_str(raw, "source"), quote=_str(raw, "quote"))


@dataclass(slots=True, frozen=True)
class EvidenceRef:
    """Somewhere a diagnoser should look, named by the benchmark that knows.

    Rendered into the prompt verbatim, so the loop never learns that a verifier
    writes ``runner_*.log`` or that stories have parameter files. ``note`` is
    where an adapter says what the thing is good for -- an assertion id can be a
    wrapper around a whole suite, and only the adapter knows that.
    """

    label: str = ""
    locator: str = ""
    note: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {"label": self.label, "locator": self.locator, "note": self.note}

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> EvidenceRef:
        return EvidenceRef(
            label=_str(raw, "label"),
            locator=_str(raw, "locator"),
            note=_str(raw, "note"),
        )


@dataclass(slots=True, frozen=True)
class Assertion:
    """One graded check that failed. ``kind`` separates the functional suite
    from the validation stories because they fail for different reasons and are
    reported in different files."""

    kind: str = "test"
    assertion_id: str = ""
    message: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {"kind": self.kind, "id": self.assertion_id, "message": self.message}

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> Assertion:
        return Assertion(
            kind=_str(raw, "kind", "test"),
            assertion_id=_str(raw, "id"),
            message=_str(raw, "message"),
        )


# -- stage 1: discovery -------------------------------------------------------


@dataclass(slots=True, frozen=True)
class FailureCase:
    """One graded attempt that did not pass.

    Self-describing on purpose. Whatever a diagnoser needs to read is listed in
    ``evidence``; whatever a replay backend needs to re-run the attempt is in
    ``backend_ref``, opaque to everything between. A stage may be handed a
    hand-edited ``cases.json`` and nothing else, so nothing here may depend on
    the adapter still being around to fill a gap in later.
    """

    case_id: str
    task_name: str
    #: The task as the agent received it.
    instruction: str = ""
    failing_assertions: tuple[Assertion, ...] = ()
    #: Where to look, in the benchmark's own terms.
    evidence: tuple[EvidenceRef, ...] = ()
    #: How this task's attempts compare, written by the adapter because judging
    #: whether an assertion id is fine-grained enough needs benchmark knowledge.
    #: Empty when there is only one attempt.
    cohort_note: str = ""
    metrics: Metrics = field(default_factory=Metrics)
    sibling_metrics: tuple[Metrics, ...] = ()
    #: Everything the replay backend needs and nobody else interprets: a
    #: workspace checkpoint, a conversation turn, a container digest -- whatever
    #: this benchmark's re-run takes.
    backend_ref: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "case_id": self.case_id,
            "task_name": self.task_name,
            "instruction": self.instruction,
            "failing_assertions": tuple(a.to_json() for a in self.failing_assertions),
            "evidence": tuple(e.to_json() for e in self.evidence),
            "cohort_note": self.cohort_note,
            "metrics": self.metrics.to_json(),
            "sibling_metrics": tuple(m.to_json() for m in self.sibling_metrics),
            "backend_ref": dict(self.backend_ref),
        }

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> FailureCase:
        return FailureCase(
            case_id=_str(raw, "case_id"),
            task_name=_str(raw, "task_name"),
            instruction=_str(raw, "instruction"),
            failing_assertions=tuple(
                Assertion.from_json(a) for a in _objects(raw, "failing_assertions")
            ),
            evidence=tuple(EvidenceRef.from_json(e) for e in _objects(raw, "evidence")),
            cohort_note=_str(raw, "cohort_note"),
            metrics=Metrics.from_json(_mapping(raw, "metrics")),
            sibling_metrics=tuple(
                Metrics.from_json(m) for m in _objects(raw, "sibling_metrics")
            ),
            backend_ref=_mapping(raw, "backend_ref"),
        )

    @property
    def replayable(self) -> bool:
        """Whether a backend has anything to re-run from."""
        return bool(self.backend_ref)


# -- stage 2: diagnose --------------------------------------------------------

#: Why the attempt is wrong. The last two exist so the loop can retire a case
#: instead of grinding on it: one dev-set task is graded against a
#: reference-internal parameter name the task never states, and another's graded
#: suite fails nondeterministically in code no patch touches.
CAUSE_CLASSES: tuple[str, ...] = (
    "misread_intent",
    "omitted",
    "wrong_implementation",
    "unverified_claim",
    "environment_self_break",
    "grading_artifact",
    "harness_nondeterminism",
)


@dataclass(slots=True, frozen=True)
class Diagnosis:
    """The diagnoser's answer: what grading wanted, what was built instead, and
    where the two parted."""

    diagnosis_id: str
    case_id: str
    task_name: str = ""
    required_behaviour: str = ""
    reference_idea: str = ""
    agent_idea: str = ""
    divergence_turn: int = -1
    divergence_quote: str = ""
    cause_class: str = ""
    mechanism: str = ""
    #: What was being chosen at ``divergence_turn``, in the agent's own terms.
    decision: str = ""
    #: The step not taken: a question the agent could have put to itself at that
    #: moment, and would have had to run something to answer.
    #:
    #: Phrased without hindsight, on purpose. "Did I miss the working-tree
    #: overlay" is the answer wearing a question mark -- it is only available to
    #: someone who already knows. "What happens to a path that is modified but
    #: still in the index" is available before the fact, and is what makes the
    #: lesson transfer to a task where the overlay is not the issue.
    unasked_question: str = ""
    #: What asking it would have produced, and what the agent believed instead.
    #:
    #: Load-bearing: it is what separates a question with an answer from a
    #: question that would have come out the same either way. Every case in the
    #: first batch had the agent consult something real and stop -- an in-repo
    #: test, a status code, a flag -- where what it consulted looked identical
    #: whether it was right or wrong. A question with no discriminating answer is
    #: not the missing step, however sensible it sounds.
    discriminating_answer: str = ""
    #: What carries to the next task in this same repository.
    #:
    #: Separate from ``mechanism`` because they are answers to different
    #: questions and the useful one is easy to skip. The mechanism is what went
    #: wrong here, in full detail, and it is what makes the diagnosis auditable.
    #: The lesson is the part that is still true when the file, the function and
    #: the requirement have all changed, and it is the only part with any value
    #: downstream.
    #:
    #: The scope is one repository, not all software. A repository has a
    #: prevailing way of doing things, and knowing it is genuinely useful to
    #: whoever works here next -- which is why a lesson may name a convention, a
    #: layer, a habit of this codebase. What it may not do is name this task's
    #: defect, because that is the answer rather than a lesson.
    lesson: str = ""
    reachable: bool = True
    reachable_rationale: str = ""
    evidence: tuple[Evidence, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "diagnosis_id": self.diagnosis_id,
            "case_id": self.case_id,
            "task_name": self.task_name,
            "required_behaviour": self.required_behaviour,
            "reference_idea": self.reference_idea,
            "agent_idea": self.agent_idea,
            "divergence_turn": self.divergence_turn,
            "divergence_quote": self.divergence_quote,
            "cause_class": self.cause_class,
            "mechanism": self.mechanism,
            "decision": self.decision,
            "unasked_question": self.unasked_question,
            "discriminating_answer": self.discriminating_answer,
            "lesson": self.lesson,
            "reachable": self.reachable,
            "reachable_rationale": self.reachable_rationale,
            "evidence": tuple(e.to_json() for e in self.evidence),
        }

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> Diagnosis:
        return Diagnosis(
            diagnosis_id=_str(raw, "diagnosis_id"),
            case_id=_str(raw, "case_id"),
            task_name=_str(raw, "task_name"),
            required_behaviour=_str(raw, "required_behaviour"),
            reference_idea=_str(raw, "reference_idea"),
            agent_idea=_str(raw, "agent_idea"),
            divergence_turn=_int(raw, "divergence_turn", -1),
            divergence_quote=_str(raw, "divergence_quote"),
            cause_class=_str(raw, "cause_class"),
            mechanism=_str(raw, "mechanism"),
            decision=_str(raw, "decision"),
            unasked_question=_str(raw, "unasked_question"),
            discriminating_answer=_str(raw, "discriminating_answer"),
            lesson=_str(raw, "lesson"),
            reachable=_bool(raw, "reachable", True),
            reachable_rationale=_str(raw, "reachable_rationale"),
            evidence=tuple(Evidence.from_json(e) for e in _objects(raw, "evidence")),
        )

    @property
    def worth_abstracting(self) -> bool:
        """A grading artifact or a flaky harness has no check to learn."""
        return self.reachable and self.cause_class not in {
            "grading_artifact",
            "harness_nondeterminism",
        }


# -- stage 3: abstract --------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Candidate:
    """A general check proposed from one diagnosis.

    ``observation`` is the load-bearing field. A check that can be satisfied by
    reasoning gets satisfied by reasoning: an offline A/B had the reviewing
    model produce a structured, confident endorsement of the very diagnosis that
    was wrong. A candidate that names nothing to run is rejected here, before it
    costs a replay.
    """

    candidate_id: str
    check: str = ""
    when_note: str = ""
    precondition_note: str = ""
    observation: str = ""
    checkpoint: str = "stop"
    dimension: str = ""
    from_task: str = ""
    from_session: str = ""
    from_diagnosis: str = ""
    #: The turn this was derived from: where the approach it questions became
    #: fixed. Carried because it is where the check has to arrive to be worth
    #: anything -- delivered at the end it lands after the work it would undo.
    from_turn: int = -1

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "candidate_id": self.candidate_id,
            "check": self.check,
            "when_note": self.when_note,
            "precondition_note": self.precondition_note,
            "observation": self.observation,
            "checkpoint": self.checkpoint,
            "dimension": self.dimension,
            "from_task": self.from_task,
            "from_session": self.from_session,
            "from_diagnosis": self.from_diagnosis,
            "from_turn": self.from_turn,
        }

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> Candidate:
        return Candidate(
            candidate_id=_str(raw, "candidate_id"),
            check=_str(raw, "check"),
            when_note=_str(raw, "when_note"),
            precondition_note=_str(raw, "precondition_note"),
            observation=_str(raw, "observation"),
            checkpoint=_str(raw, "checkpoint", "stop"),
            dimension=_str(raw, "dimension"),
            from_task=_str(raw, "from_task"),
            from_session=_str(raw, "from_session"),
            from_diagnosis=_str(raw, "from_diagnosis"),
            from_turn=_int(raw, "from_turn", -1),
        )

    def rejection(self) -> str:
        """Why this candidate must not proceed, or empty when it may."""
        if not self.check.strip():
            return "empty check"
        if not self.observation.strip():
            return "names no observation: satisfiable by reasoning alone"
        return ""


# -- stage 4: compile ---------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PredicateProposal:
    name: str = ""
    description: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {"name": self.name, "description": self.description}

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> PredicateProposal:
        return PredicateProposal(
            name=_str(raw, "name"), description=_str(raw, "description")
        )


@dataclass(slots=True, frozen=True)
class CompiledCandidate:
    """A candidate with a firing condition it can actually be gated on.

    ``trigger`` is a boolean expression over tagger predicates -- what the
    session *looks* like. ``precondition`` is SQL over the data plane -- whether
    the situation the check presupposes has arrived. Both must hold. Measured
    need: six of eleven firings of the validation-scope item landed in sessions
    that had run no test at all, so its stated premise was false.
    """

    candidate: Candidate
    item_id: str = ""
    trigger: str = "always"
    precondition: str = ""
    new_predicates: tuple[PredicateProposal, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "candidate": self.candidate.to_json(),
            "item_id": self.item_id,
            "trigger": self.trigger,
            "precondition": self.precondition,
            "new_predicates": tuple(p.to_json() for p in self.new_predicates),
        }

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> CompiledCandidate:
        return CompiledCandidate(
            candidate=Candidate.from_json(_mapping(raw, "candidate")),
            item_id=_str(raw, "item_id"),
            trigger=_str(raw, "trigger", "always"),
            precondition=_str(raw, "precondition"),
            new_predicates=tuple(
                PredicateProposal.from_json(p) for p in _objects(raw, "new_predicates")
            ),
        )


# -- stage 5: replay ----------------------------------------------------------

#: ``candidate`` injects the check; ``placebo`` injects a contentless nudge at
#: the same fork point. The placebo is not optional -- without it an improvement
#: cannot be attributed to the check's content rather than to the extra turn it
#: buys.
#:
#: There is no ``baseline`` arm. What the attempt already scored is read from
#: the recording, not re-run, so naming it here only invited a caller to pass it
#: and get a second placebo run wearing the wrong label.
ARMS: tuple[str, ...] = ("placebo", "candidate")

#: ``lost`` must never collapse into ``unchanged``. Four of six replays run by
#: hand today were lost to infrastructure, and scoring those as "no improvement"
#: would have retired working candidates.
OUTCOMES: tuple[str, ...] = ("improved", "unchanged", "regressed", "lost")


@dataclass(slots=True, frozen=True)
class ReplayMeasurement:
    """One counterfactual: the same attempt, from the same fork point, with and
    without the candidate."""

    candidate_id: str
    case_id: str
    arm: str
    outcome: str = "lost"
    lost_reason: str = ""
    #: Where the attempt was asked to resume, and where it actually did. They
    #: differ when a backend could not reach the requested turn, and the
    #: difference decides what the row means.
    requested_turn: int = -1
    resumed_turn: int = -1
    metrics_before: Metrics = field(default_factory=Metrics)
    metrics_after: Metrics = field(default_factory=Metrics)
    job_dir: str = ""

    @property
    def delta(self) -> Metrics:
        return self.metrics_after.minus(self.metrics_before)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "candidate_id": self.candidate_id,
            "case_id": self.case_id,
            "arm": self.arm,
            "outcome": self.outcome,
            "lost_reason": self.lost_reason,
            "requested_turn": self.requested_turn,
            "resumed_turn": self.resumed_turn,
            "metrics_before": self.metrics_before.to_json(),
            "metrics_after": self.metrics_after.to_json(),
            "delta": self.delta.to_json(),
            "job_dir": self.job_dir,
        }

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> ReplayMeasurement:
        return ReplayMeasurement(
            candidate_id=_str(raw, "candidate_id"),
            case_id=_str(raw, "case_id"),
            arm=_str(raw, "arm"),
            outcome=_str(raw, "outcome", "lost"),
            lost_reason=_str(raw, "lost_reason"),
            requested_turn=_int(raw, "requested_turn", -1),
            resumed_turn=_int(raw, "resumed_turn", -1),
            metrics_before=Metrics.from_json(_mapping(raw, "metrics_before")),
            metrics_after=Metrics.from_json(_mapping(raw, "metrics_after")),
            job_dir=_str(raw, "job_dir"),
        )


# -- stage 6: select ----------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Verdict:
    candidate_id: str
    accepted: bool = False
    reason: str = ""
    improved_cases: tuple[str, ...] = ()
    regressed_cases: tuple[str, ...] = ()
    lost_cases: tuple[str, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "candidate_id": self.candidate_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "improved_cases": self.improved_cases,
            "regressed_cases": self.regressed_cases,
            "lost_cases": self.lost_cases,
        }

    @staticmethod
    def from_json(raw: Mapping[str, JsonValue]) -> Verdict:
        return Verdict(
            candidate_id=_str(raw, "candidate_id"),
            accepted=_bool(raw, "accepted"),
            reason=_str(raw, "reason"),
            improved_cases=_strings(raw, "improved_cases"),
            regressed_cases=_strings(raw, "regressed_cases"),
            lost_cases=_strings(raw, "lost_cases"),
        )


# -- artifact io --------------------------------------------------------------


def write_artifact(path: Path, records: Sequence[object]) -> None:
    """One stage's output. Sorted keys and a trailing newline so two runs of the
    same stage diff cleanly -- the standalone-equals-chained check depends on
    byte equality."""
    payload = [_encode(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_artifact(path: Path) -> list[Mapping[str, JsonValue]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{path}: expected a JSON list of records")
    return [item for item in raw if isinstance(item, Mapping)]


def _encode(record: object) -> JsonValue:
    to_json = getattr(record, "to_json", None)  # code-health: ignore[AM021]
    if to_json is None:
        raise TypeError(f"{type(record).__name__} has no to_json")
    encoded = to_json()
    if not isinstance(encoded, Mapping):
        raise TypeError(f"{type(record).__name__}.to_json did not return a mapping")
    return dict(encoded)


__all__ = [
    "ARMS",
    "CAUSE_CLASSES",
    "OUTCOMES",
    "Assertion",
    "Candidate",
    "CompiledCandidate",
    "Diagnosis",
    "Evidence",
    "EvidenceRef",
    "FailureCase",
    "Metrics",
    "PredicateProposal",
    "ReplayMeasurement",
    "Verdict",
    "json_int",
    "json_str",
    "read_artifact",
    "write_artifact",
]
