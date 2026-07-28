"""The protocol between loop stages.

Each stage reads one artifact and writes the next, so these records are the
whole interface: a stage can be re-run against a hand-edited file, and a stage
can be replaced without touching its neighbours.

Every shape is written once, as a pydantic model, and serves three readers
from that one definition: the artifact on disk (``to_json``/``from_json``),
the agent's tool schema (the ``*Payload`` classes — ``FunctionTool`` converts
a model class to JSON Schema, and the runner validates the call against it, so
a malformed payload is a retry rather than an empty string downstream), and
the typed record the next stage receives. The previous arrangement wrote each
shape three times by hand, and the three had already begun to disagree.

A ``*Payload`` class holds exactly the fields the model fills in; the record
that extends it adds what the loop assigns — ids, provenance, turn indices.
Parsing is validating: an artifact missing a required field fails loudly at
load rather than defaulting to ``""`` and failing quietly three stages later.

Everything is plain JSON on disk. ``AGENTM_HOME/policy-loop/<run-id>/`` holds
one file per stage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field
from pydantic import JsonValue as PydanticJsonValue

from agentm.core.abi import JsonValue

# -- json helpers --------------------------------------------------------------
# For benchmark adapters reading foreign, untyped JSON (result files, judge
# output) that has no model. Loop artifacts do not need these.


def json_str(raw: Mapping[str, JsonValue], key: str, default: str = "") -> str:
    """One string out of untyped JSON, or ``default``."""
    value = raw.get(key, default)
    return value if isinstance(value, str) else default  # code-health: ignore[AM025]


def json_int(raw: Mapping[str, JsonValue], key: str, default: int = 0) -> int:
    """One int out of untyped JSON. ``bool`` is an ``int`` in Python and is not
    one here: ``True`` arriving where a turn index belongs is bad data."""
    value = raw.get(key, default)
    if isinstance(value, bool):  # code-health: ignore[AM025]
        return default
    return value if isinstance(value, int) else default  # code-health: ignore[AM025]


# -- the record base -----------------------------------------------------------


class Record(BaseModel):
    """One shape, three readers: artifact row, tool schema, typed value."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    def to_json(self) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", self.model_dump(mode="json", by_alias=True))

    @classmethod
    def from_json(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls.model_validate(raw)


# -- shared --------------------------------------------------------------------


class Metrics(Record):
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

    values: dict[str, float | None] = Field(default_factory=dict)
    primary: str = ""

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


class Evidence(Record):
    """A quote with where it came from, so a claim can be re-checked."""

    source: str = ""
    quote: str = ""


class EvidenceRef(Record):
    """Somewhere a diagnoser should look, named by the benchmark that knows.

    Rendered into the prompt verbatim, so the loop never learns that a verifier
    writes ``runner_*.log`` or that stories have parameter files. ``note`` is
    where an adapter says what the thing is good for -- an assertion id can be a
    wrapper around a whole suite, and only the adapter knows that.
    """

    label: str = ""
    locator: str = ""
    note: str = ""


class Assertion(Record):
    """One graded check that failed. ``kind`` separates the functional suite
    from the validation stories because they fail for different reasons and are
    reported in different files."""

    kind: str = "test"
    assertion_id: str = Field(
        default="", validation_alias="id", serialization_alias="id"
    )
    message: str = ""


# -- stage 1: discovery --------------------------------------------------------


class FailureCase(Record):
    """One graded attempt that did not pass.

    Self-describing on purpose. Whatever a diagnoser needs to read is listed in
    ``evidence``; whatever a replay backend needs to re-run the attempt is in
    ``backend_ref``, opaque to everything between. A stage may be handed a
    hand-edited ``cases.json`` and nothing else, so nothing here may depend on
    the adapter still being around to fill a gap in later.
    """

    case_id: str
    task_name: str
    #: Which model made this attempt. Empty when the source cannot say, which
    #: costs only the model-scope notes -- they group on it.
    model_name: str = ""
    #: The task as the agent received it.
    instruction: str = ""
    failing_assertions: tuple[Assertion, ...] = ()
    #: Where to look, in the benchmark's own terms.
    evidence: tuple[EvidenceRef, ...] = ()
    #: How this task's attempts compare, written by the adapter because judging
    #: whether an assertion id is fine-grained enough needs benchmark knowledge.
    #: Empty when there is only one attempt.
    cohort_note: str = ""
    metrics: Metrics = Field(default_factory=Metrics)
    sibling_metrics: tuple[Metrics, ...] = ()
    #: Everything the replay backend needs and nobody else interprets: a
    #: workspace checkpoint, a conversation turn, a container digest -- whatever
    #: this benchmark's re-run takes. Typed with pydantic's own JsonValue: the
    #: ABI's recursive alias sends schema generation into infinite recursion.
    backend_ref: dict[str, PydanticJsonValue] = Field(default_factory=dict)

    @property
    def replayable(self) -> bool:
        """Whether a backend has anything to re-run from."""
        return bool(self.backend_ref)


# -- stage 2: diagnose ---------------------------------------------------------

#: Why the attempt is wrong. The last two exist so the loop can retire a case
#: instead of grinding on it: one dev-set task is graded against a
#: reference-internal parameter name the task never states, and another's graded
#: suite fails nondeterministically in code no patch touches.
CauseClass = Literal[
    "misread_intent",
    "omitted",
    "wrong_implementation",
    "unverified_claim",
    "environment_self_break",
    "grading_artifact",
    "harness_nondeterminism",
]
CAUSE_CLASSES: tuple[str, ...] = get_args(CauseClass)


class DiagnosisPayload(Record):
    """What the diagnoser fills in. Doubles as the ``submit_diagnosis`` tool
    schema, so a field description here is what the model reads."""

    required_behaviour: str = Field(
        description="What grading demands, quoted from the failing assertions."
    )
    reference_idea: str = Field(
        description="The reference solution's central idea, one sentence."
    )
    agent_idea: str = Field(
        description="What the agent built instead, in the same terms."
    )
    divergence_turn: int = Field(
        default=-1,
        description="Turn index where the approach stopped being open. -1 if "
        "you cannot locate it.",
    )
    divergence_quote: str = Field(
        default="", description="What was said or done at that turn."
    )
    cause_class: CauseClass
    mechanism: str = Field(
        description="Why it is wrong, concretely: what breaks, under what "
        "condition. Not 'it did not test enough'."
    )
    decision: str = Field(
        description="What was being chosen at the divergence turn, in the "
        "agent's own terms."
    )
    unasked_question: str = Field(
        description="The step not taken: a question the agent could have asked "
        "itself at that moment, answerable only by running something. Must be "
        "askable without knowing the answer -- if it names the defect, it is "
        "hindsight, not a question."
    )
    discriminating_answer: str = Field(
        description="What asking it would have produced, and what the agent "
        "believed instead. If the answer would look the same whether or not "
        "the agent was right, this is not the missing step; find the question "
        "whose answer differs."
    )
    lesson: str = Field(
        description="What someone starting a different task in this same "
        "repository should know because of this. May name this repository's "
        "conventions and traps; may not name this task's defect. Empty if this "
        "case teaches nothing."
    )
    reachable: bool = Field(
        description="Could anything said to the agent before it submitted have "
        "changed this outcome?"
    )
    reachable_rationale: str = Field(
        default="",
        description="When not reachable, why not. A justified no is worth more "
        "than an invented yes.",
    )
    evidence: tuple[Evidence, ...] = Field(
        description="Every claim with a quote and where it came from."
    )


class Diagnosis(DiagnosisPayload):
    """The diagnoser's answer, plus what the loop assigns: identity and the
    case it came from."""

    diagnosis_id: str
    case_id: str
    task_name: str = ""
    #: Carried from the case so notes can group by it without re-reading the
    #: batch. Empty when the source cannot say, which costs only model-scope
    #: notes -- they group on it.
    model_name: str = ""

    @property
    def worth_abstracting(self) -> bool:
        """A grading artifact or a flaky harness has no check to learn."""
        return self.reachable and self.cause_class not in {
            "grading_artifact",
            "harness_nondeterminism",
        }


# -- stage 3: abstract ---------------------------------------------------------


class CandidatePayload(Record):
    """What the abstractor fills in. Doubles as the ``submit_candidate`` tool
    schema.

    ``observation`` is the load-bearing field. A check that can be satisfied by
    reasoning gets satisfied by reasoning: an offline A/B had the reviewing
    model produce a structured, confident endorsement of the very diagnosis
    that was wrong. A candidate that names nothing to run is rejected before it
    costs a replay.
    """

    check: str = Field(
        description="The message the agent will receive. One doubt, plainly "
        "put. No file, function or value from this task."
    )
    observation: str = Field(
        description="What running this check actually produces: a number, an "
        "output, an exit status. Empty means the check is reflective, and the "
        "candidate will be discarded."
    )
    when_note: str = Field(
        description="In plain language, what must be true of the session for "
        "this to be worth sending."
    )
    precondition_note: str = Field(
        default="",
        description="In plain language, what must be observably true of the "
        "session's recorded actions. Empty means unconditional.",
    )
    checkpoint: Literal["continuous", "stop"] = Field(
        description="continuous: mid-work. stop: when the agent wraps up."
    )
    dimension: str = Field(
        default="",
        description="Which representation-chain dimension this guards, D1 to D7.",
    )


class Candidate(CandidatePayload):
    """A general check proposed from one diagnosis, plus its provenance."""

    candidate_id: str
    from_task: str = ""
    from_session: str = ""
    from_diagnosis: str = ""
    #: The turn this was derived from: where the approach it questions became
    #: fixed. Carried because it is where the check has to arrive to be worth
    #: anything -- delivered at the end it lands after the work it would undo.
    from_turn: int = -1

    def rejection(self) -> str:
        """Why this candidate must not proceed, or empty when it may."""
        if not self.check.strip():
            return "empty check"
        if not self.observation.strip():
            return "names no observation: satisfiable by reasoning alone"
        return ""


# -- stage 4: compile ----------------------------------------------------------


class PredicateProposal(Record):
    name: str = Field(default="", description="Predicate name, snake_case.")
    description: str = Field(
        default="",
        description="What must be observable in the session's events for this "
        "predicate to hold.",
    )


class GatePayload(Record):
    """What the compiler fills in. Doubles as the ``submit_gate`` tool schema."""

    trigger: str = Field(
        description="Boolean expression over vocabulary predicates, or `always`."
    )
    precondition: str = Field(
        description="SQL returning rows when the presupposed situation has "
        "arrived. Empty when the check needs no prior state."
    )
    new_predicates: tuple[PredicateProposal, ...] = Field(
        default=(),
        description="Only what the vocabulary lacks. Each must be decidable "
        "from the session's events alone.",
    )


class CompiledCandidate(GatePayload):
    """A candidate with a firing condition it can actually be gated on.

    ``trigger`` is a boolean expression over tagger predicates -- what the
    session *looks* like. ``precondition`` is SQL over the data plane -- whether
    the situation the check presupposes has arrived. Both must hold. Measured
    need: six of eleven firings of the validation-scope item landed in sessions
    that had run no test at all, so its stated premise was false.
    """

    candidate: Candidate
    item_id: str = ""


# -- stage 5: replay -----------------------------------------------------------

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


class ReplayMeasurement(Record):
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
    metrics_before: Metrics = Field(default_factory=Metrics)
    metrics_after: Metrics = Field(default_factory=Metrics)
    job_dir: str = ""
    #: What the review said, when the run had one. Kept beside the numbers so
    #: alignment can be judged without going back to a session store.
    review_report: str = ""

    @property
    def delta(self) -> Metrics:
        return self.metrics_after.minus(self.metrics_before)

    def to_json(self) -> dict[str, JsonValue]:
        """The row plus its derived delta, so an artifact reads without a
        calculator."""
        encoded = super().to_json()
        encoded["delta"] = self.delta.to_json()
        return encoded


# -- repository notes ----------------------------------------------------------


#: What a note is about, and therefore where it transfers. A repository note
#: holds for one codebase and is mined from its own failures. A model note
#: holds for one model wherever it works, and is mined only across
#: repositories -- anything supported by a single repository's failures is a
#: repository note wearing the wrong label.
NOTE_SCOPES: tuple[str, ...] = ("repository", "model")


class Note(Record):
    """One thing a reviewer has to be told before it starts.

    Distinct from a ``Candidate``, and the difference is who reads it. A
    candidate is sent to the agent doing the work, at the moment it is working:
    "before you go further, check this". A note is given to whoever reviews that
    work, and it holds for the repository rather than for a moment: "here, a
    change of this kind meets these conditions, and this is where a plausible
    shortcut gives a wrong answer".

    The distinction earned its place. Twenty-one candidates moved no score. One
    note -- that the graded tests run against a real database, so a mocked store
    proves nothing about a fix concerned with locking -- changed how the reviewer
    worked on the first attempt: it stood up a real Postgres instead of modelling
    the interleaving it needed.

    ``situation`` is the load-bearing field, and it is why this is not a lesson.
    A note nobody can put a change into is prose. This one names conditions a
    reviewer can go and create.
    """

    note_id: str
    #: "repository" or "model" -- see NOTE_SCOPES.
    scope: str = "repository"
    #: The repository or the model this holds for.
    subject: str = ""
    #: The conditions to put a change into: what to run it against, and under
    #: what circumstances. Empty is rejected.
    situation: str = ""
    #: What a reviewer would otherwise conclude, and why it would be wrong.
    without_it: str = ""
    from_cases: tuple[str, ...] = ()

    #: Repositories the evidence came from. A model note needs more than one;
    #: with one it cannot be told apart from a fact about that codebase.
    from_repositories: tuple[str, ...] = ()

    def rejection(self) -> str:
        if not self.situation.strip():
            return "no situation: a note nobody can put a change into is prose"
        if not self.subject.strip():
            return "no subject: a note that is true everywhere belongs in the prompt"
        if self.scope not in NOTE_SCOPES:
            return f"unknown scope {self.scope!r}"
        if self.scope == "model" and len(set(self.from_repositories)) < 2:
            return (
                "a model note drawn from one repository is a repository note: "
                "what separates the model from the codebase is surviving both"
            )
        return ""


class NoteEntry(Record):
    """One note as the noter submits it; the loop adds identity and provenance."""

    situation: str = Field(
        description="What to put the change into, reachable from a checkout: "
        "something to run, a service to start, a second process to introduce."
    )
    without_it: str = Field(
        description="What a reviewer who did not know this would conclude, and "
        "why that is wrong."
    )


class NotesPayload(Record):
    """The ``submit_notes`` tool schema: the merged set, not additions to it."""

    notes: tuple[NoteEntry, ...] = Field(
        description="The merged set. Empty is a valid answer."
    )
    dropped: str = Field(
        default="",
        description="Any note you removed from the existing set, and why the "
        "evidence no longer supports it.",
    )


# -- alignment -----------------------------------------------------------------

#: How close a review came to what grading actually punished. The point of
#: measuring this rather than the score is density: a score moves on whether the
#: agent then fixed the thing correctly, which is a second question and a rarer
#: event, while this answers the first one on every run.
AlignmentVerdict = Literal["same", "adjacent", "elsewhere", "none"]
ALIGNMENTS: tuple[str, ...] = get_args(AlignmentVerdict)


class AlignmentPayload(Record):
    """What the aligner fills in. Doubles as the ``submit_alignment`` tool
    schema."""

    verdict: AlignmentVerdict = Field(
        description="same: fixing the review's finding makes the assertion "
        "pass. adjacent: same code, different defect. elsewhere: real but "
        "unrelated. none: no case reported."
    )
    finding: str = Field(
        default="", description="The review's strongest finding, in one line."
    )
    graded_failure: str = Field(
        default="", description="The assertion you compared it against."
    )
    reason: str = Field(
        description="Why that verdict. For 'same', the line from the finding "
        "to the assertion no longer failing."
    )


class Alignment(AlignmentPayload):
    """Whether a review found the thing grading punished, tied to its case."""

    case_id: str
    candidate_id: str = ""


# -- stage 6: select -----------------------------------------------------------


class Verdict(Record):
    candidate_id: str
    accepted: bool = False
    reason: str = ""
    improved_cases: tuple[str, ...] = ()
    regressed_cases: tuple[str, ...] = ()
    lost_cases: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        """Whether any arm produced evidence, either way.

        ``accepted`` is one bit and it collapses two very different failures:
        measured and not useful, versus never measured because the replay was
        lost. Only the first is grounds for removing an installed item -- the
        second is an experiment that did not finish, and four of the first six
        replays run by hand were exactly that.
        """
        return bool(self.improved_cases or self.regressed_cases)


# -- artifact io ---------------------------------------------------------------


def write_artifact(path: Path, records: Sequence[Record]) -> None:
    """One stage's output. Sorted keys and a trailing newline so two runs of the
    same stage diff cleanly -- the standalone-equals-chained check depends on
    byte equality."""
    payload = [record.to_json() for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_artifact(path: Path) -> list[Mapping[str, JsonValue]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):  # code-health: ignore[AM025]
        raise TypeError(f"{path}: expected a JSON list of records")
    return [
        item
        for item in raw
        if isinstance(item, Mapping)  # code-health: ignore[AM025]
    ]


__all__ = [
    "ALIGNMENTS",
    "ARMS",
    "CAUSE_CLASSES",
    "OUTCOMES",
    "Alignment",
    "AlignmentPayload",
    "Assertion",
    "Candidate",
    "CandidatePayload",
    "CompiledCandidate",
    "Diagnosis",
    "DiagnosisPayload",
    "Evidence",
    "EvidenceRef",
    "FailureCase",
    "GatePayload",
    "Metrics",
    "Note",
    "NoteEntry",
    "NotesPayload",
    "PredicateProposal",
    "Record",
    "ReplayMeasurement",
    "Verdict",
    "json_int",
    "json_str",
    "read_artifact",
    "write_artifact",
]
