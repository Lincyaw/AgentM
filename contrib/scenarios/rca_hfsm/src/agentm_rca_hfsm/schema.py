"""L1 HypothesisGraph data model.

Mirrors design §3.1 (node types), §3.2 (ObservationLog) and §6 (Worker →
Orchestrator two-column contract). The only ``Literal`` enum is
``Prediction.polarity`` — every other classification-like field is free-text
per CLAUDE.md "no preset enums for subjective dimensions".

The dataclasses are plain ``@dataclass`` (mutable, not frozen) because the
store appends to ``Prediction.checks`` and ``Hypothesis.predictions`` in place
under the single-writer discipline of §7.4. Roundtripping through
``dataclasses.asdict`` is the wire format consumed by the observability JSONL
sink — every field is therefore primitive or a list of primitives/nested
dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Observation:
    """A raw, citable fact recorded into the ObservationLog (design §3.2).

    ``tool_signature`` is the canonical ``sha256(tool_name +
    canonical_json(args))`` hash used by the future ObservationLog
    memoization wrapper (commit 3) to short-circuit idempotent tool calls.
    """

    id: str
    text: str
    source_tool_call: str
    tool_signature: str
    related_symptoms: list[str] = field(default_factory=list)
    related_predictions: list[str] = field(default_factory=list)
    ts: float = 0.0


@dataclass
class Interpretation:
    """Worker-supplied advisory accompanying a ``CheckResult`` (design §6).

    Free-text fields by deliberate choice — the orchestrator re-derives the
    update operator from ``observations`` alone, so the worker cannot
    smuggle a verdict in via a preset enum. ``confidence`` is free-text for
    the same reason: a preset scale would invite scoring rather than
    evidence-grounded explanation.
    """

    proposed_update: str
    reasoning: str
    confidence: str


@dataclass
class CheckResult:
    """A single worker session's structured output for one prediction
    (design §3.1 / §6).

    ``worker_session_id`` is load-bearing for the §7.1 independence check —
    two ``CheckResult``\\ s with the same session id cannot jointly satisfy
    the "≥1 independent positive verification" precondition.
    """

    id: str
    prediction_id: str
    worker_session_id: str
    observations: list[Observation] = field(default_factory=list)
    interpretation: Interpretation | None = None
    verdict_proposal: str = ""
    ts: float = 0.0


@dataclass
class Prediction:
    """An observable consequence the FSM will dispatch a worker to test
    (design §3.1).

    ``polarity`` is the only structural enum in the schema — the
    confirm-gate (§7.1) and refute-gate (§7.2) both branch on it, so a free
    string would push that policy decision into prompt-space.
    """

    id: str
    hypothesis_id: str
    claim: str
    polarity: Literal["positive", "negative"]
    test_plan: str | None = None
    checks: list[CheckResult] = field(default_factory=list)


@dataclass
class Hypothesis:
    """A node in the hypothesis DAG (design §3.1).

    ``status`` is intentionally free-form: it carries refine/split/merge
    pointers like ``"refined→H17"`` or ``"split→[H18,H19]"`` that don't fit
    a closed vocabulary. The gate atom (commit 2) is responsible for keeping
    these strings consistent with the operator it just applied.
    """

    id: str
    claim: str
    parent_ids: list[str] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    status: str = "open"
    generation: int = 0
    rationale: str = ""


@dataclass
class Symptom:
    """An observed problem reported during INTAKE (design §3.1).

    ``source`` is free-text — it can be a ``tool_call_id``, the literal
    string ``"user_intake"``, or anything an upstream channel adapter
    constructs.
    """

    id: str
    text: str
    source: str
    ts: float = 0.0


@dataclass
class WorkerReturn:
    """The two-column worker → orchestrator contract (design §6).

    Carries facts (``observations``, ingested into L1) separately from
    advice (``interpretation``, recorded in the trace but not into the
    graph). The orchestrator re-derives the update operator from
    ``observations`` alone — this is the structural anti-bias core of the
    design.
    """

    observations: list[Observation]
    interpretation: Interpretation
