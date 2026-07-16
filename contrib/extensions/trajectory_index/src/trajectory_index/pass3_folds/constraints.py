"""Constraint satisfaction analysis — Pass 0 / E / J / L.

Extends the grounding analysis with constraint-level verification (see
designs/constraint-satisfaction.md for the contracts P1-P7). Organized like
a compiler pipeline: each pass is one function with an explicit
input → output signature, all diagnostics flow into one sink, and the
driver (:func:`analyze_constraints`) only chains passes.

    (constraint nodes come from Pass 1 — the task text is part of the
    trajectory; there is no separate extraction pass here)

    Pass E1  _detect_commit        index, steps         → Commit | None
    Pass E2  _map_about            grounded steps       → evidence steps
    Pass E3  _judge_entailment     constraints, window  → {cid: Verdict}
    Pass J   _check_omitted        unsettled, trace     → {cid: Verdict}
    Pass L   _emit_findings        verdicts             → [ConstraintFinding]

Contracts honored here:

* oracle judgments run over code-selected windows of WHOLE steps — selection
  never cuts content mid-step, and every deselection is a logged prune; the
  judgments assert only positive facts about the presented content (P4);
* a missing/unparseable verdict is unknown and never escalates (P5);
* Omitted requires two independent absence checks — a lexical code-negative
  over the whole trace AND an attested coverage sweep with citation-on-yes
  and abstention on truncation (P4-iii);
* every model influence flows through a recorded transcript row, so
  Pass J/L output is a deterministic function of (facts, transcript) (P3);
* every code-side prune is recorded (P2: no silent false negatives).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..ir.diagnostics import Diagnostics
from ..ir.models import (
    Constraint,
    ConstraintFinding,
    FindingStatus,
    Step,
    Symbol,
    mentions_symbol,
    normalize_name,
)
from ..oracle import SessionFactory, _ask_model, _index_by_id, _safe_float

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex

# No mid-content truncation anywhere in this module: oracle windows are
# whole-step selections (P4-i bounds by SELECTION, logged as prunes — never
# by cutting content). An oversized window fails the model call and every
# affected tuple degrades to unknown (P5) — honest failure beats a silently
# partial view. The one numeric knob left is the sweep abstention budget:
# it guards a NEGATIVE claim (needle-in-haystack recall over long context is
# the known small-model weak spot), fails safe (over budget → unknown), and
# is a parameter to be calibrated on the dev slice (validation plan step 3).

_SWEEP_CHAR_BUDGET = 20000    # default sweep abstention budget (chars)


# ---------------------------------------------------------------------------
# Pass-to-pass value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Commit:
    """Pass E1 output: what the agent answered with, and where.

    ``binding`` is the committed answer as text — the symbol's canonical
    name when the phrase resolves to an indexed entity, else the extracted
    phrase itself. Keeping the phrase usable when resolution fails is what
    frees commit detection from Pass 1 extraction recall (the answer entity
    is exactly the one hard trajectories fail to extract).
    """

    binding: str
    step: Step
    confidence: float
    symbol: Symbol | None = None

    @property
    def names(self) -> set[str]:
        out = {self.binding}
        if self.symbol is not None:
            out |= self.symbol.all_names
        return {n for n in out if n.strip()}


@dataclass(frozen=True, slots=True)
class Verdict:
    """One constraint's judged status before Pass L anchoring."""

    status: FindingStatus
    confidence: float
    source: str                       # "code" | "oracle:<relation>"
    evidence_step_ids: tuple[str, ...] = ()
    reason: str = ""


_UNKNOWN = Verdict("unknown", 0.0, "code", (), "never judged")


@dataclass(slots=True)
class ConstraintAnalysis:
    """Driver output: findings plus the record that explains them."""

    constraints: list[Constraint] = field(default_factory=list)
    findings: list[ConstraintFinding] = field(default_factory=list)
    candidate: str = ""
    commit_step_id: str | None = None
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "commit_step_id": self.commit_step_id,
            "constraints": [
                {
                    "id": c.id, "description": c.description,
                    "normalized": dict(c.normalized) if c.normalized else None,
                }
                for c in self.constraints
            ],
            "findings": [
                {
                    "constraint_id": f.constraint_id, "candidate": f.candidate,
                    "status": f.status, "evidence_step_ids": list(f.evidence_step_ids),
                    "commit_step_id": f.commit_step_id, "confidence": f.confidence,
                    "confidence_source": f.confidence_source, "reason": f.reason,
                }
                for f in self.findings
            ],
            "transcript": self.diagnostics.transcript,
            "prune_log": self.diagnostics.prune_log,
        }



# ---------------------------------------------------------------------------
# Pass E1 — Commit detection (code proposes, oracle picks)
# ---------------------------------------------------------------------------

def _resolve_binding(index: TrajectoryIndex, phrase: str) -> Symbol | None:
    """Resolve an extracted answer phrase to an indexed symbol (pure code).

    Exact normalized lookup first, then containment over symbol names.
    Failure is fine — the Commit keeps the phrase as its binding.
    """
    sym = index.resolve_symbol_by_name(phrase)
    if sym is not None:
        return sym
    norm = normalize_name(phrase)
    if len(norm) < 3:
        return None
    for s in index.symbols.values():
        for name in s.all_names:
            n = normalize_name(name)
            if n and (n in norm or norm in n):
                return s
    return None


async def _detect_commit(
    index: TrajectoryIndex,
    steps: list[Step],
    *,
    question: str,
    model: str | None,
    session_factory: SessionFactory,
    diag: Diagnostics,
) -> Commit | None:
    """Pass E1: extract the committed answer, then resolve it to an indexed
    symbol (code, best-effort).

    A Pass 1 commit-role claim (``⟦claim role=commit|…⟧``) anchors the
    step and narrows the oracle's input to the committed sentence itself;
    without one, the oracle reads the final assistant message whole.
    """
    steps_by_id = {s.step_id: s for s in steps}
    commit_claims = sorted(
        (
            (steps_by_id[c.step_id], c)
            for c in index.claims.values()
            if c.role == "commit" and c.step_id in steps_by_id
        ),
        key=lambda pair: pair[0].index,
    )
    final: Step | None
    if commit_claims:
        final, claim = commit_claims[-1]     # the last commitment stands
        commit_text: str | None = claim.text
        diag.record("commit", final.step_id, None, 0.0,
                    f"Pass 1 commit claim anchors the step: {claim.text[:80]!r}")
    else:
        final = next((s for s in reversed(steps) if s.action_segment), None)
        commit_text = final.action_segment if final is not None else None
    if final is None:
        diag.record("commit", "-", None, 0.0, "no final assistant step")
        return None

    payload = json.dumps({
        "question": question,
        "final_message": commit_text,
    }, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        "constraint_commit", payload, model,
        session_factory=session_factory, purpose="constraint_commit",
    )
    item = _index_by_id(raw or []).get(0)
    phrase = str(item.get("answer", "")).strip() if item else ""
    conf = _safe_float(item, "confidence") if item else 0.0
    if not phrase:
        diag.record("commit", final.step_id, None, conf, "no committed answer extracted")
        return None

    sym = _resolve_binding(index, phrase)
    binding = sym.canonical_name if sym is not None else phrase
    diag.record(
        "commit", final.step_id, binding, conf,
        f"phrase={phrase[:80]!r} resolved={'yes' if sym else 'no'}",
    )
    return Commit(binding=binding, step=final, confidence=conf, symbol=sym)


# ---------------------------------------------------------------------------
# Pass E2 — About: map grounded steps to the committed candidate
# ---------------------------------------------------------------------------



async def _map_about(
    grounded: list[Step],
    commit: Commit,
    *,
    model: str | None,
    session_factory: SessionFactory,
    diag: Diagnostics,
) -> list[Step]:
    """Pass E2: grounded steps carrying evidence about the committed candidate.

    Sweeps the WHOLE grounded evidence space in deterministic char-budget
    partitions — code does not pre-filter relevance (the earlier lexical
    blocking was code recognizing "aboutness" and pruned anaphoric
    evidence before any model saw it). Positive polarity per partition; a
    failed partition loses only its own steps, recorded.
    """
    from ..pass2_edges.claims import _PARTITION_CHAR_BUDGET, _step_chars, partition_by_budget

    about: list[Step] = []
    for partition in partition_by_budget(grounded, _PARTITION_CHAR_BUDGET, _step_chars):
        rows = [
            {"id": i, "target": commit.binding, "excerpt": s.observation_segment or ""}
            for i, s in enumerate(partition)
        ]
        raw = await _ask_model(
            "constraint_about", json.dumps(rows, ensure_ascii=False, indent=2),
            model, session_factory=session_factory, purpose="constraint_about",
        )
        if raw is None:
            diag.record("about", "-", None, 0.0,
                        f"partition of {len(partition)} steps failed; its steps unmapped")
            continue
        by_id = _index_by_id(raw)
        for i, s in enumerate(partition):
            item = by_id.get(i)
            is_about = bool(item.get("about", False)) if item else False
            conf = _safe_float(item, "confidence") if item else 0.0
            diag.record("about", s.step_id, is_about, conf)
            if is_about:
                about.append(s)
    return about


# ---------------------------------------------------------------------------
# Pass E3 — Entails/Contradicts per (constraint, candidate) over the step set
# ---------------------------------------------------------------------------

async def _judge_entailment(
    constraints: list[Constraint],
    commit: Commit,
    about_steps: list[Step],
    *,
    model: str | None,
    session_factory: SessionFactory,
    diag: Diagnostics,
) -> dict[str, Verdict]:
    """Pass E3: settle each constraint against the candidate's evidence set.

    The window is ALL About-true steps in reading order, content in full —
    the selection already happened in Pass E2's blocking (logged), and no
    content is ever cut. Machine-checkable constraints are decided by code
    from the oracle's quoted value (P6). "neither" leaves no verdict — the
    constraint flows to Pass J's Omitted path.
    """
    if not about_steps:
        return {}
    window = sorted(about_steps, key=lambda s: s.index)
    window_ids = {s.step_id for s in window}

    payload = json.dumps({
        "candidate": commit.binding,
        "constraints": [
            {"id": i, "desc": c.description}
            for i, c in enumerate(constraints)
        ],
        "evidence": [
            {"id": s.step_id, "excerpt": s.observation_segment or ""}
            for s in window
        ],
    }, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        "constraint_entails", payload, model,
        session_factory=session_factory, purpose="constraint_entails",
    )
    by_id = _index_by_id(raw or [])

    verdicts: dict[str, Verdict] = {}
    for i, c in enumerate(constraints):
        item = by_id.get(i)
        if not item:
            diag.record("entails", c.id, None, 0.0, "no verdict")
            continue
        outcome = str(item.get("outcome", "neither"))
        quote = str(item.get("quote", ""))
        conf = _safe_float(item, "confidence")
        ev = tuple(
            str(s) for s in item.get("steps", [])
            if isinstance(s, str | int) and str(s) in window_ids
        )
        diag.record("entails", c.id, outcome, conf, f"quote={quote[:120]}")

        if outcome not in ("establish", "refute"):
            continue  # "neither" → Omitted path (Pass J)
        status: FindingStatus = "verified" if outcome == "establish" else "violated"
        verdicts[c.id] = Verdict(
            status, conf, "oracle:entails", ev, str(item.get("reason", "")),
        )
    return verdicts


# ---------------------------------------------------------------------------
# Pass J — Omitted double-negative for unsettled constraints
# ---------------------------------------------------------------------------

async def _check_omitted(
    unsettled: list[Constraint],
    grounded: list[Step],
    commit: Commit,
    *,
    model: str | None,
    session_factory: SessionFactory,
    diag: Diagnostics,
    sweep_char_budget: int = _SWEEP_CHAR_BUDGET,
) -> dict[str, Verdict]:
    """Pass J: Omitted only when the lexical negative AND the attested
    coverage sweep both say absent; anything else stays unknown (P5)."""
    verdicts: dict[str, Verdict] = {}

    if not grounded:
        # Empty evidence space: with zero tool-output steps we cannot
        # tell "the agent never verified anything" from "ingestion lost
        # provenance" (e.g. spans arriving as raw text with no roles). A
        # negative over an empty evidence space is vacuous — abstain. The
        # grounding warnings already flag the no-tool-output condition.
        diag.record("sweep", "-", "abstain", 0.0, "no grounded tool-output steps")
        for c in unsettled:
            verdicts[c.id] = Verdict(
                "unknown", 0.0, "code", (),
                "no grounded tool-output steps in trajectory; cannot judge omission",
            )
        return verdicts

    sweep_targets = list(unsettled)

    # The abstain gate counts full length, and past it the sweep reads full
    # step texts: a "no evidence" verdict over any partial view is the
    # vacuous-closed-world bug in miniature, so the sweep either sees
    # everything or asserts nothing.
    total_chars = sum(len(s.observation_segment or "") for s in grounded)
    if total_chars > sweep_char_budget:
        diag.record("sweep", "-", "abstain", 0.0,
                    f"{total_chars} chars > budget {sweep_char_budget}")
        for c in sweep_targets:
            verdicts[c.id] = Verdict(
                "unknown", 0.0, "code", (),
                f"sweep abstained: grounded text {total_chars} chars over cap",
            )
        return verdicts
    snippets = [{"id": s.step_id, "excerpt": s.observation_segment or ""} for s in grounded]

    payload = json.dumps({
        "constraints": [
            {"id": i, "desc": c.description} for i, c in enumerate(sweep_targets)
        ],
        "excerpts": snippets,
    }, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        "constraint_sweep", payload, model,
        session_factory=session_factory, purpose="constraint_sweep",
    )
    by_id = _index_by_id(raw or [])
    valid_ids = {s.step_id for s in grounded}

    for i, c in enumerate(sweep_targets):
        item = by_id.get(i)
        if not item:
            diag.record("sweep", c.id, None, 0.0, "no verdict")
            verdicts[c.id] = Verdict("unknown", 0.0, "code", (), "sweep returned no verdict")
            continue
        exists = bool(item.get("evidence_exists", True))
        cited = str(item.get("step", ""))
        conf = _safe_float(item, "confidence")
        diag.record("sweep", c.id, exists, conf, f"step={cited}")

        if exists and cited in valid_ids:
            verdicts[c.id] = Verdict(
                "unknown", conf, "oracle:sweep", (cited,),
                "sweep found evidence not settled by the window",
            )
        elif exists:
            verdicts[c.id] = Verdict(
                "unknown", 0.0, "code", (), "sweep said yes without a valid citation",
            )
        else:
            verdicts[c.id] = Verdict(
                "omitted",
                min(conf, commit.confidence) if commit.confidence else conf,
                "oracle:sweep", (),
                "no lexical match and coverage sweep found no evidence",
            )
    return verdicts


# ---------------------------------------------------------------------------
# Pass L — anchor verdicts into findings (pure code)
# ---------------------------------------------------------------------------


def _first_assertion_step(steps: list[Step], commit: Commit) -> Step:
    """Earliest assistant step asserting the committed binding (pure code).

    The minimal bad prefix of "commit without verification" ends where the
    agent FIRST commits the claim, not at its final restatement — validated
    against TELBench gold (30/40 omitted-case golds lie strictly earlier
    than the final report; only 10/40 include it). P7: anchors follow the
    benchmark's measured label convention.
    """
    for s in steps:
        seg = s.action_segment
        if seg and mentions_symbol(seg, commit.names):
            return s
    return commit.step


def _emit_findings(
    constraints: list[Constraint],
    verdicts: dict[str, Verdict],
    commit: Commit,
    steps: list[Step],
) -> list[ConstraintFinding]:
    """Pass L: emit localization FACTS, never a designated error step.

    Two code-heuristic anchor conventions failed their gold-validation gate
    on TELBench (final restatement: 10/40 gold hits; first assertion: 7/39)
    — localization is a semantic judgment that belongs to the auditor. Each
    finding therefore carries the fact set {first assertion step, final
    commitment step, evidence steps} and the consumer decides what to make
    of it (P7: anchors are per-benchmark empirical, and none has passed).
    """
    first_assert = _first_assertion_step(steps, commit)
    findings: list[ConstraintFinding] = []
    for c in constraints:
        v = verdicts.get(c.id, _UNKNOWN)
        findings.append(ConstraintFinding(
            constraint_id=c.id,
            candidate=commit.binding,
            status=v.status,
            evidence_step_ids=v.evidence_step_ids,
            commit_step_id=commit.step.step_id,
            first_assertion_step_id=first_assert.step_id,
            confidence=round(v.confidence, 3),
            confidence_source=v.source,
            reason=v.reason,
        ))
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def analyze_constraints(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    question: str | None = None,
    constraints: list[Constraint] | None = None,
    model: str | None = None,
    session_factory: SessionFactory | None = None,
    sweep_char_budget: int = _SWEEP_CHAR_BUDGET,
) -> ConstraintAnalysis:
    """Chain E1 → E2 → E3 → J → L over Pass 1 constraint nodes.

    Constraints come from the index (extracted by Pass 1 from the task
    text — the question is part of the trajectory, so there is no
    separate extraction call). Best-effort like the other adjudication
    passes: any oracle failure degrades the affected tuples to unknown,
    which never escalates (P5). Idempotent: reruns replace
    ``constraint_findings`` wholesale (constraint nodes are Pass 1's).

    ``sweep_char_budget`` is the abstention threshold for the Omitted
    coverage sweep — the one numeric knob in this module. It guards a
    negative claim and fails safe (over budget → unknown, never a finding);
    calibrate it per oracle model on the dev slice.
    """
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")

    analysis = ConstraintAnalysis()
    diag = analysis.diagnostics

    if constraints is None:
        constraints = list(index.constraints.values())
    else:
        index.constraints = {c.id: c for c in constraints}
    analysis.constraints = list(constraints)
    index.constraint_findings = []
    if not constraints:
        logger.info("constraints: no constraint nodes in the index; analysis is empty")
        return analysis

    steps = sorted(
        (s for s in index.steps.values() if not run_id or s.run_id == run_id),
        key=lambda s: s.index,
    )
    # Evidence space: attested tool_result steps plus Pass 1
    # provenance-labeled observation content (observation_segment covers
    # both — mixed steps contribute their retrieved portion only).
    grounded = [s for s in steps if s.observation_segment]

    # Pass E1 — empty Commit: no violation can fire, omission has no anchor (v1: stop).
    commit = await _detect_commit(
        index, steps, question=question or "",
        model=model, session_factory=session_factory, diag=diag,
    )
    if commit is None:
        logger.info("constraints: agent commits to no candidate; no findings emitted")
        return analysis
    analysis.candidate = commit.binding
    analysis.commit_step_id = commit.step.step_id

    # No self-verification: the agent's own commitment step cannot be
    # evidence that its answer satisfies a constraint. When Pass 1 labels
    # the final report's answer synthesis as an observation region (the
    # extracted-final-answer block is retrieved-looking), E3 would read the
    # agent restating "I used method X" as tool confirmation of X — a
    # circular verified. Excluding the commit step keeps constraint
    # evidence to what INDEPENDENTLY grounds the answer (P6: decidable, a
    # step-id exclusion, no semantics).
    evidence = [s for s in grounded if s.step_id != commit.step.step_id]
    if len(evidence) != len(grounded):
        diag.record("evidence", commit.step.step_id, None, 0.0,
                    "commit step excluded from constraint evidence (no self-verification)")

    # Pass E2 / E3
    about_steps = await _map_about(
        evidence, commit, model=model, session_factory=session_factory, diag=diag,
    )
    verdicts = await _judge_entailment(
        constraints, commit, about_steps,
        model=model, session_factory=session_factory, diag=diag,
    )

    # Pass J — same independent-evidence set (self-verification excluded)
    unsettled = [c for c in constraints if c.id not in verdicts]
    verdicts.update(await _check_omitted(
        unsettled, evidence, commit,
        model=model, session_factory=session_factory, diag=diag,
        sweep_char_budget=sweep_char_budget,
    ))

    # Pass L
    analysis.findings = _emit_findings(constraints, verdicts, commit, steps)
    index.constraint_findings = analysis.findings

    by_status: dict[str, int] = {}
    for f in analysis.findings:
        by_status[f.status] = by_status.get(f.status, 0) + 1
    logger.info(
        "constraints: candidate='{}' findings={}",
        commit.binding, by_status,
    )
    return analysis
