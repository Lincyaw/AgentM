"""Constraint satisfaction analysis — Pass 0 / E / J / L.

Extends the grounding analysis with constraint-level verification (see
designs/constraint-satisfaction.md for the contracts P1-P7). Organized like
a compiler pipeline: each pass is one function with an explicit
input → output signature, all diagnostics flow into one sink, and the
driver (:func:`analyze_constraints`) only chains passes.

    Pass 0   extract_constraints   question text        → [Constraint]
    Pass E1  _detect_commit        index, steps         → Commit | None
    Pass E2  _map_about            grounded steps       → evidence steps
    Pass E3  _judge_entailment     constraints, window  → {cid: Verdict}
    Pass J   _check_omitted        unsettled, trace     → {cid: Verdict}
    Pass L   _emit_findings        verdicts             → [ConstraintFinding]

Contracts honored here:

* oracle judgments run over code-selected windows of WHOLE steps — selection
  never cuts content mid-step, and every deselection is a logged prune; the
  judgments assert only positive facts about the presented content (P4);
* machine-checkable constraints are decided by code — the oracle just
  locates the value (P6);
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
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .adjudicate import SessionFactory, _ask_model, _index_by_id, _safe_float
from .index import (
    Constraint,
    ConstraintFinding,
    FindingStatus,
    Step,
    StepRole,
    Symbol,
    TrajectoryIndex,
    normalize_name,
)

# No mid-content truncation anywhere in this module: oracle windows are
# whole-step selections (P4-i bounds by SELECTION, logged as prunes — never
# by cutting content). An oversized window fails the model call and every
# affected tuple degrades to unknown (P5) — honest failure beats a silently
# partial view. The one numeric knob left is the sweep abstention budget:
# it guards a NEGATIVE claim (needle-in-haystack recall over long context is
# the known small-model weak spot), fails safe (over budget → unknown), and
# is a parameter to be calibrated on the dev slice (validation plan step 3).

_SWEEP_CHAR_BUDGET = 20000    # default sweep abstention budget (chars)
_MIN_LEXICAL_TOKEN = 4        # lexical-negative token length floor

_STOPWORDS = frozenset({
    "the", "and", "that", "with", "from", "this", "have", "been", "were",
    "was", "for", "are", "not", "his", "her", "their", "its", "who", "than",
    "then", "when", "what", "which", "where", "before", "after", "during",
    "into", "about", "there", "they", "them", "also", "some", "same",
})

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _content_tokens(text: str) -> set[str]:
    return {
        t for t in re.split(r"\W+", text.lower())
        if len(t) >= _MIN_LEXICAL_TOKEN and t not in _STOPWORDS
    }


# ---------------------------------------------------------------------------
# Diagnostics sink + pass-to-pass value types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Diagnostics:
    """The one sink every pass writes into.

    ``transcript`` is the oracle-tuple record (P3): one row per model
    judgment. ``prune_log`` records every code-side pruning decision (P2).
    """

    transcript: list[dict[str, Any]] = field(default_factory=list)
    prune_log: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self, relation: str, key: str, verdict: Any,
        confidence: float, detail: str = "",
    ) -> None:
        self.transcript.append({
            "relation": relation, "key": key, "verdict": verdict,
            "confidence": round(confidence, 3), "detail": detail,
        })

    def prune(self, stage: str, what: str, why: str) -> None:
        self.prune_log.append({"stage": stage, "what": what, "why": why})


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
                    "id": c.id, "subject": c.subject, "description": c.description,
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
# Pass 0 — constraint extraction + normalization (LLM, question only)
# ---------------------------------------------------------------------------

_EXTRACT_INSTRUCTIONS = """\
Extract the requirements the final answer must satisfy from this question.
A constraint is one requirement stated in the question. Extract only what the
question states — do not invent, decompose reasoning chains, or add
background knowledge.

For each constraint:
  - "subject": what the requirement is about — "answer" for the answer entity
    itself, or a named related party ("the answer's parent", "the film").
  - "desc": the requirement in plain words.
  - "normalized": ONLY when the requirement is a checkable number/date
    comparison, emit exactly one of:
      {"kind": "year_range", "lo": <int>, "hi": <int>}
      {"kind": "number", "op": "==", "value": <number>}  (op: ==, <=, >=, <, >)
    Omit "normalized" entirely for semantic requirements (occupation,
    nationality, relationship, event descriptions).

Return ONLY a JSON object:
{"verdicts": [{"id": 0, "subject": "...", "desc": "...", "normalized": {...} or null}, ...]}
"""


async def extract_constraints(
    question: str,
    model: str | None = None,
    session_factory: SessionFactory | None = None,
) -> list[Constraint]:
    """Pass 0: one LLM call over the question text only."""
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")
    raw = await _ask_model(
        _EXTRACT_INSTRUCTIONS, question.strip(), model,
        session_factory=session_factory, purpose="constraint_extraction",
    )
    out: list[Constraint] = []
    for i, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        desc = str(item.get("desc", "")).strip()
        if not desc:
            continue
        normalized = item.get("normalized")
        if not isinstance(normalized, dict) or not normalized.get("kind"):
            normalized = None
        out.append(Constraint(
            id=f"c{i}",
            subject=str(item.get("subject", "answer")).strip() or "answer",
            description=desc,
            normalized=normalized,
        ))
    return out


# ---------------------------------------------------------------------------
# Code checkers (P6) — the decidable subproblem never goes to the model
# ---------------------------------------------------------------------------


_NUM_OPS: dict[str, Any] = {
    "==": lambda a, b: a == b, "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, ">": lambda a, b: a > b,
}


def check_normalized(
    normalized: dict[str, Any], quote: str,
) -> tuple[FindingStatus, str] | None:
    """Decide a machine-checkable constraint against a model-located quote.

    Returns (status, detail), or None when code cannot decide — no comparable
    value in the quote, or SEVERAL values that disagree (a sentence quote like
    "active 1985-2007, born 1955" carries years on both sides of a range, and
    code cannot tell which is decisive without semantics). Then the tuple
    stays unknown: the oracle's opinion on arithmetic is not accepted (P6),
    and ambiguity never escalates (P5).
    """
    if not quote:
        return None
    kind = normalized.get("kind")

    if kind == "year_range":
        years = [int(y) for y in _YEAR_RE.findall(quote)]
        if not years:
            return None
        lo, hi = int(normalized.get("lo", 0)), int(normalized.get("hi", 9999))
        inside = [lo <= y <= hi for y in years]
        if all(inside):
            return ("verified", f"years {years} all in [{lo}, {hi}]")
        if not any(inside):
            return ("violated", f"years {years} all outside [{lo}, {hi}]")
        return None  # mixed — quote carries years on both sides of the range

    if kind == "number":
        vals = [float(m) for m in _NUMBER_RE.findall(quote)]
        op_fn = _NUM_OPS.get(str(normalized.get("op", "==")))
        if not vals or op_fn is None:
            return None
        target = float(normalized.get("value", 0))
        checks = [bool(op_fn(v, target)) for v in vals]
        op = str(normalized.get("op", "=="))
        if all(checks):
            return ("verified", f"values {vals} all satisfy {op} {target}")
        if not any(checks):
            return ("violated", f"values {vals} all fail {op} {target}")
        return None  # mixed — several values disagree; which is decisive is semantics

    return None


def lexical_evidence_exists(
    constraint: Constraint, grounded_texts: list[str],
) -> bool:
    """Lexical half of the Omitted double-negative (pure code).

    True means some grounded step shares the constraint's content tokens or
    normalized values — evidence *may* exist, so Omitted is suppressed.
    """
    tokens = _content_tokens(constraint.description)
    norm = dict(constraint.normalized) if constraint.normalized else {}
    lo = int(norm.get("lo", 0) or 0)
    hi = int(norm.get("hi", 0) or 0)
    target = str(norm.get("value", "")) if norm.get("kind") == "number" else ""

    for text in grounded_texts:
        lowered = text.lower()
        if any(t in lowered for t in tokens):
            return True
        if norm.get("kind") == "year_range" and lo and any(
            lo <= int(y) <= hi for y in _YEAR_RE.findall(text)
        ):
            return True
        if target and target in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Pass E1 — Commit detection (code proposes, oracle picks)
# ---------------------------------------------------------------------------

_COMMIT_INSTRUCTIONS = """\
You are shown the task/question an agent worked on and its final message.
Extract the answer the agent COMMITS to — the entity/value it presents as
its conclusion. Committing means presenting it as the answer, not
mentioning it as a rejected or considered option.

- "answer": the committed answer as a SHORT verbatim phrase copied from the
  final message (a name, a title, a value — not a sentence). Empty string
  if the agent commits to nothing (aborts, reports failure, gives no answer).

Return ONLY: {"verdicts": [{"id": 0, "answer": "...", "confidence": 0.9}]}
"""


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
    """Pass E1: extract the committed answer from the final message (oracle,
    free text), then resolve it to an indexed symbol (code, best-effort)."""
    final = next(
        (s for s in reversed(steps) if s.role == StepRole.ASSISTANT and s.content),
        None,
    )
    if final is None:
        diag.record("commit", "-", None, 0.0, "no final assistant step")
        return None

    payload = json.dumps({
        "question": question,
        "final_message": final.content,
    }, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        _COMMIT_INSTRUCTIONS, payload, model,
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

_ABOUT_INSTRUCTIONS = """\
Each numbered item is an excerpt from a tool/search result in an agent
trajectory, plus a target entity. Decide, per item independently, whether the
excerpt carries evidence ABOUT the target entity — directly, or via a closely
related party (their parent, their work, an event in their life). Judge only
what the excerpt shows.

Return ONLY: {"verdicts": [{"id": 0, "about": true, "confidence": 0.9}]}
"""


def _mentions(text: str, names: set[str]) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in names)


def _block_about(
    index: TrajectoryIndex,
    grounded: list[Step],
    commit: Commit,
    diag: Diagnostics,
) -> list[Step]:
    """Code half of Pass E2: block candidate evidence steps, log every prune.

    A step is blocked in if it mentions the candidate, or shares a symbol
    with a step that does (one hop — satellite entities: the parent, the
    book, the year).
    """
    # step_id is unique only within a run — key on (run_id, step_id)
    refs_by_step: dict[tuple[str, str], set[str]] = {}
    for r in index.references.values():
        refs_by_step.setdefault((r.run_id, r.step_id), set()).add(r.symbol_id)

    direct = [s for s in grounded if _mentions(s.content, commit.names)]
    direct_ids = {s.step_id for s in direct}
    direct_syms = {
        sid for s in direct for sid in refs_by_step.get((s.run_id, s.step_id), set())
    }

    linked = [
        s for s in grounded
        if s.step_id not in direct_ids
        and refs_by_step.get((s.run_id, s.step_id), set()) & direct_syms
    ]
    blocked = sorted(direct + linked, key=lambda s: s.index)

    blocked_ids = {s.step_id for s in blocked}
    for s in grounded:
        if s.step_id not in blocked_ids:
            diag.prune("about", s.step_id,
                       "no candidate mention and no shared symbol with a mentioning step")
    return blocked


async def _map_about(
    index: TrajectoryIndex,
    grounded: list[Step],
    commit: Commit,
    *,
    model: str | None,
    session_factory: SessionFactory,
    diag: Diagnostics,
) -> list[Step]:
    """Pass E2: grounded steps carrying evidence about the committed candidate."""
    blocked = _block_about(index, grounded, commit, diag)
    if not blocked:
        return []

    rows = [
        {"id": i, "target": commit.binding, "excerpt": s.content}
        for i, s in enumerate(blocked)
    ]
    raw = await _ask_model(
        _ABOUT_INSTRUCTIONS, json.dumps(rows, ensure_ascii=False, indent=2),
        model, session_factory=session_factory, purpose="constraint_about",
    )
    by_id = _index_by_id(raw or [])

    about: list[Step] = []
    for i, s in enumerate(blocked):
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

_ENTAILS_INSTRUCTIONS = """\
An agent answered a question with the candidate named below. You are given
the question's constraints and excerpts from the tool/search results the
agent gathered (its evidence). For each constraint, judge what THESE EXCERPTS
establish about the candidate — judge only the presented content:

  - "establish": the excerpts contain facts showing the candidate satisfies
    the constraint. Quote the decisive fact verbatim in "quote".
  - "refute": the excerpts contain facts showing the candidate does NOT
    satisfy it. Quote the decisive fact verbatim in "quote".
  - "neither": these excerpts do not settle this constraint either way.
    (This says nothing about evidence elsewhere — only about these excerpts.)

For constraints involving dates or numbers, "quote" must be the MINIMAL
phrase containing only the decisive value (e.g. "born 1965", not the whole
sentence) — and never do the comparison arithmetic yourself.
List the excerpt ids you relied on in "steps".

Return ONLY:
{"verdicts": [{"id": 0, "outcome": "establish|refute|neither", "quote": "...", "steps": ["3"], "confidence": 0.9, "reason": "..."}]}
"""


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
            {"id": i, "desc": c.description, "machine_checkable": c.normalized is not None}
            for i, c in enumerate(constraints)
        ],
        "evidence": [
            {"id": s.step_id, "excerpt": s.content}
            for s in window
        ],
    }, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        _ENTAILS_INSTRUCTIONS, payload, model,
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
        if c.normalized is not None:
            decided = check_normalized(dict(c.normalized), quote)
            verdicts[c.id] = (
                Verdict(decided[0], conf or 1.0, "code", ev, decided[1])
                if decided
                else Verdict("unknown", 0.0, "code", ev, "no parseable value in quote")
            )
        else:
            status: FindingStatus = "verified" if outcome == "establish" else "violated"
            verdicts[c.id] = Verdict(
                status, conf, "oracle:entails", ev, str(item.get("reason", "")),
            )
    return verdicts


# ---------------------------------------------------------------------------
# Pass J — Omitted double-negative for unsettled constraints
# ---------------------------------------------------------------------------

_SWEEP_INSTRUCTIONS = """\
You are shown ALL tool/search result excerpts an agent gathered, and a list
of constraints from the question it was answering. For each constraint,
decide whether ANY excerpt carries evidence bearing on it (about any entity).
If yes, cite one excerpt id in "step" — a yes without a citation is invalid.
Recall matters more than precision here: when in doubt, answer yes.

Return ONLY: {"verdicts": [{"id": 0, "evidence_exists": true, "step": "12", "confidence": 0.9}]}
"""


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
        # Empty evidence universe: with zero tool-output steps we cannot
        # tell "the agent never verified anything" from "ingestion lost
        # provenance" (e.g. spans arriving as raw text with no roles). A
        # negative over an empty universe is vacuous — abstain. The
        # grounding warnings already flag the no-tool-output condition.
        diag.record("sweep", "-", "abstain", 0.0, "no grounded tool-output steps")
        for c in unsettled:
            verdicts[c.id] = Verdict(
                "unknown", 0.0, "code", (),
                "no grounded tool-output steps in trajectory; cannot judge omission",
            )
        return verdicts

    grounded_texts = [s.content for s in grounded]

    sweep_targets: list[Constraint] = []
    for c in unsettled:
        if lexical_evidence_exists(c, grounded_texts):
            verdicts[c.id] = Verdict(
                "unknown", 0.0, "code", (),
                "lexical evidence exists in trace; not settled by window",
            )
        else:
            sweep_targets.append(c)
    if not sweep_targets:
        return verdicts

    # The abstain gate counts full length, and past it the sweep reads full
    # step texts: a "no evidence" verdict over any partial view is the
    # vacuous-closed-world bug in miniature, so the sweep either sees
    # everything or asserts nothing.
    total_chars = sum(len(s.content) for s in grounded)
    if total_chars > sweep_char_budget:
        diag.record("sweep", "-", "abstain", 0.0,
                    f"{total_chars} chars > budget {sweep_char_budget}")
        for c in sweep_targets:
            verdicts[c.id] = Verdict(
                "unknown", 0.0, "code", (),
                f"sweep abstained: grounded text {total_chars} chars over cap",
            )
        return verdicts
    snippets = [{"id": s.step_id, "excerpt": s.content} for s in grounded]

    payload = json.dumps({
        "constraints": [
            {"id": i, "desc": c.description} for i, c in enumerate(sweep_targets)
        ],
        "excerpts": snippets,
    }, ensure_ascii=False, indent=2)
    raw = await _ask_model(
        _SWEEP_INSTRUCTIONS, payload, model,
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
        if s.role == StepRole.ASSISTANT and s.content and _mentions(s.content, commit.names):
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
    """Chain Pass 0 → E1 → E2 → E3 → J → L; store results on the index.

    Best-effort like the other adjudication passes: any oracle failure
    degrades the affected tuples to unknown, which never escalates (P5).
    Idempotent: reruns replace ``index.constraints`` / ``constraint_findings``
    wholesale.

    ``sweep_char_budget`` is the abstention threshold for the Omitted
    coverage sweep — the one numeric knob in this module. It guards a
    negative claim and fails safe (over budget → unknown, never a finding);
    calibrate it per oracle model on the dev slice.
    """
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")

    analysis = ConstraintAnalysis()
    diag = analysis.diagnostics

    # Pass 0
    if constraints is None:
        if not question:
            raise ValueError("either constraints or question must be provided")
        constraints = await extract_constraints(
            question, model=model, session_factory=session_factory,
        )
    analysis.constraints = list(constraints)
    index.constraints = {c.id: c for c in constraints}
    index.constraint_findings = []
    if not constraints:
        logger.info("constraints: Pass 0 extracted nothing; analysis is empty")
        return analysis

    steps = sorted(
        (s for s in index.steps.values() if not run_id or s.run_id == run_id),
        key=lambda s: s.index,
    )
    grounded = [s for s in steps if s.role == StepRole.TOOL_RESULT and s.content]

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

    # Pass E2 / E3
    about_steps = await _map_about(
        index, grounded, commit, model=model, session_factory=session_factory, diag=diag,
    )
    verdicts = await _judge_entailment(
        constraints, commit, about_steps,
        model=model, session_factory=session_factory, diag=diag,
    )

    # Pass J
    unsettled = [c for c in constraints if c.id not in verdicts]
    verdicts.update(await _check_omitted(
        unsettled, grounded, commit,
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
