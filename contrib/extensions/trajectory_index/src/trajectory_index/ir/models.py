"""Trajectory index IR — the shared data model every pass reads and writes.

Pure data: dataclasses for the symbol/reference/dependency graph, the
constraint-satisfaction records, the query-result shapes, plus the two
id/name utilities. No pass logic and no intra-package imports live here —
this is the leaf module the whole stack builds on. ``index.py`` re-exports
every name below, so ``from trajectory_index.index import X`` keeps working.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Literal

type MetadataValue = str | int | float | bool | None
type NameKey = tuple[str, str]

# --- Grounding / def-use analysis (see SCHEMA.md) -----------------
# SSA unifies the name world and the value world: every entity is a cell with
# versioned bindings, a use resolves to a reaching binding, and risk is read off
# that binding. `entity_class` (LLM, Pass 1) says which world an entity lives in.
#   identifier — const/name world: the binding's "value" is its own existence.
#   value      — variable/SSA world: each binding carries an actual value.
#   unknown    — vague/anaphoric surface not tied to a concrete entity; excluded
#                from the def-use layer unless Pass 2 coreference resolves it.
type EntityClass = Literal["identifier", "value", "unknown"]

# How a reference points at its entity.
#   direct  — the name appears verbatim (string-matchable).
#   anaphor — a pronoun/description ("this", "it", "the previous result"); carries
#             no entity until Pass 2 coreference resolves it (LLM).
type RefForm = Literal["direct", "anaphor"]

# The taint bit is `Reference.grounded` (tool-backed vs model-conjured). Whether an
# ungrounded use was bold (name in a tool call) vs idle (named in reasoning) is
# recoverable from `kind` — not a separate grade.
#
# Risk on a def-use edge (reaching binding -> use). One axis, name ⊂ value:
#   grounded    — reaching binding grounded and current; safe.
#   premature   — reaching binding ungrounded, but the entity IS grounded later and
#                 consistent (ran ahead of evidence, held up).
#   ungrounded  — entity never grounded anywhere; fabricated (name: fake id;
#                 value: made-up number).
#   contradicted — used an asserted value a later grounded binding differs from
#                  (value world; Pass 3.5).
#   stale       — used an older grounded version while a newer grounded one exists
#                 (value world).
type Risk = Literal["grounded", "premature", "ungrounded", "contradicted", "stale"]

# Runtime value sets — the single source the load path validates against (kept in
# lockstep with the Literal aliases).
_ENTITY_CLASS_VALUES: frozenset[str] = frozenset({"identifier", "value", "unknown"})
_REF_FORM_VALUES: frozenset[str] = frozenset({"direct", "anaphor"})
_RISK_VALUES: frozenset[str] = frozenset(
    {"grounded", "premature", "ungrounded", "contradicted", "stale"}
)

# ---------------------------------------------------------------------------
# Role enum (structural, not vocabulary-driven)
# ---------------------------------------------------------------------------


class StepRole:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Location:
    run_id: str
    step_id: str
    start: int
    end: int

    def contains(self, offset: int) -> bool:
        return self.start <= offset < self.end


@dataclass(frozen=True, slots=True)
class Step:
    run_id: str
    step_id: str
    index: int
    role: str
    content: str
    tool_name: str | None = None
    call_id: str | None = None
    timestamp: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    obs_regions: tuple[tuple[int, int], ...] = ()

    @property
    def observation_region_texts(self) -> tuple[str, ...]:
        """Each observation region's text, separately.

        Attested tool_result roles contribute their whole content as one
        region; otherwise the Pass 1 obs regions. Quote verification is
        per region (SCHEMA §2.3): a quote must sit inside ONE region,
        never spliced across a seam — verifiers use this, not the joined
        segment.
        """
        if self.role == StepRole.TOOL_RESULT:
            return (self.content,)
        return tuple(self.content[a:b] for a, b in self.obs_regions)

    @property
    def observation_segment(self) -> str | None:
        """The retrieved/environment portion of this step, or None.

        Pass 1 obs regions extend the evidence space to trajectories
        whose serialization lost structural roles (all downstream
        evidence selection reads this, never ``role`` directly).
        Multiple regions join with a newline — display and prompt use
        only; verification goes through ``observation_region_texts``.
        """
        texts = self.observation_region_texts
        if not texts:
            return None
        return "\n".join(texts)

    @property
    def action_segment(self) -> str | None:
        """The agent-authored portion of an assistant step, or None."""
        if self.role != StepRole.ASSISTANT:
            return None
        if not self.obs_regions:
            return self.content
        parts: list[str] = []
        pos = 0
        for a, b in self.obs_regions:
            if a > pos:
                parts.append(self.content[pos:a])
            pos = max(pos, b)
        if pos < len(self.content):
            parts.append(self.content[pos:])
        joined = "\n".join(p for p in parts if p.strip())
        return joined or None

    @property
    def provenance(self) -> str | None:
        """Derived display label: None / "mixed" / "observation"."""
        if not self.obs_regions:
            return None
        covered = sum(b - a for a, b in self.obs_regions)
        return "observation" if covered >= len(self.content.strip()) else "mixed"


@dataclass(slots=True)
class Symbol:
    id: str
    canonical_name: str
    kind: str = "unknown"
    aliases: set[str] = field(default_factory=set)
    summary: str | None = None
    definition_ref_id: str | None = None
    entity_class: EntityClass = "identifier"  # LLM (Pass 1): name vs value vs vague
    metadata: MutableMapping[str, MetadataValue] = field(default_factory=dict)

    @property
    def all_names(self) -> set[str]:
        return {self.canonical_name, *self.aliases}


@dataclass(frozen=True, slots=True)
class Reference:
    id: str
    symbol_id: str
    run_id: str
    step_id: str
    location: Location
    text: str
    role: str
    kind: str = "unknown"
    grounded: bool = False               # is this occurrence's value tool-backed?
    grounds_ref_id: str | None = None    # if grounded by copying a prior def, which
    structured: bool = True              # entity drives def-use? (entity_class != "unknown")
    form: RefForm = "direct"             # direct name vs anaphor (LLM flags anaphor)
    value: str | None = None             # value asserted/observed here (value entities)
    resolved_from: str | None = None     # original anaphor text, if Pass 2 rewrote it
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Dependency:
    """One def-use edge: a use step relied on a value an earlier def produced.

    ``risk`` is the grounding verdict on the use — whether the reaching def was
    tool-backed, ran ahead of grounding, or was never grounded. See
    SCHEMA.md for the build rule.
    """

    id: str
    symbol_id: str
    run_id: str
    def_step_id: str
    def_ref_id: str
    def_version: int          # SSA version of the reaching binding (0-based, per entity/run)
    use_step_id: str
    use_ref_id: str
    risk: Risk
    grounded_by_step_id: str | None = None   # if ungrounded at use, a later step that grounds it
    def_value: str | None = None             # value world: the two sides Pass 3.5 compares
    use_value: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Relation:
    id: str
    from_symbol_id: str
    to_symbol_id: str
    type: str
    run_id: str
    step_id: str
    weight: float = 1.0
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


# --- Constraint satisfaction layer (Pass 0/E/J/L; designs/constraint-satisfaction.md) ---

# Verdict for one (constraint, committed candidate) pair. "unknown" never
# escalates into a warning (P5); "omitted" requires the conjoined lexical +
# attested-coverage negatives (P4-iii).
type FindingStatus = Literal["verified", "violated", "omitted", "unknown"]

_FINDING_STATUS_VALUES: frozenset[str] = frozenset(
    {"verified", "violated", "omitted", "unknown"}
)


@dataclass(frozen=True, slots=True)
class Claim:
    """A settled-fact assertion by the agent, extracted verbatim in Pass 1.

    First-class extraction output alongside symbols: the trajectory is
    visited once and downstream passes (source-claim consistency,
    constraint linkage, commitment detection) consume the same claims —
    no per-consumer re-extraction. Verbatim presence in the step content
    is code-verified at populate time.
    """

    id: str
    run_id: str
    step_id: str
    text: str
    role: str = ""          # "" = plain assertion; "commit" = the final answer
    symbol_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Edge:
    """A Pass 2 relation between two index nodes.

    Pass 1 emits nodes; Pass 2 connects them. The model proposes edges,
    code verifies the decidable part: both endpoints exist, and ``quote``
    — the passage in the destination node's content that witnesses the
    relation — is verbatim-present there. An edge that fails verification
    is rejected and logged, never stored.

    ``kind`` names the relation. Current kinds: ``supports`` /
    ``conflicts`` — one claim against one observation excerpt; polarity
    sits on the edge because it is a local pairwise fact, while the
    global per-claim status is the Pass 3 fold's job (verification.py).

    ``evidence_position`` records the timeline fact (code-computed step
    comparison): the evidence precedes the claim, shares its step, or
    arrived after it. Consistency is time-agnostic, so no position is
    rejected — an after-conflicts edge is the "committed early, refuted
    later, never retracted" signature; an after-supports edge marks a
    premature commitment that happened to be right. Interpretation is
    the auditor's.
    """

    id: str
    kind: str
    run_id: str
    src: str                 # source node id (e.g. claim id)
    dst: str                 # destination node id (e.g. step id)
    quote: str = ""          # verbatim witness in the destination content
    evidence_position: str = ""   # before | same | after ("" = unknown/legacy)


@dataclass(frozen=True, slots=True)
class ClaimFinding:
    """Pass 3 output: one claim's evidence status, folded from edges by code.

    ``status`` values: ``supported`` (a supports edge exists), ``conflicted``
    (a conflicts edge exists — dominates), ``unsourced`` (the edge pass swept
    the WHOLE evidence space with complete coverage and found neither),
    ``unknown`` (coverage broken — never escalates, P5). ``evidence_empty``
    distinguishes "swept and found nothing" from "there was nothing to
    sweep" (a trajectory whose serialization carries no observation content
    at all — a strong trajectory-level fact in its own right).
    """

    claim_id: str
    run_id: str
    step_id: str
    status: str
    edge_ids: tuple[str, ...] = ()
    evidence_empty: bool = False


@dataclass(frozen=True, slots=True)
class Constraint:
    """An answer-level requirement extracted from the question (Pass 0).

    ``normalized`` is present only when the predicate is machine-checkable
    (dates, quantities) — then satisfaction is decided by code and the model
    only locates candidate values in text (P6).
    """

    id: str
    description: str
    normalized: Mapping[str, MetadataValue] | None = None
    symbol_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConstraintFinding:
    """Pass J/L verdict for one (constraint, committed candidate) pair.

    ``confidence`` is an ordinal ranking prior over findings, never a
    probability (oracle confidences are uncalibrated self-reports; code
    checks are ~1.0 — the min over mixed sources has no unit).
    ``confidence_source`` records what attained the min so the auditor can
    weigh source type instead of trusting the scalar.
    """

    constraint_id: str
    candidate: str
    status: FindingStatus
    evidence_step_ids: tuple[str, ...] = ()
    commit_step_id: str | None = None            # final commitment step (fact)
    first_assertion_step_id: str | None = None   # earliest assertion of the binding (fact)
    confidence: float = 1.0
    confidence_source: str = ""      # "code" | "oracle:<relation>"
    reason: str = ""


# ---------------------------------------------------------------------------
# Query / result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexStats:
    run_id: str
    step_count: int = 0
    symbol_count: int = 0
    reference_count: int = 0
    relation_count: int = 0
    dependency_count: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    symbol: Symbol
    score: float
    matched_fields: tuple[str, ...] = ()
    references: tuple[Reference, ...] = ()
    related: tuple[RelatedSymbol, ...] = ()


@dataclass(frozen=True, slots=True)
class RelatedSymbol:
    symbol: Symbol
    score: float
    relations: tuple[Relation, ...] = ()


@dataclass(frozen=True, slots=True)
class TimelineItem:
    step: Step
    reference: Reference


@dataclass(frozen=True, slots=True)
class ContextSnippet:
    focus_step: Step
    focus_ref: Reference
    before: tuple[Step, ...] = ()
    after: tuple[Step, ...] = ()


@dataclass(frozen=True, slots=True)
class SymbolContext:
    symbol: Symbol
    definition: Reference | None
    references: tuple[Reference, ...]
    related: tuple[RelatedSymbol, ...]
    timeline: tuple[TimelineItem, ...]
    snippets: tuple[ContextSnippet, ...]


@dataclass(frozen=True, slots=True)
class AliasCandidate:
    """A deterministically-blocked pair of structured symbols that MIGHT be the
    same entity. The merge decision is a name-resolution model judgment (Pass 2),
    not a rule — this only proposes the pair with enough context for it."""

    symbol_a_id: str
    symbol_b_id: str
    name_a: str
    name_b: str
    kind_a: str
    kind_b: str
    signal: str          # why blocked: "substring" | "similar"
    score: float         # normalized name similarity in [0, 1]
    context_a: tuple[str, ...] = ()   # sample reference snippets for the model
    context_b: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_NORM_WS = re.compile(r"[\s\-]+")
_NORM_STRIP = re.compile(r"[^a-z0-9_一-鿿.]+")


def normalize_name(text: str) -> str:
    text = text.strip().lower()
    text = _NORM_WS.sub("_", text)
    text = _NORM_STRIP.sub("", text)
    return text


def mentions_symbol(text: str, names: Iterable[str], min_len: int = 3) -> bool:
    """Word-bounded check: does *text* mention any of *names*?

    Shared blocking function for passes that need to group items by symbol
    mention (self-contradiction, constraint aboutness, etc.). The identity
    decision itself is upstream (Pass 1 extraction + Pass 2 alias merge);
    this is deterministic string matching on known surfaces.
    """
    low = text.lower()
    for n in names:
        if len(n) < min_len:
            continue
        if re.search(rf"(?<!\w){re.escape(n.lower())}(?!\w)", low):
            return True
    return False


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    raw = "||".join(str(p) for p in parts)
    digest = sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"
