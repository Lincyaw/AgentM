"""In-memory trajectory semantic index.

LSP-inspired symbol-reference-relation graph over agent trajectory steps.
Storage is in-memory; replace with ClickHouse/SQLite/vector-db later while
keeping the query interface stable.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

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
    TOOL_CALL = "tool_call"
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
    timestamp: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    # Pass 1 provenance: (start, end) content regions the extractor marked
    # as retrieved/environment material (``⟦obs|…⟧``), offset-exact via the
    # markup strip-and-compare verification. Multiple regions express the
    # query/content/summary sandwiches a single boundary cannot. Labels can
    # only ADD observation status to an assistant step — an attested
    # tool_result role always wins and needs no spans.
    obs_regions: tuple[tuple[int, int], ...] = ()

    @property
    def observation_segment(self) -> str | None:
        """The retrieved/environment portion of this step, or None.

        Attested tool_result roles contribute their whole content; Pass 1
        obs spans extend the evidence space to trajectories whose
        serialization lost structural roles (all downstream evidence
        selection reads this, never ``role`` directly). Multiple regions
        join with a newline.
        """
        if self.role == StepRole.TOOL_RESULT:
            return self.content
        if self.obs_regions:
            return "\n".join(self.content[a:b] for a, b in self.obs_regions)
        return None

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
    confidence: float = 1.0
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
    confidence: float = 1.0
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
    confidence: float = 1.0
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
    subject: str
    description: str
    normalized: Mapping[str, MetadataValue] | None = None


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


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    raw = "||".join(str(p) for p in parts)
    digest = sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


# --- Def-use / grounding helpers (Pass 3: dataflow, deterministic) ---------

# Reference kinds that produce/introduce a resource value (a "write"). The
# extractor only emits tool_output (data.py); define/write/observe are inert for
# freshly-built indexes and only matter for loaded/external ones. This is the
# file's single producing-kind vocabulary — DEFINITION_PREFERRED_KINDS aliases it.
_PRODUCING_KINDS: frozenset[str] = frozenset({"tool_output", "define", "write", "observe"})

_MISSING_STEP_INDEX = 10**12  # sort sentinel for a reference whose step is absent
_MIN_BLOCK_SUBSTR = 3         # alias blocking: shortest name a substring match may involve
_SNIPPET_CHARS = 120          # per-reference context snippet cap for the merge adjudicator

def drives_defuse(entity_class: str) -> bool:
    """Does an entity of this class drive the def-use layer?

    The name-vs-value-vs-vague judgment is the LLM's (Pass 1, ``entity_class``) —
    it knows a rigid identifier from an anaphor and a proper noun from a vague
    concept, which a regex cannot. Only ``unknown`` (vague, unresolved) is excluded;
    once Pass 2 coreference ties a vague surface to a concrete entity it inherits a
    real class. Replaces the old ``looks_structured`` regex gate.
    """
    return entity_class in ("identifier", "value")


def grounded_from_kind(kind: str) -> bool:
    """Base per-reference grounding from block-derived kind (no look-back).

    A tool-produced value is grounded; anything the model wrote (``tool_input``,
    ``mention``) starts ungrounded. A ``tool_input`` may still be upgraded to
    grounded during dependency build if it copies a prior grounded value.
    """
    return kind in _PRODUCING_KINDS


def _provenance_kind(step: Step, kind: str, start: int, end: int) -> str:
    """Upgrade a block-derived reference kind with Pass 1 provenance.

    On role-degraded records everything arrives as text blocks, so every
    occurrence is kind="mention" even inside a ``⟦obs⟧``-labeled region —
    which made the grounding layer call Pass-1-attested observation
    content "fabricated". An occurrence fully inside the step's
    observation spans IS tool output; the upgrade only ADDs observation
    status (attested roles keep their block-derived kind, including the
    deliberate non-deterministic downgrade).
    """
    if kind != "mention" or not step.obs_regions:
        return kind
    for a, b in step.obs_regions:
        if start >= a and end <= b:
            return "tool_output"
    return kind


def _constraint_normalized(attrs: dict[str, str]) -> dict[str, MetadataValue] | None:
    """Validate model-emitted machine-checkable attrs (code owns the types).

    Recognized shapes: kind=year_range lo=<int> hi=<int>;
    kind=number op∈{==,<=,>=,<,>} value=<number>. Anything else → None
    (the constraint stays semantic — the oracle judges it, code does not).
    """
    kind = attrs.get("kind", "")
    try:
        if kind == "year_range":
            return {"kind": "year_range", "lo": int(attrs["lo"]), "hi": int(attrs["hi"])}
        if kind == "number":
            op = attrs.get("op", "==")
            if op in {"==", "<=", ">=", "<", ">"}:
                return {"kind": "number", "op": op, "value": float(attrs["value"])}
    except (KeyError, ValueError):
        return None
    return None


def _message_step_content(msg: dict[str, Any]) -> tuple[str, str | None]:
    """Step content + tool name, from the single shared message walk.

    Derived from ``data.message_parts`` — the same pairs the extractor's
    view is built from — so the two representations that offset alignment
    depends on cannot drift apart.
    """
    from .data import message_parts

    pairs, tool_name = message_parts(msg)
    return "\n".join(c for c, _ in pairs if c), tool_name


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------


class TrajectoryIndex:
    """In-memory symbol-reference-relation index over trajectory steps."""

    DEFINITION_PREFERRED_KINDS: frozenset[str] = _PRODUCING_KINDS

    def __init__(self) -> None:
        self.steps: dict[tuple[str, str], Step] = {}

        self.symbols: dict[str, Symbol] = {}
        self._symbol_ids_by_norm: dict[NameKey, set[str]] = defaultdict(set)

        self.references: dict[str, Reference] = {}
        self._ref_ids_by_symbol: dict[str, list[str]] = defaultdict(list)
        self._ref_ids_by_step: dict[tuple[str, str], list[str]] = defaultdict(list)

        self.relations: dict[str, Relation] = {}
        self._relation_ids_by_symbol: dict[str, list[str]] = defaultdict(list)

        # Def-use / grounding layer (Pass 3). Built by build_dependencies().
        self.dependencies: dict[str, Dependency] = {}
        self._dep_ids_by_symbol: dict[str, list[str]] = defaultdict(list)

        # Claims (Pass 1 output, verbatim-verified). Keyed by claim id.
        self.claims: dict[str, Claim] = {}

        # Edges (Pass 2 output, endpoint- and quote-verified). Keyed by edge id.
        self.edges: dict[str, Edge] = {}

        # Claim statuses (Pass 3 output, pure-code fold over edges + coverage).
        self.claim_findings: list[ClaimFinding] = []

        # Constraint layer (Pass 0/E/J/L). Populated by constraints.analyze_constraints().
        self.constraints: dict[str, Constraint] = {}
        self.constraint_findings: list[ConstraintFinding] = []

        self.indexed_message_count: int = 0

    # ---- write path ----

    @staticmethod
    def _name_key(name: str, namespace: str = "") -> NameKey:
        return (namespace, normalize_name(name))

    def _index_symbol_name(self, symbol_id: str, name: str, namespace: str = "") -> None:
        self._symbol_ids_by_norm[self._name_key(name, namespace)].add(symbol_id)

    def add_step(self, step: Step) -> None:
        self.steps[(step.run_id, step.step_id)] = step

    def upsert_symbol(
        self,
        name: str,
        kind: str = "unknown",
        summary: str | None = None,
        aliases: Sequence[str] = (),
        namespace: str = "",
        entity_class: EntityClass | str = "identifier",
    ) -> Symbol:
        if entity_class not in _ENTITY_CLASS_VALUES:
            entity_class = "identifier"
        norm_key = self._name_key(name, namespace)
        existing_ids = self._symbol_ids_by_norm.get(norm_key)
        if existing_ids:
            symbol_id = sorted(existing_ids)[0]
            symbol = self.symbols[symbol_id]
            if kind != "unknown" and symbol.kind == "unknown":
                symbol.kind = kind
            if summary and not symbol.summary:
                symbol.summary = summary
            # a concrete class wins over a prior "unknown" (a later step pinned it down)
            if entity_class != "unknown" and symbol.entity_class == "unknown":
                symbol.entity_class = entity_class  # type: ignore[assignment]
            for alias in aliases:
                symbol.aliases.add(alias)
                self._index_symbol_name(symbol_id, alias, namespace)
            return symbol

        norm = normalize_name(name)
        symbol_id = stable_id("sym", namespace, norm) if namespace else stable_id("sym", norm)
        metadata: dict[str, MetadataValue] = {}
        if namespace:
            metadata["namespace"] = namespace
        symbol = Symbol(
            id=symbol_id,
            canonical_name=name.strip(),
            kind=kind,
            summary=summary,
            aliases=set(aliases),
            entity_class=entity_class,  # type: ignore[arg-type]
            metadata=metadata,
        )
        self.symbols[symbol_id] = symbol
        self._index_symbol_name(symbol_id, name, namespace)
        for alias in aliases:
            self._index_symbol_name(symbol_id, alias, namespace)
        return symbol

    def add_reference(
        self,
        symbol: Symbol,
        step: Step,
        text: str,
        kind: str = "unknown",
        start: int = 0,
        end: int | None = None,
        confidence: float = 1.0,
        form: RefForm = "direct",
        value: str | None = None,
        resolved_from: str | None = None,
    ) -> Reference:
        if end is None:
            end = start + len(text)
        kind = _provenance_kind(step, kind, start, end)
        loc = Location(step.run_id, step.step_id, start, end)
        ref_id = stable_id(
            "ref", loc.run_id, loc.step_id, loc.start, loc.end, symbol.id,
        )
        if ref_id in self.references:
            return self.references[ref_id]

        ref = Reference(
            id=ref_id,
            symbol_id=symbol.id,
            run_id=step.run_id,
            step_id=step.step_id,
            location=loc,
            text=text,
            role=step.role,
            kind=kind,
            confidence=confidence,
            grounded=grounded_from_kind(kind),
            structured=drives_defuse(symbol.entity_class),
            form=form,
            value=value,
            resolved_from=resolved_from,
        )
        self.references[ref_id] = ref
        self._ref_ids_by_symbol[symbol.id].append(ref_id)
        self._ref_ids_by_step[(step.run_id, step.step_id)].append(ref_id)

        if symbol.definition_ref_id is None:
            symbol.definition_ref_id = ref_id
        elif kind in self.DEFINITION_PREFERRED_KINDS:
            current = self.references.get(symbol.definition_ref_id)
            if current and current.kind not in self.DEFINITION_PREFERRED_KINDS:
                symbol.definition_ref_id = ref_id

        return ref

    def add_relation(
        self,
        from_symbol: Symbol,
        to_symbol: Symbol,
        rel_type: str,
        step: Step,
        weight: float = 1.0,
        confidence: float = 1.0,
    ) -> Relation:
        relation_id = stable_id(
            "rel", from_symbol.id, to_symbol.id, rel_type,
        )
        if relation_id in self.relations:
            return self.relations[relation_id]

        relation = Relation(
            id=relation_id,
            from_symbol_id=from_symbol.id,
            to_symbol_id=to_symbol.id,
            type=rel_type,
            run_id=step.run_id,
            step_id=step.step_id,
            weight=weight,
            confidence=confidence,
        )
        self.relations[relation_id] = relation
        self._relation_ids_by_symbol[from_symbol.id].append(relation_id)
        self._relation_ids_by_symbol[to_symbol.id].append(relation_id)
        return relation

    # ---- alias resolution: deterministic candidates + merge mechanism
    #      the merge DECISION is a name-resolution model judgment (Pass 2), injected via apply_alias_merges

    def _rebuild_symbol_name_index(self) -> None:
        self._symbol_ids_by_norm = defaultdict(set)
        for sid, sym in self.symbols.items():
            ns = str(sym.metadata.get("namespace", ""))
            for name in sym.all_names:
                self._index_symbol_name(sid, name, ns)

    def _rebuild_relation_index(self) -> None:
        self._relation_ids_by_symbol = defaultdict(list)
        for rid, rel in self.relations.items():
            self._relation_ids_by_symbol[rel.from_symbol_id].append(rid)
            self._relation_ids_by_symbol[rel.to_symbol_id].append(rid)

    def _choose_canonical(self, symbol_ids: list[str]) -> str:
        """Representative of a merge group: most-referenced, tie broken by stable id.
        Which id represents the merged entity is cosmetic; the choice is deterministic."""
        return sorted(
            symbol_ids, key=lambda sid: (-len(self._ref_ids_by_symbol.get(sid, [])), sid),
        )[0]

    def _ref_snippets(self, symbol_id: str, limit: int = 3) -> tuple[str, ...]:
        """A few reference texts for a symbol — context for the merge adjudicator."""
        out: list[str] = []
        for rid in self._ref_ids_by_symbol.get(symbol_id, [])[:limit]:
            text = self.references[rid].text.strip()
            if text:
                out.append(text[:_SNIPPET_CHARS])
        return tuple(out)

    @staticmethod
    def _tokenize_name(name: str) -> set[str]:
        """Split a symbol name into tokens on common delimiters."""
        return {
            t for t in re.split(r"[-_./\s]+", normalize_name(name))
            if len(t) >= 2
        }

    @staticmethod
    def _token_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def alias_candidates(self, min_jaccard: float = 0.5) -> list[AliasCandidate]:
        """Deterministically block structured-symbol pairs that MIGHT be one entity.

        Uses token-level Jaccard similarity on names split by common
        delimiters (``-_./``). This avoids false matches between names
        that share long prefixes/suffixes but differ in the distinctive
        token (e.g. ``ts-cancel-service`` vs ``ts-config-service``
        shares only ``{ts, service}`` → jaccard 0.33, below threshold).

        Substring containment is also checked. Pairs above threshold are
        proposed for LLM judgment via :meth:`apply_alias_merges`.
        """
        items = [
            (sid, sym, normalize_name(sym.canonical_name),
             self._tokenize_name(sym.canonical_name), self._ref_snippets(sid))
            for sid, sym in self.symbols.items()
            if drives_defuse(sym.entity_class)
        ]
        out: list[AliasCandidate] = []
        for i in range(len(items)):
            aid, asym, anorm, atokens, asnip = items[i]
            for j in range(i + 1, len(items)):
                bid, bsym, bnorm, btokens, bsnip = items[j]
                if anorm == bnorm:
                    continue
                substring = (
                    (anorm in bnorm or bnorm in anorm)
                    and min(len(anorm), len(bnorm)) >= _MIN_BLOCK_SUBSTR
                )
                jaccard = self._token_jaccard(atokens, btokens)
                if not substring and jaccard < min_jaccard:
                    continue
                out.append(AliasCandidate(
                    symbol_a_id=aid, symbol_b_id=bid,
                    name_a=asym.canonical_name, name_b=bsym.canonical_name,
                    kind_a=asym.kind, kind_b=bsym.kind,
                    signal="substring" if substring else "similar",
                    score=round(jaccard, 3),
                    context_a=asnip, context_b=bsnip,
                ))
        return sorted(out, key=lambda c: (-c.score, c.symbol_a_id, c.symbol_b_id))

    def apply_alias_merges(self, groups: list[list[str]]) -> None:
        """Fold each decided group of symbol ids into one canonical symbol.

        The mechanism (deterministic, idempotent); *which* symbols form a group is
        decided upstream — a name-resolution judgment over :meth:`alias_candidates`, not
        a rule here. References and relations are re-pointed; folded names become
        aliases.
        """
        merged = False
        for sids in groups:
            live = [s for s in dict.fromkeys(sids) if s in self.symbols]
            if len(live) < 2:
                continue
            canonical_id = self._choose_canonical(live)
            canon = self.symbols[canonical_id]
            for other_id in live:
                if other_id == canonical_id:
                    continue
                other = self.symbols[other_id]
                canon.aliases.update(other.all_names)
                canon.aliases.discard(canon.canonical_name)
                for rid in self._ref_ids_by_symbol.get(other_id, []):
                    self.references[rid] = replace(self.references[rid], symbol_id=canonical_id)
                    self._ref_ids_by_symbol[canonical_id].append(rid)
                self._ref_ids_by_symbol.pop(other_id, None)
                # only relations touching other_id need re-pointing (a folded symbol
                # is never a canonical target, so this index entry is complete).
                for rel_id in dict.fromkeys(self._relation_ids_by_symbol.get(other_id, [])):
                    rel = self.relations.get(rel_id)
                    if rel is None:
                        continue
                    nf = canonical_id if rel.from_symbol_id == other_id else rel.from_symbol_id
                    nt = canonical_id if rel.to_symbol_id == other_id else rel.to_symbol_id
                    if nf == nt:  # self-loop after the merge — drop
                        self.relations.pop(rel_id, None)
                    else:
                        self.relations[rel_id] = replace(rel, from_symbol_id=nf, to_symbol_id=nt)
                self.symbols.pop(other_id, None)
                merged = True

        if merged:
            self._rebuild_symbol_name_index()
            self._rebuild_relation_index()
            # dependencies are derived from symbols; a merge invalidates them.
            self.dependencies = {}
            self._dep_ids_by_symbol = defaultdict(list)
            # constraint findings name candidate symbols; a merge invalidates
            # them wholesale (Pass E/J/L rebuild from facts + transcript).
            self.constraint_findings = []

    # ---- def-use / grounding layer (Pass 3: dataflow, deterministic) ----

    def _ref_step_index(self, ref: Reference) -> int:
        step = self.steps.get((ref.run_id, ref.step_id))
        return step.index if step else _MISSING_STEP_INDEX

    def build_dependencies(self) -> None:
        """Build the def-use layer over structured entities (Pass 3: dataflow).

        Idempotent: clears and rebuilds every edge. Deterministic (no model),
        global traversal. See SCHEMA.md for the def/use classification,
        reaching-def selection with forward grounding propagation, and the
        grounded/premature/ungrounded risk derivation.

        Name resolution (Pass 2) is a separate, upstream step: run apply_alias_merges()
        with the model-decided groups from alias_candidates() *before* this, if
        desired. This method does no merging on its own.
        """
        self.dependencies = {}
        self._dep_ids_by_symbol = defaultdict(list)

        # Reset tool_input grounding upgrades from prior builds so the rebuild
        # is truly idempotent — without this, a re-run sees already-upgraded
        # refs and may flip risk labels (premature -> grounded).
        for ref_id, ref in self.references.items():
            if ref.grounds_ref_id is not None:
                self.references[ref_id] = replace(
                    ref, grounded=grounded_from_kind(ref.kind), grounds_ref_id=None,
                )

        for symbol_id in self.symbols:
            refs = self.get_references(symbol_id)  # sorted (run_id, step index, start)
            if not refs or not any(r.structured for r in refs):
                continue  # only structured entities drive the def-use layer
            by_run: dict[str, list[Reference]] = defaultdict(list)
            for r in refs:
                by_run[r.run_id].append(r)
            for run_id, run_refs in by_run.items():
                self._build_run_dependencies(symbol_id, run_id, run_refs)

    def _build_run_dependencies(
        self, symbol_id: str, run_id: str, refs: list[Reference],
    ) -> None:
        """Chain one symbol's references within a single run into def-use edges.

        The reaching def for a use is the most-recent def at a *strictly earlier*
        step (most-recent at a strictly earlier step). Defs in the use's own step do not
        count — a same-step def/use is not cross-step reliance, so it produces no
        edge. Defs are therefore committed only when we cross a step boundary;
        within a step they buffer in ``pending``.
        """
        # (step_index, step_id) of every grounded def, for the look-ahead that tells
        # "used before grounded, but grounded later" (premature) from "never grounded".
        grounded_defs: list[tuple[int, str]] = sorted(
            (self._ref_step_index(r), r.step_id) for r in refs if r.grounded
        )

        reaching: Reference | None = None          # committed from a strictly earlier step
        grounded_reaching: Reference | None = None
        pending: Reference | None = None           # latest def in the current step
        pending_grounded: Reference | None = None
        cur_step: int | None = None
        reaching_version = -1                       # SSA version of `reaching`

        for ref in refs:
            use_idx = self._ref_step_index(ref)
            if use_idx != cur_step:
                # crossed a step boundary: the prior step's defs become reaching
                if pending is not None:
                    reaching = pending
                    reaching_version += 1
                if pending_grounded is not None:
                    grounded_reaching = pending_grounded
                pending = pending_grounded = None
                cur_step = use_idx

            # A tool_input that copies a prior grounded def is itself grounded.
            if (
                ref.kind == "tool_input"
                and not ref.grounded
                and grounded_reaching is not None
            ):
                ref = replace(ref, grounded=True, grounds_ref_id=grounded_reaching.id)
                self.references[ref.id] = ref

            is_def = ref.kind in _PRODUCING_KINDS
            if is_def:
                pending = ref
                if ref.grounded:
                    pending_grounded = ref
                continue

            # use → link to the most-recent def at a strictly earlier step
            if reaching is None:
                continue
            grounded_by: str | None = None
            if reaching.grounded:
                risk: Risk = "grounded"
                grounded_by = reaching.step_id
            else:
                grounded_by = next(
                    (sid for idx, sid in grounded_defs if idx > use_idx), None,
                )
                risk = "premature" if grounded_by else "ungrounded"

            # ref ids discriminate multiple uses of one reaching def in a step
            dep_id = stable_id("dep", run_id, reaching.id, ref.id, symbol_id)
            dep = Dependency(
                id=dep_id,
                symbol_id=symbol_id,
                run_id=run_id,
                def_step_id=reaching.step_id,
                def_ref_id=reaching.id,
                def_version=max(0, reaching_version),
                use_step_id=ref.step_id,
                use_ref_id=ref.id,
                risk=risk,
                grounded_by_step_id=grounded_by,
                def_value=reaching.value,
                use_value=ref.value,
                confidence=min(reaching.confidence, ref.confidence),
            )
            self.dependencies[dep.id] = dep
            self._dep_ids_by_symbol[symbol_id].append(dep.id)

    def get_dependencies(self, symbol_id: str = "") -> list[Dependency]:
        """All dependency edges, or those for one symbol, in run/step order."""
        if symbol_id:
            deps = [self.dependencies[d] for d in self._dep_ids_by_symbol.get(symbol_id, [])]
        else:
            deps = list(self.dependencies.values())
        return sorted(deps, key=lambda d: (d.run_id, d.use_step_id, d.symbol_id))

    def registry_snapshot(self) -> list[dict[str, str | list[str]]]:
        """Compact symbol registry for incremental extraction prompts."""
        out: list[dict[str, str | list[str]]] = []
        for s in self.symbols.values():
            entry: dict[str, str | list[str]] = {
                "name": s.canonical_name,
                "kind": s.kind,
                "entity_class": s.entity_class,
            }
            if s.aliases:
                entry["aliases"] = sorted(s.aliases)
            out.append(entry)
        return out

    def resolve_symbol_by_name(self, name: str, namespace: str = "") -> Symbol | None:
        """Look up a symbol by name or alias (normalized match)."""
        if namespace:
            ids = self._symbol_ids_by_norm.get(self._name_key(name, namespace))
            if ids:
                return self.symbols[sorted(ids)[0]]
        ids = self._symbol_ids_by_norm.get(self._name_key(name))
        if ids:
            return self.symbols[sorted(ids)[0]]
        return None

    def populate_from_extraction(
        self,
        result: Any,
        messages: list[Any],
        *,
        run_id: str = "",
        namespace_fn: Any | None = None,
        reference_confidence: float = 0.8,
        message_id_start: int = 0,
        diagnostics: Any | None = None,
    ) -> None:
        """Populate index from an extraction result and source messages.

        ``result`` carries the unified markup output (``.annotated`` — see
        ``markup.py``); every Pass 1 node kind is parsed out of it here.
        Verification is strip-and-compare: the annotated body, with all
        ``⟦…⟧`` removed, must reproduce the extractor's view of the message
        (whitespace-tolerant), which makes every span offset exact. A
        message that fails is rejected whole; every rejection lands in
        ``diagnostics.prune_log`` when a sink is passed (P2), and in the
        debug log always.

        ``messages`` can be typed ``AgentMessage`` objects or pre-serialized
        dicts. ``namespace_fn(run_id, sym_dict) -> str`` optionally scopes
        symbols. ``message_id_start`` is the absolute offset of the first
        message in the full trajectory.
        """
        from loguru import logger as _clog

        from .data import _build_references, view_body_with_map
        from .markup import GAP_TAG, MarkupError, align, align_gapped
        from .markup import parse as parse_markup

        if messages and not isinstance(messages[0], dict):
            from .atom import _agentmsg_to_extraction_dict
            # truncate=False: steps/references are the substrate every
            # downstream pass reads — content goes in whole. Truncation is
            # legitimate only in the extractor's own prompt window.
            messages = [
                d for i, m in enumerate(messages, start=message_id_start)
                if (d := _agentmsg_to_extraction_dict(m, i, truncate=False))
            ]

        def _prune(what: str, why: str) -> None:
            if diagnostics is not None:
                diagnostics.prune("populate", what, why)
            _clog.debug("populate: {} — {}", what, why)

        role_map = {"user": StepRole.USER, "assistant": StepRole.ASSISTANT, "tool_result": StepRole.TOOL_RESULT}
        base_idx = len(self.steps)
        steps_by_id: dict[str, Step] = {}

        ann_by_mid: dict[str, str] = {
            str(am.message_id): am.text
            for am in getattr(result, "annotated", []) or []
        }

        # node collections parsed out of the markup, keyed by message id
        obs_by_mid: dict[str, tuple[tuple[int, int], ...]] = {}
        claims_by_mid: dict[str, list[tuple[int, int, str]]] = {}
        constraints_new: list[tuple[str, int, int, dict[str, str]]] = []
        syms: list[tuple[str, dict[str, str]]] = []   # (surface, attrs)

        for i, msg in enumerate(messages):
            mid = str(msg.get("id", f"s{base_idx + i}"))
            annotated = ann_by_mid.pop(mid, None)
            if annotated is None:
                continue
            try:
                plain, annotations = parse_markup(annotated)
            except MarkupError as exc:
                _prune(f"step {mid}", f"malformed markup: {exc}")
                continue
            view, vmap = view_body_with_map(msg)
            gap_offsets = [a.start for a in annotations if a.tag == GAP_TAG]
            annotations = [a for a in annotations if a.tag != GAP_TAG]
            if gap_offsets:
                amap, gap_err = align_gapped(plain, gap_offsets, view)
                if amap is None:
                    _prune(f"step {mid}", f"gapped re-emission rejected: {gap_err}")
                    continue
            else:
                amap = align(plain, view)
                if amap is None:
                    _prune(f"step {mid}",
                           f"re-emission diverges from original ({len(plain)} vs {len(view)} chars)")
                    continue

            def _to_content(
                plain_off: int,
                _amap: list[int] = amap,
                _vmap: list[int | None] = vmap,
            ) -> int | None:
                return _vmap[_amap[plain_off]]

            obs_regions: list[tuple[int, int]] = []
            for a in annotations:
                if a.tag == "known" or a.end <= a.start:
                    continue
                start_c = _to_content(a.start)
                end_c = _to_content(a.end)
                if end_c is None:
                    last = _to_content(a.end - 1)
                    end_c = last + 1 if last is not None else None
                if start_c is None or end_c is None:
                    _prune(f"step {mid}",
                           f"⟦{a.tag}⟧ span falls in the truncation ellipsis")
                    continue
                if a.tag == "obs":
                    obs_regions.append((start_c, end_c))
                elif a.tag == "claim":
                    claims_by_mid.setdefault(mid, []).append(
                        (start_c, end_c, str(a.attrs.get("role", ""))),
                    )
                elif a.tag == "constraint":
                    constraints_new.append((mid, start_c, end_c, dict(a.attrs)))
                elif a.tag == "sym":
                    syms.append((plain[a.start:a.end], dict(a.attrs)))
                else:
                    _prune(f"step {mid}", f"unknown annotation tag {a.tag!r}")
            if obs_regions:
                obs_regions.sort()
                merged: list[tuple[int, int]] = []
                for span in obs_regions:
                    if merged and span[0] <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
                    else:
                        merged.append(span)
                obs_by_mid[mid] = tuple(merged)

        for mid in ann_by_mid:
            _prune(f"step {mid}", "annotated message references unknown id")

        for i, msg in enumerate(messages):
            mid = str(msg.get("id", f"s{base_idx + i}"))
            role = role_map.get(str(msg.get("role", "")), StepRole.USER)
            content, tool_name = _message_step_content(msg)
            spans = obs_by_mid.get(mid, ())
            if spans and role == StepRole.TOOL_RESULT:
                spans = ()   # attested — the whole content already counts
            step = Step(
                run_id=run_id,
                step_id=mid,
                index=int(mid) if mid.isdigit() else base_idx + i,
                role=role,
                content=content,
                tool_name=tool_name,
                obs_regions=spans,
            )
            self.add_step(step)
            steps_by_id[mid] = step

        # Symbols: canonical name from the ``name`` attr when the surface is
        # not canonical; the surface itself becomes an alias. Aliases are no
        # longer hand-listed — they emerge from marked surfaces.
        for surface, attrs in syms:
            canonical = (attrs.get("name") or surface).strip()
            if not canonical:
                continue
            aliases = (
                [surface.strip()]
                if surface.strip() and normalize_name(surface) != normalize_name(canonical)
                else []
            )
            kind = attrs.get("kind", "unknown").lower()
            raw_class = attrs.get("class", "identifier")
            entity_class: EntityClass = (
                raw_class if raw_class in _ENTITY_CLASS_VALUES else "identifier"  # type: ignore[assignment]
            )
            sym_dict = {"name": canonical, "kind": kind, "aliases": aliases}
            ns = namespace_fn(run_id, sym_dict) if namespace_fn else ""
            self.upsert_symbol(
                name=canonical, kind=kind, aliases=aliases,
                namespace=ns, entity_class=entity_class,
            )

        # Claims: text is an exact content slice — verbatim by construction.
        for mid, spans_list in claims_by_mid.items():
            cstep = steps_by_id.get(mid)
            if cstep is None:
                continue
            for start_c, end_c, role in spans_list:
                text = cstep.content[start_c:end_c].strip()
                if not text:
                    continue
                # start offset discriminates same-prefix claims within a step
                cid = stable_id("clm", run_id, mid, start_c, text[:80])
                self.claims[cid] = Claim(
                    id=cid, run_id=run_id, step_id=mid, text=text, role=role,
                )

        # Constraints: task requirements, verbatim; machine-checkable
        # normalization comes from model-emitted attrs, code-validated
        # (unparseable attrs degrade to a semantic constraint, logged).
        for mid, start_c, end_c, attrs in constraints_new:
            nstep = steps_by_id.get(mid)
            if nstep is None:
                continue
            text = nstep.content[start_c:end_c].strip()
            if not text:
                continue
            normalized = _constraint_normalized(attrs)
            if attrs.get("kind") and normalized is None:
                _prune(f"step {mid}",
                       f"constraint attrs unparseable, kept as semantic: {attrs!r}")
            conid = stable_id("con", run_id, mid, start_c, text[:80])
            self.constraints[conid] = Constraint(
                id=conid,
                subject=attrs.get("subject", "answer"),
                description=text,
                normalized=normalized,
            )

        all_syms = self.registry_snapshot()
        namespaces = {str(s["name"]): namespace_fn(run_id, s) if namespace_fn else "" for s in all_syms}
        refs = _build_references(all_syms, messages)
        for ref in refs:
            rsym = self.resolve_symbol_by_name(
                ref.symbol_name,
                namespace=namespaces.get(ref.symbol_name, ""),
            )
            rstep = steps_by_id.get(ref.turn_id)
            if rsym and rstep:
                self.add_reference(
                    symbol=rsym, step=rstep, text=ref.text,
                    kind=ref.kind, start=ref.start,
                    confidence=reference_confidence,
                )

    def stats(self, run_id: str = "") -> IndexStats:
        return IndexStats(
            run_id=run_id,
            step_count=len(self.steps),
            symbol_count=len(self.symbols),
            reference_count=len(self.references),
            relation_count=len(self.relations),
            dependency_count=len(self.dependencies),
        )

    # ---- warnings (code-only, no LLM) ----

    @dataclass(frozen=True, slots=True)
    class Warning:
        kind: str
        symbol_id: str
        symbol_name: str
        detail: str
        step_ids: tuple[str, ...] = ()

    def warnings(self) -> list[Warning]:
        """Compute grounding warnings from the index structure.

        Pure code — no LLM. Runs after ``build_dependencies()``.
        """
        out: list[TrajectoryIndex.Warning] = []

        for sym_id, sym in self.symbols.items():
            refs = self._ref_ids_by_symbol.get(sym_id, [])
            if not refs:
                out.append(self.Warning(
                    kind="orphan",
                    symbol_id=sym_id,
                    symbol_name=sym.canonical_name,
                    detail="extracted but has 0 references in the trajectory",
                ))
                continue

            ref_objs = [self.references[rid] for rid in refs]
            ref_kinds = {r.kind for r in ref_objs}
            has_grounded_def = any(r.grounded for r in ref_objs)

            if not has_grounded_def:
                steps = tuple(dict.fromkeys(r.step_id for r in ref_objs))
                if ref_kinds == {"tool_input"} or ref_kinds <= {"tool_input", "mention"}:
                    if "tool_input" in ref_kinds:
                        out.append(self.Warning(
                            kind="blind_query",
                            symbol_id=sym_id,
                            symbol_name=sym.canonical_name,
                            detail="used in tool calls but never returned by any tool",
                            step_ids=steps,
                        ))
                    else:
                        out.append(self.Warning(
                            kind="fabricated_name",
                            symbol_id=sym_id,
                            symbol_name=sym.canonical_name,
                            detail="mentioned in reasoning but never returned by any tool",
                            step_ids=steps,
                        ))

        # Dependency-level warnings
        deps = self.get_dependencies()
        premature = [d for d in deps if d.risk == "premature"]
        ungrounded = [d for d in deps if d.risk == "ungrounded"]

        for d in premature:
            dsym = self.symbols.get(d.symbol_id)
            if dsym:
                out.append(self.Warning(
                    kind="premature_use",
                    symbol_id=d.symbol_id,
                    symbol_name=dsym.canonical_name,
                    detail=f"used at step {d.use_step_id} before grounded at step {d.grounded_by_step_id}",
                    step_ids=(d.use_step_id, d.def_step_id),
                ))

        for d in ungrounded:
            dsym = self.symbols.get(d.symbol_id)
            if dsym:
                out.append(self.Warning(
                    kind="ungrounded_use",
                    symbol_id=d.symbol_id,
                    symbol_name=dsym.canonical_name,
                    detail=f"used at step {d.use_step_id} with ungrounded def at step {d.def_step_id}",
                    step_ids=(d.use_step_id, d.def_step_id),
                ))

        return sorted(out, key=lambda w: (
            {"fabricated_name": 0, "blind_query": 1, "ungrounded_use": 2, "premature_use": 3, "orphan": 4}.get(w.kind, 5),
            w.symbol_name,
        ))

    def warning_summary(self) -> dict[str, int]:
        """Count warnings by kind."""
        from collections import Counter
        return dict(Counter(w.kind for w in self.warnings()))

    def constraint_attention(self) -> list[dict[str, str]]:
        """Constraint findings as attention hints (violated / omitted only).

        Kept separate from :meth:`warnings` — which stays pure code — because
        these stand on recorded oracle tuples. The caller merges them into
        the auditor's attention-hint feed.
        """
        out: list[dict[str, str]] = []
        for f in self.constraint_findings:
            if f.status not in ("violated", "omitted"):
                continue
            c = self.constraints.get(f.constraint_id)
            desc = c.description if c else f.constraint_id
            summary = f"constraint '{desc}' {f.status} for candidate '{f.candidate}'"
            if f.reason:
                summary += f" — {f.reason}"
            out.append({
                "kind": f"constraint_{f.status}",
                "summary": summary,
                "symbol": f.candidate,
                "step_id": f.first_assertion_step_id or f.commit_step_id or "",
            })
        return out

    # ---- integrity ----

    def validate(self) -> list[str]:
        """Check referential integrity. Returns a list of error descriptions."""
        errors: list[str] = []
        for sym in self.symbols.values():
            if sym.definition_ref_id and sym.definition_ref_id not in self.references:
                errors.append(f"symbol {sym.id} definition_ref_id {sym.definition_ref_id} not found")
        for ref in self.references.values():
            if ref.symbol_id not in self.symbols:
                errors.append(f"reference {ref.id} points to missing symbol {ref.symbol_id}")
            if ref.grounds_ref_id and ref.grounds_ref_id not in self.references:
                errors.append(f"reference {ref.id} grounds_ref_id {ref.grounds_ref_id} not found")
        for rel in self.relations.values():
            if rel.from_symbol_id not in self.symbols:
                errors.append(f"relation {rel.id} from_symbol {rel.from_symbol_id} not found")
            if rel.to_symbol_id not in self.symbols:
                errors.append(f"relation {rel.id} to_symbol {rel.to_symbol_id} not found")
        for dep in self.dependencies.values():
            if dep.symbol_id not in self.symbols:
                errors.append(f"dependency {dep.id} symbol {dep.symbol_id} not found")
            if dep.def_ref_id not in self.references:
                errors.append(f"dependency {dep.id} def_ref {dep.def_ref_id} not found")
            if dep.use_ref_id not in self.references:
                errors.append(f"dependency {dep.id} use_ref {dep.use_ref_id} not found")
            if dep.def_ref_id in self.references and self.references[dep.def_ref_id].symbol_id != dep.symbol_id:
                errors.append(f"dependency {dep.id} def_ref belongs to {self.references[dep.def_ref_id].symbol_id}, not {dep.symbol_id}")
            if dep.use_ref_id in self.references and self.references[dep.use_ref_id].symbol_id != dep.symbol_id:
                errors.append(f"dependency {dep.id} use_ref belongs to {self.references[dep.use_ref_id].symbol_id}, not {dep.symbol_id}")
        return errors

    # ---- persistence ----

    def dump(self, path: str | Path) -> None:
        """Write the full index to a JSON file. Validates integrity first."""
        errors = self.validate()
        if errors:
            from loguru import logger

            for e in errors:
                logger.warning(f"index integrity: {e}")
        steps = [
            {
                "run_id": s.run_id,
                "step_id": s.step_id,
                "index": s.index,
                "role": s.role,
                "content": s.content,
                "tool_name": s.tool_name,
                "timestamp": s.timestamp,
                "obs_regions": [list(span) for span in s.obs_regions],
                "metadata": dict(s.metadata) if s.metadata else {},
            }
            for s in self.steps.values()
        ]
        symbols = [
            {
                "id": s.id,
                "name": s.canonical_name,
                "kind": s.kind,
                "aliases": sorted(s.aliases) if s.aliases else [],
                "summary": s.summary,
                "definition_ref_id": s.definition_ref_id,
                "entity_class": s.entity_class,
                "namespace": str(s.metadata.get("namespace", "")),
                "metadata": dict(s.metadata) if s.metadata else {},
            }
            for s in self.symbols.values()
        ]
        refs = [
            {
                "id": r.id,
                "symbol_id": r.symbol_id,
                "run_id": r.run_id,
                "step_id": r.step_id,
                "start": r.location.start,
                "end": r.location.end,
                "text": r.text,
                "role": r.role,
                "kind": r.kind,
                "confidence": r.confidence,
                "grounded": r.grounded,
                "grounds_ref_id": r.grounds_ref_id,
                "structured": r.structured,
                "form": r.form,
                "value": r.value,
                "resolved_from": r.resolved_from,
                "metadata": dict(r.metadata) if r.metadata else {},
            }
            for r in self.references.values()
        ]
        relations = [
            {
                "id": r.id,
                "from": r.from_symbol_id,
                "to": r.to_symbol_id,
                "type": r.type,
                "run_id": r.run_id,
                "step_id": r.step_id,
                "weight": r.weight,
                "confidence": r.confidence,
                "metadata": dict(r.metadata) if r.metadata else {},
            }
            for r in self.relations.values()
        ]
        dependencies = [
            {
                "id": d.id,
                "symbol_id": d.symbol_id,
                "run_id": d.run_id,
                "def_step_id": d.def_step_id,
                "def_ref_id": d.def_ref_id,
                "def_version": d.def_version,
                "use_step_id": d.use_step_id,
                "use_ref_id": d.use_ref_id,
                "risk": d.risk,
                "grounded_by_step_id": d.grounded_by_step_id,
                "def_value": d.def_value,
                "use_value": d.use_value,
                "confidence": d.confidence,
                "metadata": dict(d.metadata) if d.metadata else {},
            }
            for d in self.dependencies.values()
        ]
        claims = [
            {"id": c.id, "run_id": c.run_id, "step_id": c.step_id, "text": c.text, "role": c.role}
            for c in self.claims.values()
        ]
        edges = [
            {
                "id": e.id, "kind": e.kind, "run_id": e.run_id,
                "src": e.src, "dst": e.dst,
                "quote": e.quote, "evidence_position": e.evidence_position,
            }
            for e in self.edges.values()
        ]
        claim_findings = [
            {
                "claim_id": f.claim_id, "run_id": f.run_id, "step_id": f.step_id,
                "status": f.status, "edge_ids": list(f.edge_ids),
                "evidence_empty": f.evidence_empty,
            }
            for f in self.claim_findings
        ]
        constraints = [
            {
                "id": c.id,
                "subject": c.subject,
                "description": c.description,
                "normalized": dict(c.normalized) if c.normalized else None,
            }
            for c in self.constraints.values()
        ]
        constraint_findings = [
            {
                "constraint_id": f.constraint_id,
                "candidate": f.candidate,
                "status": f.status,
                "evidence_step_ids": list(f.evidence_step_ids),
                "commit_step_id": f.commit_step_id,
                "first_assertion_step_id": f.first_assertion_step_id,
                "confidence": f.confidence,
                "confidence_source": f.confidence_source,
                "reason": f.reason,
            }
            for f in self.constraint_findings
        ]
        data = {
            "stats": {
                "steps": len(self.steps),
                "symbols": len(self.symbols),
                "references": len(self.references),
                "relations": len(self.relations),
                "dependencies": len(self.dependencies),
                "claims": len(self.claims),
                "edges": len(self.edges),
                "constraints": len(self.constraints),
                "constraint_findings": len(self.constraint_findings),
                "indexed_message_count": self.indexed_message_count,
            },
            "steps": steps,
            "symbols": symbols,
            "references": refs,
            "relations": relations,
            "dependencies": dependencies,
            "claims": claims,
            "edges": edges,
            "claim_findings": claim_findings,
            "constraints": constraints,
            "constraint_findings": constraint_findings,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryIndex:
        """Load an index from a JSON file written by :meth:`dump`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        index.indexed_message_count = data.get("stats", {}).get("indexed_message_count", 0)

        from loguru import logger as _log

        for s in data.get("steps", []):
            content = str(s.get("content", ""))
            raw_spans = s.get("obs_regions", s.get("obs_spans"))
            if isinstance(raw_spans, list):
                obs_regions = tuple(
                    (int(sp[0]), int(sp[1]))
                    for sp in raw_spans
                    if isinstance(sp, list) and len(sp) == 2
                )
            elif s.get("provenance") == "observation":
                # legacy single-label format
                obs_regions = ((0, len(content)),)
            elif s.get("provenance") == "mixed" and isinstance(s.get("obs_offset"), int):
                obs_regions = ((int(s["obs_offset"]), len(content)),)
            else:
                obs_regions = ()
            step = Step(
                run_id=str(s.get("run_id", "")),
                step_id=str(s["step_id"]),
                index=int(s.get("index", 0)),
                role=str(s.get("role", StepRole.USER)),
                content=content,
                tool_name=s.get("tool_name") if isinstance(s.get("tool_name"), str) else None,
                timestamp=s.get("timestamp") if isinstance(s.get("timestamp"), int | float) else None,
                metadata=s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {},
                obs_regions=obs_regions,
            )
            index.add_step(step)

        for s in data.get("symbols", data.get("entities", [])):
            metadata = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}
            namespace = str(s.get("namespace") or metadata.get("namespace") or "")
            if namespace:
                metadata = {**metadata, "namespace": namespace}
            aliases = set(s.get("aliases", []))
            default_id = (
                stable_id("sym", namespace, normalize_name(str(s["name"])))
                if namespace
                else stable_id("sym", normalize_name(str(s["name"])))
            )
            symbol_id = str(s.get("id") or default_id)
            raw_ec = s.get("entity_class")
            entity_class: EntityClass = raw_ec if raw_ec in _ENTITY_CLASS_VALUES else "identifier"
            if raw_ec and raw_ec not in _ENTITY_CLASS_VALUES:
                _log.warning(f"load: symbol {s.get('name')!r} entity_class {raw_ec!r} invalid, defaulting to 'identifier'")
            symbol = Symbol(
                id=symbol_id,
                canonical_name=str(s["name"]).strip(),
                kind=str(s.get("kind", "unknown")),
                aliases=aliases,
                summary=s.get("summary") if isinstance(s.get("summary"), str) else None,
                entity_class=entity_class,
                metadata=metadata,
            )
            symbol.definition_ref_id = s.get("definition_ref_id", s.get("definition_mention_id"))
            index.symbols[symbol.id] = symbol
            index._index_symbol_name(symbol.id, symbol.canonical_name, namespace)
            for alias in symbol.aliases:
                index._index_symbol_name(symbol.id, alias, namespace)

        for r in data.get("references", data.get("mentions", [])):
            sym_id = r.get("symbol_id", r.get("entity_id", ""))
            sym = index.symbols.get(sym_id)
            if not sym:
                _log.warning(f"load: reference {r['id']} -> missing symbol {sym_id}, skipped")
                continue
            run_id = str(r.get("run_id", ""))
            step_id = str(r["step_id"])
            ref_step = index.steps.get((run_id, step_id))
            if not ref_step:
                ref_step = Step(
                    run_id=run_id, step_id=step_id, index=0,
                    role=StepRole.USER, content="",
                )
                index.add_step(ref_step)
            text = str(r.get("text", ""))
            start = int(r.get("start", r.get("offset", 0)))
            end = int(r.get("end", start + len(text)))
            ref_id = str(r.get("id") or stable_id("ref", run_id, step_id, start, end, sym.id))
            if ref_id in index.references:
                continue
            ref_kind = str(r.get("kind", r.get("mention_type", "unknown")))
            raw_grounded = r.get("grounded")
            grounded = bool(raw_grounded) if isinstance(raw_grounded, bool) else grounded_from_kind(ref_kind)
            structured = bool(r.get("structured", drives_defuse(sym.entity_class)))
            grounds_ref_id = r.get("grounds_ref_id") if isinstance(r.get("grounds_ref_id"), str) else None
            raw_form = r.get("form")
            form: RefForm = raw_form if raw_form in _REF_FORM_VALUES else "direct"
            ref = Reference(
                id=ref_id,
                symbol_id=sym.id,
                run_id=run_id,
                step_id=step_id,
                location=Location(run_id, step_id, start, end),
                text=text,
                role=str(r.get("role", ref_step.role)),
                kind=ref_kind,
                confidence=float(r.get("confidence", 1.0)),
                grounded=grounded,
                grounds_ref_id=grounds_ref_id,
                structured=structured,
                form=form,
                value=r.get("value") if isinstance(r.get("value"), str) else None,
                resolved_from=r.get("resolved_from") if isinstance(r.get("resolved_from"), str) else None,
                metadata=r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {},
            )
            index.references[ref.id] = ref
            index._ref_ids_by_symbol[sym.id].append(ref.id)
            index._ref_ids_by_step[(run_id, step_id)].append(ref.id)

        for r in data.get("relations", []):
            from_s = index.symbols.get(r["from"])
            to_s = index.symbols.get(r["to"])
            if not from_s or not to_s:
                missing = [s for s, v in [("from", from_s), ("to", to_s)] if not v]
                _log.warning(f"load: relation {r['id']} missing {missing} symbol(s), skipped")
                continue
            run_id = str(r.get("run_id", ""))
            step_id = str(r["step_id"])
            rel_step = index.steps.get((run_id, step_id))
            if not rel_step:
                rel_step = Step(
                    run_id=run_id, step_id=step_id, index=0,
                    role=StepRole.USER, content="",
                )
                index.add_step(rel_step)
            rel_id = str(r.get("id") or stable_id("rel", from_s.id, to_s.id, r["type"]))
            if rel_id in index.relations:
                continue
            relation = Relation(
                id=rel_id,
                from_symbol_id=from_s.id,
                to_symbol_id=to_s.id,
                type=str(r["type"]),
                run_id=run_id,
                step_id=step_id,
                weight=float(r.get("weight", 1.0)),
                confidence=float(r.get("confidence", 1.0)),
                metadata=r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {},
            )
            index.relations[relation.id] = relation
            index._relation_ids_by_symbol[from_s.id].append(relation.id)
            index._relation_ids_by_symbol[to_s.id].append(relation.id)

        for d in data.get("dependencies", []):
            sym_id = str(d.get("symbol_id", ""))
            if sym_id not in index.symbols:
                _log.warning(f"load: dependency {d.get('id')} -> missing symbol {sym_id}, skipped")
                continue
            raw_risk = d.get("risk")
            risk: Risk = raw_risk if raw_risk in _RISK_VALUES else "grounded"
            if raw_risk and raw_risk not in _RISK_VALUES:
                _log.warning(f"load: dependency {d.get('id')} risk {raw_risk!r} invalid, defaulting to 'grounded'")
            run_id = str(d.get("run_id", ""))
            def_step_id = str(d.get("def_step_id", ""))
            use_step_id = str(d.get("use_step_id", ""))
            dep_id = str(
                d.get("id") or stable_id("dep", run_id, def_step_id, use_step_id, sym_id)
            )
            if dep_id in index.dependencies:
                continue
            dep = Dependency(
                id=dep_id,
                symbol_id=sym_id,
                run_id=run_id,
                def_step_id=def_step_id,
                def_ref_id=str(d.get("def_ref_id", "")),
                def_version=int(d.get("def_version", 0)),
                use_step_id=use_step_id,
                use_ref_id=str(d.get("use_ref_id", "")),
                risk=risk,
                grounded_by_step_id=(
                    d.get("grounded_by_step_id")
                    if isinstance(d.get("grounded_by_step_id"), str) else None
                ),
                def_value=d.get("def_value") if isinstance(d.get("def_value"), str) else None,
                use_value=d.get("use_value") if isinstance(d.get("use_value"), str) else None,
                confidence=float(d.get("confidence", 1.0)),
                metadata=d.get("metadata", {}) if isinstance(d.get("metadata"), dict) else {},
            )
            index.dependencies[dep.id] = dep
            index._dep_ids_by_symbol[sym_id].append(dep.id)

        for c in data.get("claims", []):
            cid = str(c.get("id", ""))
            text = str(c.get("text", ""))
            if not cid or not text:
                continue
            index.claims[cid] = Claim(
                id=cid,
                run_id=str(c.get("run_id", "")),
                step_id=str(c.get("step_id", "")),
                text=text,
                role=str(c.get("role", "")),
            )

        for e in data.get("edges", []):
            eid = str(e.get("id", ""))
            if not eid:
                continue
            index.edges[eid] = Edge(
                id=eid,
                kind=str(e.get("kind", "")),
                run_id=str(e.get("run_id", "")),
                src=str(e.get("src", "")),
                dst=str(e.get("dst", "")),
                quote=str(e.get("quote", "")),
                evidence_position=str(e.get("evidence_position", "")),
            )

        for cf in data.get("claim_findings", []):
            index.claim_findings.append(ClaimFinding(
                claim_id=str(cf.get("claim_id", "")),
                run_id=str(cf.get("run_id", "")),
                step_id=str(cf.get("step_id", "")),
                status=str(cf.get("status", "unknown")),
                edge_ids=tuple(str(e) for e in cf.get("edge_ids", [])),
                evidence_empty=bool(cf.get("evidence_empty", cf.get("universe_empty", False))),
            ))

        for c in data.get("constraints", []):
            cid = str(c.get("id", ""))
            if not cid:
                continue
            normalized = c.get("normalized")
            index.constraints[cid] = Constraint(
                id=cid,
                subject=str(c.get("subject", "answer")),
                description=str(c.get("description", "")),
                normalized=normalized if isinstance(normalized, dict) else None,
            )

        for f in data.get("constraint_findings", []):
            raw_status = f.get("status")
            status: FindingStatus = (
                raw_status if raw_status in _FINDING_STATUS_VALUES else "unknown"
            )
            if raw_status and raw_status not in _FINDING_STATUS_VALUES:
                _log.warning(
                    f"load: constraint finding status {raw_status!r} invalid, defaulting to 'unknown'"
                )
            index.constraint_findings.append(ConstraintFinding(
                constraint_id=str(f.get("constraint_id", "")),
                candidate=str(f.get("candidate", "")),
                status=status,
                evidence_step_ids=tuple(
                    str(s) for s in f.get("evidence_step_ids", []) if isinstance(s, str | int)
                ),
                commit_step_id=(
                    f.get("commit_step_id")
                    if isinstance(f.get("commit_step_id"), str) else None
                ),
                first_assertion_step_id=(
                    f.get("first_assertion_step_id")
                    if isinstance(f.get("first_assertion_step_id"), str) else None
                ),
                confidence=float(f.get("confidence", 1.0)),
                confidence_source=str(f.get("confidence_source", "")),
                reason=str(f.get("reason", "")),
            ))

        return index

    # ---- read path ----

    def search(
        self,
        query: str,
        *,
        kinds: set[str] | None = None,
        limit: int = 20,
        include_references: bool = True,
        include_related: bool = False,
    ) -> list[SearchResult]:
        norm_query = normalize_name(query)
        if not norm_query:
            return []

        scored: dict[str, tuple[float, set[str]]] = {}

        for symbol_id, symbol in self.symbols.items():
            if kinds and symbol.kind not in kinds:
                continue

            score = 0.0
            matched: set[str] = set()
            norm_names = {normalize_name(n) for n in symbol.all_names}

            if norm_query in norm_names:
                score = max(score, 1.0)
                matched.add("name_exact")
            for nn in norm_names:
                if norm_query in nn or nn in norm_query:
                    score = max(score, 0.75)
                    matched.add("name_partial")
            if symbol.summary and norm_query in normalize_name(symbol.summary):
                score = max(score, 0.5)
                matched.add("summary")
            if score > 0:
                scored[symbol_id] = (score, matched)

        ranked = sorted(
            scored.items(),
            key=lambda item: (-item[1][0], self.symbols[item[0]].canonical_name),
        )

        results: list[SearchResult] = []
        for symbol_id, (score, matched) in ranked[:limit]:
            refs: tuple[Reference, ...] = ()
            related: tuple[RelatedSymbol, ...] = ()
            if include_references:
                refs = tuple(self.get_references(symbol_id)[:5])
            if include_related:
                related = tuple(self.get_related(symbol_id, limit=5))
            results.append(SearchResult(
                symbol=self.symbols[symbol_id],
                score=score,
                matched_fields=tuple(sorted(matched)),
                references=refs,
                related=related,
            ))
        return results

    def get_symbol(self, symbol_id: str) -> Symbol:
        try:
            return self.symbols[symbol_id]
        except KeyError as exc:
            raise KeyError(f"Symbol not found: {symbol_id}") from exc

    def get_references(self, symbol_id: str) -> list[Reference]:
        self.get_symbol(symbol_id)
        ref_ids = self._ref_ids_by_symbol.get(symbol_id, [])
        refs = [self.references[rid] for rid in ref_ids]
        return sorted(refs, key=self._ref_sort_key)

    def get_definition(self, symbol_id: str) -> Reference | None:
        symbol = self.get_symbol(symbol_id)
        if not symbol.definition_ref_id:
            return None
        return self.references.get(symbol.definition_ref_id)

    def get_related(
        self,
        symbol_id: str,
        *,
        limit: int = 20,
        max_depth: int = 1,
        relation_types: set[str] | None = None,
    ) -> list[RelatedSymbol]:
        self.get_symbol(symbol_id)

        scores: dict[str, float] = defaultdict(float)
        rels_by_other: dict[str, list[Relation]] = defaultdict(list)

        visited = {symbol_id}
        queue: deque[tuple[str, int]] = deque([(symbol_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for rel_id in self._relation_ids_by_symbol.get(current_id, []):
                rel = self.relations[rel_id]
                if relation_types and rel.type not in relation_types:
                    continue
                other_id = (
                    rel.to_symbol_id if rel.from_symbol_id == current_id
                    else rel.from_symbol_id
                )
                if other_id == symbol_id:
                    continue
                decay = 1.0 / (depth + 1)
                scores[other_id] += rel.weight * rel.confidence * decay
                rels_by_other[other_id].append(rel)
                if other_id not in visited:
                    visited.add(other_id)
                    queue.append((other_id, depth + 1))

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.symbols[item[0]].canonical_name),
        )
        return [
            RelatedSymbol(
                symbol=self.symbols[oid],
                score=sc,
                relations=tuple(rels_by_other[oid]),
            )
            for oid, sc in ranked[:limit]
        ]

    def get_timeline(self, symbol_id: str) -> list[TimelineItem]:
        refs = self.get_references(symbol_id)
        items: list[TimelineItem] = []
        for ref in refs:
            step = self.steps.get((ref.run_id, ref.step_id))
            if step:
                items.append(TimelineItem(step=step, reference=ref))
        return sorted(
            items,
            key=lambda it: (it.step.run_id, it.step.index, it.reference.location.start),
        )

    def get_context(
        self,
        symbol_id: str,
        *,
        max_references: int = 20,
        max_related: int = 10,
        window: int = 1,
    ) -> SymbolContext:
        symbol = self.get_symbol(symbol_id)
        refs = tuple(self.get_references(symbol_id)[:max_references])
        definition = self.get_definition(symbol_id)
        related = tuple(self.get_related(symbol_id, limit=max_related))
        timeline = tuple(self.get_timeline(symbol_id))

        steps_by_run: dict[str, list[Step]] = defaultdict(list)
        for (rid, _), s in self.steps.items():
            steps_by_run[rid].append(s)
        for v in steps_by_run.values():
            v.sort(key=lambda s: s.index)

        snippets: list[ContextSnippet] = []
        for ref in refs:
            step = self.steps.get((ref.run_id, ref.step_id))
            if not step:
                continue
            same_run = steps_by_run.get(ref.run_id, [])
            before = tuple(
                s for s in same_run
                if step.index - window <= s.index < step.index
            )
            after = tuple(
                s for s in same_run
                if step.index < s.index <= step.index + window
            )
            snippets.append(ContextSnippet(
                focus_step=step, focus_ref=ref,
                before=before, after=after,
            ))

        return SymbolContext(
            symbol=symbol, definition=definition,
            references=refs, related=related,
            timeline=timeline, snippets=tuple(snippets),
        )

    def reference_at(self, run_id: str, step_id: str, offset: int) -> Reference | None:
        ref_ids = self._ref_ids_by_step.get((run_id, step_id), [])
        candidates = [
            self.references[rid]
            for rid in ref_ids
            if self.references[rid].location.contains(offset)
        ]
        if not candidates:
            return None
        return sorted(
            candidates, key=lambda r: (r.location.end - r.location.start, -r.confidence),
        )[0]

    def _ref_sort_key(self, ref: Reference) -> tuple[str, int, int]:
        return (ref.run_id, self._ref_step_index(ref), ref.location.start)
