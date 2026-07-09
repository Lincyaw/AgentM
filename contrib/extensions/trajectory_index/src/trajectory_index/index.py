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
from difflib import SequenceMatcher
from hashlib import sha1
from pathlib import Path
from typing import Literal

type MetadataValue = str | int | float | bool | None
type NameKey = tuple[str, str]

# --- Grounding / def-use analysis (see SCHEMA-readwrite.md) -----------------
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
    SCHEMA-readwrite.md for the build rule.
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
    not a rule — this only proposes the pair with enough context for it (SCHEMA §7)."""

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
        entity_class: EntityClass = "identifier",
    ) -> Symbol:
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
                symbol.entity_class = entity_class
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
            entity_class=entity_class,
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
    ) -> Reference:
        if end is None:
            end = start + len(text)
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

    # ---- alias resolution: deterministic candidates + merge mechanism (SCHEMA §7)
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

    def alias_candidates(self, min_ratio: float = 0.82) -> list[AliasCandidate]:
        """Deterministically block structured-symbol pairs that MIGHT be one entity.

        Cheap lexical signals only — a proper substring relation, or normalized-name
        similarity ≥ ``min_ratio``. This proposes pairs; whether to merge is a
        name-resolution model judgment (Pass 2) fed by :meth:`apply_alias_merges` (SCHEMA §7). No
        scenario-specific rules, no model here.
        """
        # Normalized name + context snippets computed once per symbol (reused across
        # every pair the symbol appears in), not per emitted candidate.
        items = [
            (sid, sym, normalize_name(sym.canonical_name), self._ref_snippets(sid))
            for sid, sym in self.symbols.items()
            if drives_defuse(sym.entity_class)
        ]
        out: list[AliasCandidate] = []
        for i in range(len(items)):
            aid, asym, anorm, asnip = items[i]
            for j in range(i + 1, len(items)):
                bid, bsym, bnorm, bsnip = items[j]
                if anorm == bnorm:
                    continue  # exact-normalized dups already share a symbol via upsert
                substring = (
                    (anorm in bnorm or bnorm in anorm)
                    and min(len(anorm), len(bnorm)) >= _MIN_BLOCK_SUBSTR
                )
                ratio = SequenceMatcher(None, anorm, bnorm).ratio()
                if not substring and ratio < min_ratio:
                    continue
                out.append(AliasCandidate(
                    symbol_a_id=aid, symbol_b_id=bid,
                    name_a=asym.canonical_name, name_b=bsym.canonical_name,
                    kind_a=asym.kind, kind_b=bsym.kind,
                    signal="substring" if substring else "similar",
                    score=round(ratio, 3),
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

    # ---- def-use / grounding layer (Pass 3: dataflow, deterministic) ----

    def _ref_step_index(self, ref: Reference) -> int:
        step = self.steps.get((ref.run_id, ref.step_id))
        return step.index if step else _MISSING_STEP_INDEX

    def build_dependencies(self) -> None:
        """Build the def-use layer over structured entities (Pass 3: dataflow).

        Idempotent: clears and rebuilds every edge. Deterministic (no model),
        global traversal. See SCHEMA-readwrite.md for the def/use classification,
        reaching-def selection with forward grounding propagation, and the
        grounded/premature/ungrounded risk derivation.

        Name resolution (Pass 2) is a separate, upstream step: run apply_alias_merges()
        with the model-decided groups from alias_candidates() *before* this, if
        desired. This method does no merging on its own.
        """
        self.dependencies = {}
        self._dep_ids_by_symbol = defaultdict(list)

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
        step (SCHEMA §5: ``def_step < use_step``). Defs in the use's own step do not
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

        for i, ref in enumerate(refs):
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

            is_def = ref.kind in _PRODUCING_KINDS or i == 0
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

            dep_id = stable_id("dep", run_id, reaching.step_id, ref.step_id, symbol_id)
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
            }
            if s.summary:
                entry["summary"] = s.summary
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

    def stats(self, run_id: str = "") -> IndexStats:
        return IndexStats(
            run_id=run_id,
            step_count=len(self.steps),
            symbol_count=len(self.symbols),
            reference_count=len(self.references),
            relation_count=len(self.relations),
            dependency_count=len(self.dependencies),
        )

    # ---- integrity ----

    def validate(self) -> list[str]:
        """Check referential integrity. Returns a list of error descriptions."""
        errors: list[str] = []
        for ref in self.references.values():
            if ref.symbol_id not in self.symbols:
                errors.append(f"reference {ref.id} points to missing symbol {ref.symbol_id}")
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
        data = {
            "stats": {
                "steps": len(self.steps),
                "symbols": len(self.symbols),
                "references": len(self.references),
                "relations": len(self.relations),
                "dependencies": len(self.dependencies),
                "indexed_message_count": self.indexed_message_count,
            },
            "steps": steps,
            "symbols": symbols,
            "references": refs,
            "relations": relations,
            "dependencies": dependencies,
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

        for s in data.get("steps", []):
            step = Step(
                run_id=str(s.get("run_id", "")),
                step_id=str(s["step_id"]),
                index=int(s.get("index", 0)),
                role=str(s.get("role", StepRole.USER)),
                content=str(s.get("content", "")),
                tool_name=s.get("tool_name") if isinstance(s.get("tool_name"), str) else None,
                timestamp=s.get("timestamp") if isinstance(s.get("timestamp"), int | float) else None,
                metadata=s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {},
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

        from loguru import logger as _log

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
