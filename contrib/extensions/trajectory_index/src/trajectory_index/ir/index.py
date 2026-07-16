"""In-memory trajectory semantic index — storage, write primitives, queries.

LSP-inspired symbol-reference-relation graph over agent trajectory steps.
Storage is in-memory; replace with ClickHouse/SQLite/vector-db later while
keeping the query interface stable.

This module is the facade: it holds the ``TrajectoryIndex`` container and its
primitive write/read operations, and re-exports the IR from ``models.py`` so
``from trajectory_index.ir.index import Symbol`` keeps working. The multi-step
passes live in per-pass packages and are exposed here as thin methods:

    pass1_nodes.populate        Pass 1   markup extraction result → IR
    pass2_edges.identity        Pass 2a  alias/coref blocking + merge
    pass2_edges.claims          Pass 2b  claim ↔ observation edges
    pass3_folds.grounding       Pass 3   def-use dataflow + warnings + value fidelity
    pass3_folds.claim_status    Pass 3   claim-status fold
    pass3_folds.constraints     Pass 3   constraint satisfaction
    ir.persistence                       dump / load / validate

See ``designs/`` and SCHEMA.md for the pass contracts.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..pass1_nodes import populate
from ..pass2_edges import identity
from ..pass3_folds import grounding
from ..pass3_folds.grounding import (
    _PRODUCING_KINDS,
    _provenance_kind,
    drives_defuse,
    grounded_from_kind,
)
from . import persistence
from .models import (
    _ENTITY_CLASS_VALUES,
    _FINDING_STATUS_VALUES,
    _REF_FORM_VALUES,
    _RISK_VALUES,
    AliasCandidate,
    Claim,
    ClaimFinding,
    Constraint,
    ConstraintFinding,
    ContextSnippet,
    Dependency,
    Edge,
    EntityClass,
    FindingStatus,
    IndexStats,
    Location,
    MetadataValue,
    NameKey,
    Reference,
    RefForm,
    RelatedSymbol,
    Relation,
    Risk,
    SearchResult,
    Step,
    StepRole,
    Symbol,
    SymbolContext,
    TimelineItem,
    normalize_name,
    stable_id,
)

# Names re-exported so existing ``from trajectory_index.index import X`` imports
# keep resolving after the split. Keep in sync with the models import above.
__all__ = [
    "_ENTITY_CLASS_VALUES",
    "_FINDING_STATUS_VALUES",
    "_REF_FORM_VALUES",
    "_RISK_VALUES",
    "AliasCandidate",
    "Claim",
    "ClaimFinding",
    "Constraint",
    "ConstraintFinding",
    "ContextSnippet",
    "Dependency",
    "Edge",
    "EntityClass",
    "FindingStatus",
    "IndexStats",
    "Location",
    "MetadataValue",
    "NameKey",
    "RefForm",
    "Reference",
    "RelatedSymbol",
    "Relation",
    "Risk",
    "SearchResult",
    "Step",
    "StepRole",
    "Symbol",
    "SymbolContext",
    "TimelineItem",
    "TrajectoryIndex",
    "drives_defuse",
    "grounded_from_kind",
    "normalize_name",
    "stable_id",
]

_MISSING_STEP_INDEX = 10**12  # sort sentinel for a reference whose step is absent


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------


class TrajectoryIndex:
    """In-memory symbol-reference-relation index over trajectory steps."""

    DEFINITION_PREFERRED_KINDS: frozenset[str] = _PRODUCING_KINDS

    # Grounding warning record (kind/symbol/detail/step_ids). Defined in
    # grounding.py; aliased here so ``TrajectoryIndex.Warning`` stays valid.
    Warning = grounding.Warning

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
        )
        self.relations[relation_id] = relation
        self._relation_ids_by_symbol[from_symbol.id].append(relation_id)
        self._relation_ids_by_symbol[to_symbol.id].append(relation_id)
        return relation

    # ---- Pass 2a: alias resolution (pass2_edges.identity) ----

    def alias_candidates(self, min_jaccard: float = 0.5) -> list[AliasCandidate]:
        """Deterministically block structured-symbol pairs that MIGHT be one entity.

        The merge DECISION is a name-resolution model judgment (Pass 2), injected
        via :meth:`apply_alias_merges`; this only proposes the pairs.
        """
        return identity.alias_candidates(self, min_jaccard)

    def apply_alias_merges(self, groups: list[list[str]]) -> None:
        """Fold each decided group of symbol ids into one canonical symbol."""
        identity.apply_alias_merges(self, groups)

    # ---- Pass 3: def-use / grounding (grounding.py) ----

    def _ref_step_index(self, ref: Reference) -> int:
        step = self.steps.get((ref.run_id, ref.step_id))
        return step.index if step else _MISSING_STEP_INDEX

    def build_dependencies(self) -> None:
        """Build the def-use layer over structured entities (Pass 3: dataflow)."""
        grounding.build_dependencies(self)

    def get_dependencies(self, symbol_id: str = "") -> list[Dependency]:
        """All dependency edges, or those for one symbol, in run/step order."""
        if symbol_id:
            deps = [self.dependencies[d] for d in self._dep_ids_by_symbol.get(symbol_id, [])]
        else:
            deps = list(self.dependencies.values())
        return sorted(deps, key=lambda d: (d.run_id, d.use_step_id, d.symbol_id))

    def warnings(self) -> list[grounding.Warning]:
        """Grounding warnings from the index structure (pure code, no LLM)."""
        return grounding.compute_warnings(self)

    def warning_summary(self) -> dict[str, int]:
        """Count warnings by kind."""
        return grounding.warning_summary(self)

    # ---- Pass 1: populate from extraction (populate.py) ----

    def populate_from_extraction(
        self,
        result: Any,
        messages: list[Any],
        *,
        run_id: str = "",
        namespace_fn: Any | None = None,
        message_id_start: int = 0,
        diagnostics: Any | None = None,
    ) -> None:
        """Populate index from an extraction result and source messages (Pass 1)."""
        populate.populate_from_extraction(
            self, result, messages,
            run_id=run_id, namespace_fn=namespace_fn,
            message_id_start=message_id_start, diagnostics=diagnostics,
        )

    # ---- registry / lookup ----

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

    def stats(self, run_id: str = "") -> IndexStats:
        return IndexStats(
            run_id=run_id,
            step_count=len(self.steps),
            symbol_count=len(self.symbols),
            reference_count=len(self.references),
            relation_count=len(self.relations),
            dependency_count=len(self.dependencies),
        )

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

    # ---- integrity + persistence (persistence.py) ----

    def validate(self) -> list[str]:
        """Check referential integrity. Returns a list of error descriptions."""
        return persistence.validate(self)

    def dump(self, path: str | Path) -> None:
        """Write the full index to a JSON file. Validates integrity first."""
        persistence.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryIndex:
        """Load an index from a JSON file written by :meth:`dump`."""
        return persistence.load(cls, path)

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
                scores[other_id] += rel.weight * decay
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
            candidates, key=lambda r: r.location.end - r.location.start,
        )[0]

    def _ref_sort_key(self, ref: Reference) -> tuple[str, int, int]:
        return (ref.run_id, self._ref_step_index(ref), ref.location.start)
