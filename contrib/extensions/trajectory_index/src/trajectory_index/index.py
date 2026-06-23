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
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha1
from pathlib import Path

type MetadataValue = str | int | float | bool | None

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class SymbolKind(StrEnum):
    VARIABLE = "variable"
    OBJECT = "object"
    SERVICE = "service"
    TOOL = "tool"
    FILE = "file"
    API = "api"
    METRIC = "metric"
    ERROR = "error"
    CONFIG = "config"
    CONCEPT = "concept"
    UNKNOWN = "unknown"


class ReferenceKind(StrEnum):
    DEFINE = "define"
    USE = "use"
    READ = "read"
    WRITE = "write"
    OBSERVE = "observe"
    HYPOTHESIZE = "hypothesize"
    CONFIRM = "confirm"
    REJECT = "reject"
    CONCLUDE = "conclude"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    QUESTION = "question"
    ANSWER = "answer"
    UNKNOWN = "unknown"


class RelationType(StrEnum):
    USES = "uses"
    DEFINES = "defines"
    UPDATES = "updates"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    INPUT_TO = "input_to"
    OUTPUT_OF = "output_of"
    EXPLAINS = "explains"
    CONTRADICTS = "contradicts"
    CORRELATES = "correlates"


# ---------------------------------------------------------------------------
# Vocabulary descriptions — loaded from vocabulary.yaml
# ---------------------------------------------------------------------------

def _load_vocabulary() -> dict[str, dict[str, str]]:
    """Load vocabulary definitions from vocabulary.yaml and validate against enums."""
    import yaml

    vocab_path = Path(__file__).parent / "vocabulary.yaml"
    data: dict[str, dict[str, str]] = yaml.safe_load(vocab_path.read_text(encoding="utf-8"))

    checks: list[tuple[str, type[StrEnum], dict[str, str]]] = [
        ("symbol_kinds", SymbolKind, data.get("symbol_kinds", {})),
        ("reference_kinds", ReferenceKind, data.get("reference_kinds", {})),
        ("relation_types", RelationType, data.get("relation_types", {})),
    ]
    for section, enum_cls, entries in checks:
        enum_values = {e.value for e in enum_cls}
        yaml_keys = set(entries)
        missing_in_yaml = enum_values - yaml_keys
        extra_in_yaml = yaml_keys - enum_values
        if missing_in_yaml:
            raise ValueError(
                f"vocabulary.yaml [{section}] missing keys for enum members: {sorted(missing_in_yaml)}"
            )
        if extra_in_yaml:
            raise ValueError(
                f"vocabulary.yaml [{section}] has extra keys not in enum: {sorted(extra_in_yaml)}"
            )

    return data


_VOCABULARY = _load_vocabulary()

SYMBOL_KIND_DESCRIPTIONS: dict[str, str] = _VOCABULARY["symbol_kinds"]
REFERENCE_KIND_DESCRIPTIONS: dict[str, str] = _VOCABULARY["reference_kinds"]
RELATION_TYPE_DESCRIPTIONS: dict[str, str] = _VOCABULARY["relation_types"]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Location:
    run_id: str
    step_id: str
    start: int
    end: int

    def contains(self, offset: int) -> bool:
        return self.start <= offset < self.end


@dataclass(frozen=True)
class Step:
    run_id: str
    step_id: str
    index: int
    role: StepRole
    content: str
    tool_name: str | None = None
    timestamp: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass
class Symbol:
    id: str
    canonical_name: str
    kind: SymbolKind = SymbolKind.UNKNOWN
    aliases: set[str] = field(default_factory=set)
    summary: str | None = None
    definition_ref_id: str | None = None
    metadata: MutableMapping[str, MetadataValue] = field(default_factory=dict)

    @property
    def all_names(self) -> set[str]:
        return {self.canonical_name, *self.aliases}


@dataclass(frozen=True)
class Reference:
    id: str
    symbol_id: str
    run_id: str
    step_id: str
    location: Location
    text: str
    role: StepRole
    kind: ReferenceKind = ReferenceKind.UNKNOWN
    confidence: float = 1.0
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    id: str
    from_symbol_id: str
    to_symbol_id: str
    type: RelationType
    run_id: str
    step_id: str
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query / result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexStats:
    run_id: str
    step_count: int = 0
    symbol_count: int = 0
    reference_count: int = 0
    relation_count: int = 0


@dataclass(frozen=True)
class SearchResult:
    symbol: Symbol
    score: float
    matched_fields: tuple[str, ...] = ()
    references: tuple[Reference, ...] = ()
    related: tuple[RelatedSymbol, ...] = ()


@dataclass(frozen=True)
class RelatedSymbol:
    symbol: Symbol
    score: float
    relations: tuple[Relation, ...] = ()


@dataclass(frozen=True)
class TimelineItem:
    step: Step
    reference: Reference


@dataclass(frozen=True)
class ContextSnippet:
    focus_step: Step
    focus_ref: Reference
    before: tuple[Step, ...] = ()
    after: tuple[Step, ...] = ()


@dataclass(frozen=True)
class SymbolContext:
    symbol: Symbol
    definition: Reference | None
    references: tuple[Reference, ...]
    related: tuple[RelatedSymbol, ...]
    timeline: tuple[TimelineItem, ...]
    snippets: tuple[ContextSnippet, ...]


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


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------


class TrajectoryIndex:
    """In-memory symbol-reference-relation index over trajectory steps."""

    def __init__(self) -> None:
        self.steps: dict[tuple[str, str], Step] = {}

        self.symbols: dict[str, Symbol] = {}
        self._symbol_ids_by_norm: dict[str, set[str]] = defaultdict(set)

        self.references: dict[str, Reference] = {}
        self._ref_ids_by_symbol: dict[str, list[str]] = defaultdict(list)
        self._ref_ids_by_step: dict[tuple[str, str], list[str]] = defaultdict(list)

        self.relations: dict[str, Relation] = {}
        self._relation_ids_by_symbol: dict[str, list[str]] = defaultdict(list)

        self.indexed_message_count: int = 0

    # ---- write path ----

    def add_step(self, step: Step) -> None:
        self.steps[(step.run_id, step.step_id)] = step

    def upsert_symbol(
        self,
        name: str,
        kind: SymbolKind = SymbolKind.UNKNOWN,
        summary: str | None = None,
        aliases: Sequence[str] = (),
    ) -> Symbol:
        norm = normalize_name(name)
        existing_ids = self._symbol_ids_by_norm.get(norm)
        if existing_ids:
            symbol_id = sorted(existing_ids)[0]
            symbol = self.symbols[symbol_id]
            if kind != SymbolKind.UNKNOWN and symbol.kind == SymbolKind.UNKNOWN:
                symbol.kind = kind
            if summary and not symbol.summary:
                symbol.summary = summary
            for alias in aliases:
                symbol.aliases.add(alias)
                self._symbol_ids_by_norm[normalize_name(alias)].add(symbol_id)
            return symbol

        symbol_id = stable_id("sym", norm)
        symbol = Symbol(
            id=symbol_id,
            canonical_name=name.strip(),
            kind=kind,
            summary=summary,
            aliases=set(aliases),
        )
        self.symbols[symbol_id] = symbol
        self._symbol_ids_by_norm[norm].add(symbol_id)
        for alias in aliases:
            self._symbol_ids_by_norm[normalize_name(alias)].add(symbol_id)
        return symbol

    def add_reference(
        self,
        symbol: Symbol,
        step: Step,
        text: str,
        kind: ReferenceKind = ReferenceKind.UNKNOWN,
        start: int = 0,
        end: int | None = None,
        confidence: float = 1.0,
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
        )
        self.references[ref_id] = ref
        self._ref_ids_by_symbol[symbol.id].append(ref_id)
        self._ref_ids_by_step[(step.run_id, step.step_id)].append(ref_id)

        preferred = {ReferenceKind.DEFINE, ReferenceKind.WRITE, ReferenceKind.TOOL_OUTPUT}
        if symbol.definition_ref_id is None:
            symbol.definition_ref_id = ref_id
        elif kind in preferred:
            current = self.references.get(symbol.definition_ref_id)
            if current and current.kind not in preferred:
                symbol.definition_ref_id = ref_id

        return ref

    def add_relation(
        self,
        from_symbol: Symbol,
        to_symbol: Symbol,
        rel_type: RelationType,
        step: Step,
        weight: float = 1.0,
        confidence: float = 1.0,
    ) -> Relation:
        relation_id = stable_id(
            "rel", from_symbol.id, to_symbol.id, rel_type.value,
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

    def registry_snapshot(self) -> list[dict[str, str | list[str]]]:
        """Compact symbol registry for incremental extraction prompts."""
        out: list[dict[str, str | list[str]]] = []
        for s in self.symbols.values():
            entry: dict[str, str | list[str]] = {
                "name": s.canonical_name,
                "kind": s.kind.value,
            }
            if s.summary:
                entry["summary"] = s.summary
            if s.aliases:
                entry["aliases"] = sorted(s.aliases)
            out.append(entry)
        return out

    def resolve_symbol_by_name(self, name: str) -> Symbol | None:
        """Look up a symbol by name or alias (normalized match)."""
        norm = normalize_name(name)
        ids = self._symbol_ids_by_norm.get(norm)
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
        )

    # ---- persistence ----

    def dump(self, path: str | Path) -> None:
        """Write the full index to a JSON file."""
        symbols = [
            {
                "id": s.id,
                "name": s.canonical_name,
                "kind": s.kind.value,
                "aliases": sorted(s.aliases) if s.aliases else [],
                "summary": s.summary,
                "definition_ref_id": s.definition_ref_id,
            }
            for s in self.symbols.values()
        ]
        refs = [
            {
                "id": r.id,
                "symbol_id": r.symbol_id,
                "step_id": r.step_id,
                "text": r.text,
                "kind": r.kind.value,
                "confidence": r.confidence,
            }
            for r in self.references.values()
        ]
        relations = [
            {
                "id": r.id,
                "from": r.from_symbol_id,
                "to": r.to_symbol_id,
                "type": r.type.value,
                "step_id": r.step_id,
                "confidence": r.confidence,
            }
            for r in self.relations.values()
        ]
        data = {
            "stats": {
                "steps": len(self.steps),
                "symbols": len(self.symbols),
                "references": len(self.references),
                "relations": len(self.relations),
                "indexed_message_count": self.indexed_message_count,
            },
            "symbols": symbols,
            "references": refs,
            "relations": relations,
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

        for s in data.get("symbols", data.get("entities", [])):
            symbol = index.upsert_symbol(
                name=s["name"],
                kind=SymbolKind(s["kind"]),
                summary=s.get("summary"),
                aliases=s.get("aliases", []),
            )
            symbol.definition_ref_id = s.get("definition_ref_id", s.get("definition_mention_id"))

        for r in data.get("references", data.get("mentions", [])):
            sym = index.symbols.get(r.get("symbol_id", r.get("entity_id", "")))
            if not sym:
                continue
            ref_step = index.steps.get(("", r["step_id"]))
            if not ref_step:
                ref_step = Step(
                    run_id="", step_id=r["step_id"], index=0,
                    role=StepRole.USER, content="",
                )
                index.add_step(ref_step)
            index.add_reference(
                symbol=sym, step=ref_step, text=r["text"],
                kind=ReferenceKind(r.get("kind", r.get("mention_type", "unknown"))),
                confidence=r.get("confidence", 1.0),
            )

        for r in data["relations"]:
            from_s = index.symbols.get(r["from"])
            to_s = index.symbols.get(r["to"])
            if not from_s or not to_s:
                continue
            rel_step = index.steps.get(("", r["step_id"]))
            if not rel_step:
                rel_step = Step(
                    run_id="", step_id=r["step_id"], index=0,
                    role=StepRole.USER, content="",
                )
                index.add_step(rel_step)
            index.add_relation(
                from_symbol=from_s, to_symbol=to_s,
                rel_type=RelationType(r["type"]),
                step=rel_step,
                confidence=r.get("confidence", 1.0),
            )

        return index

    # ---- read path ----

    def search(
        self,
        query: str,
        *,
        kinds: set[SymbolKind] | None = None,
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
        relation_types: set[RelationType] | None = None,
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
        step = self.steps.get((ref.run_id, ref.step_id))
        step_index = step.index if step else 10**12
        return (ref.run_id, step_index, ref.location.start)
