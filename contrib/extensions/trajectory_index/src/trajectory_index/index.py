"""In-memory trajectory semantic index.

LSP-inspired entity-mention-relation graph over agent trajectory steps.
Storage is in-memory; replace with ClickHouse/SQLite/vector-db later while
keeping the query interface stable.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha1

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


class EntityKind(StrEnum):
    VARIABLE = "variable"
    OBJECT = "object"
    CONCEPT = "concept"
    TOOL = "tool"
    FILE = "file"
    API = "api"
    STATE_FIELD = "state_field"
    UNKNOWN = "unknown"


class MentionType(StrEnum):
    DEFINE = "define"
    USE = "use"
    READ = "read"
    WRITE = "write"
    MUTATE = "mutate"
    QUESTION = "question"
    ANSWER = "answer"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    UNKNOWN = "unknown"


class RelationType(StrEnum):
    USES = "uses"
    DEFINES = "defines"
    UPDATES = "updates"
    DERIVED_FROM = "derived_from"
    INPUT_TO = "input_to"
    OUTPUT_OF = "output_of"
    MENTIONS = "mentions"
    EXPLAINS = "explains"
    CONTRADICTS = "contradicts"
    CO_MENTIONED = "co_mentioned"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
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
class Entity:
    id: str
    canonical_name: str
    kind: EntityKind = EntityKind.UNKNOWN
    aliases: set[str] = field(default_factory=set)
    summary: str | None = None
    definition_mention_id: str | None = None
    metadata: MutableMapping[str, MetadataValue] = field(default_factory=dict)

    @property
    def all_names(self) -> set[str]:
        return {self.canonical_name, *self.aliases}


@dataclass(frozen=True)
class Mention:
    id: str
    entity_id: str
    run_id: str
    step_id: str
    span: Span
    text: str
    role: StepRole
    mention_type: MentionType = MentionType.UNKNOWN
    confidence: float = 1.0
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    id: str
    from_entity_id: str
    to_entity_id: str
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
    entity_count: int = 0
    mention_count: int = 0
    relation_count: int = 0


@dataclass(frozen=True)
class SearchResult:
    entity: Entity
    score: float
    matched_fields: tuple[str, ...] = ()
    mentions: tuple[Mention, ...] = ()
    related: tuple[RelatedEntity, ...] = ()


@dataclass(frozen=True)
class RelatedEntity:
    entity: Entity
    score: float
    relations: tuple[Relation, ...] = ()


@dataclass(frozen=True)
class TimelineItem:
    step: Step
    mention: Mention


@dataclass(frozen=True)
class ContextSnippet:
    focus_step: Step
    focus_mention: Mention
    before: tuple[Step, ...] = ()
    after: tuple[Step, ...] = ()


@dataclass(frozen=True)
class EntityContext:
    entity: Entity
    definition: Mention | None
    mentions: tuple[Mention, ...]
    related: tuple[RelatedEntity, ...]
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
    """In-memory entity-mention-relation index over trajectory steps."""

    def __init__(self) -> None:
        self.steps: dict[tuple[str, str], Step] = {}

        self.entities: dict[str, Entity] = {}
        self._entity_ids_by_norm: dict[str, set[str]] = defaultdict(set)

        self.mentions: dict[str, Mention] = {}
        self._mention_ids_by_entity: dict[str, list[str]] = defaultdict(list)
        self._mention_ids_by_step: dict[tuple[str, str], list[str]] = defaultdict(list)

        self.relations: dict[str, Relation] = {}
        self._relation_ids_by_entity: dict[str, list[str]] = defaultdict(list)

    # ---- write path ----

    def add_step(self, step: Step) -> None:
        self.steps[(step.run_id, step.step_id)] = step

    def upsert_entity(
        self,
        name: str,
        kind: EntityKind = EntityKind.UNKNOWN,
        summary: str | None = None,
        aliases: Sequence[str] = (),
    ) -> Entity:
        norm = normalize_name(name)
        existing_ids = self._entity_ids_by_norm.get(norm)
        if existing_ids:
            entity_id = sorted(existing_ids)[0]
            entity = self.entities[entity_id]
            if kind != EntityKind.UNKNOWN and entity.kind == EntityKind.UNKNOWN:
                entity.kind = kind
            if summary and not entity.summary:
                entity.summary = summary
            for alias in aliases:
                entity.aliases.add(alias)
                self._entity_ids_by_norm[normalize_name(alias)].add(entity_id)
            return entity

        entity_id = stable_id("ent", norm)
        entity = Entity(
            id=entity_id,
            canonical_name=name.strip(),
            kind=kind,
            summary=summary,
            aliases=set(aliases),
        )
        self.entities[entity_id] = entity
        self._entity_ids_by_norm[norm].add(entity_id)
        for alias in aliases:
            self._entity_ids_by_norm[normalize_name(alias)].add(entity_id)
        return entity

    def add_mention(
        self,
        entity: Entity,
        step: Step,
        text: str,
        mention_type: MentionType = MentionType.UNKNOWN,
        start: int = 0,
        end: int | None = None,
        confidence: float = 1.0,
    ) -> Mention:
        if end is None:
            end = start + len(text)
        span = Span(step.run_id, step.step_id, start, end)
        mention_id = stable_id(
            "men", span.run_id, span.step_id, span.start, span.end, entity.id,
        )
        if mention_id in self.mentions:
            return self.mentions[mention_id]

        mention = Mention(
            id=mention_id,
            entity_id=entity.id,
            run_id=step.run_id,
            step_id=step.step_id,
            span=span,
            text=text,
            role=step.role,
            mention_type=mention_type,
            confidence=confidence,
        )
        self.mentions[mention_id] = mention
        self._mention_ids_by_entity[entity.id].append(mention_id)
        self._mention_ids_by_step[(step.run_id, step.step_id)].append(mention_id)

        preferred = {MentionType.DEFINE, MentionType.WRITE, MentionType.TOOL_OUTPUT}
        if entity.definition_mention_id is None:
            entity.definition_mention_id = mention_id
        elif mention_type in preferred:
            current = self.mentions.get(entity.definition_mention_id)
            if current and current.mention_type not in preferred:
                entity.definition_mention_id = mention_id

        return mention

    def add_relation(
        self,
        from_entity: Entity,
        to_entity: Entity,
        rel_type: RelationType,
        step: Step,
        weight: float = 1.0,
        confidence: float = 1.0,
    ) -> Relation:
        relation_id = stable_id(
            "rel", step.run_id, step.step_id,
            from_entity.id, to_entity.id, rel_type.value,
        )
        if relation_id in self.relations:
            return self.relations[relation_id]

        relation = Relation(
            id=relation_id,
            from_entity_id=from_entity.id,
            to_entity_id=to_entity.id,
            type=rel_type,
            run_id=step.run_id,
            step_id=step.step_id,
            weight=weight,
            confidence=confidence,
        )
        self.relations[relation_id] = relation
        self._relation_ids_by_entity[from_entity.id].append(relation_id)
        self._relation_ids_by_entity[to_entity.id].append(relation_id)
        return relation

    def stats(self, run_id: str = "") -> IndexStats:
        return IndexStats(
            run_id=run_id,
            step_count=len(self.steps),
            entity_count=len(self.entities),
            mention_count=len(self.mentions),
            relation_count=len(self.relations),
        )

    # ---- read path ----

    def search(
        self,
        query: str,
        *,
        kinds: set[EntityKind] | None = None,
        limit: int = 20,
        include_mentions: bool = True,
        include_related: bool = False,
    ) -> list[SearchResult]:
        norm_query = normalize_name(query)
        if not norm_query:
            return []

        scored: dict[str, tuple[float, set[str]]] = {}

        for entity_id, entity in self.entities.items():
            if kinds and entity.kind not in kinds:
                continue

            score = 0.0
            matched: set[str] = set()
            norm_names = {normalize_name(n) for n in entity.all_names}

            if norm_query in norm_names:
                score = max(score, 1.0)
                matched.add("name_exact")
            for nn in norm_names:
                if norm_query in nn or nn in norm_query:
                    score = max(score, 0.75)
                    matched.add("name_partial")
            if entity.summary and norm_query in normalize_name(entity.summary):
                score = max(score, 0.5)
                matched.add("summary")
            if score > 0:
                scored[entity_id] = (score, matched)

        ranked = sorted(
            scored.items(),
            key=lambda item: (-item[1][0], self.entities[item[0]].canonical_name),
        )

        results: list[SearchResult] = []
        for entity_id, (score, matched) in ranked[:limit]:
            mentions: tuple[Mention, ...] = ()
            related: tuple[RelatedEntity, ...] = ()
            if include_mentions:
                mentions = tuple(self.get_references(entity_id)[:5])
            if include_related:
                related = tuple(self.get_related(entity_id, limit=5))
            results.append(SearchResult(
                entity=self.entities[entity_id],
                score=score,
                matched_fields=tuple(sorted(matched)),
                mentions=mentions,
                related=related,
            ))
        return results

    def get_entity(self, entity_id: str) -> Entity:
        try:
            return self.entities[entity_id]
        except KeyError as exc:
            raise KeyError(f"Entity not found: {entity_id}") from exc

    def get_references(self, entity_id: str) -> list[Mention]:
        self.get_entity(entity_id)
        mention_ids = self._mention_ids_by_entity.get(entity_id, [])
        mentions = [self.mentions[mid] for mid in mention_ids]
        return sorted(mentions, key=self._mention_sort_key)

    def get_definition(self, entity_id: str) -> Mention | None:
        entity = self.get_entity(entity_id)
        if not entity.definition_mention_id:
            return None
        return self.mentions.get(entity.definition_mention_id)

    def get_related(
        self,
        entity_id: str,
        *,
        limit: int = 20,
        max_depth: int = 1,
        relation_types: set[RelationType] | None = None,
    ) -> list[RelatedEntity]:
        self.get_entity(entity_id)

        scores: dict[str, float] = defaultdict(float)
        rels_by_other: dict[str, list[Relation]] = defaultdict(list)

        visited = {entity_id}
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for rel_id in self._relation_ids_by_entity.get(current_id, []):
                rel = self.relations[rel_id]
                if relation_types and rel.type not in relation_types:
                    continue
                other_id = (
                    rel.to_entity_id if rel.from_entity_id == current_id
                    else rel.from_entity_id
                )
                if other_id == entity_id:
                    continue
                decay = 1.0 / (depth + 1)
                scores[other_id] += rel.weight * rel.confidence * decay
                rels_by_other[other_id].append(rel)
                if other_id not in visited:
                    visited.add(other_id)
                    queue.append((other_id, depth + 1))

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.entities[item[0]].canonical_name),
        )
        return [
            RelatedEntity(
                entity=self.entities[oid],
                score=sc,
                relations=tuple(rels_by_other[oid]),
            )
            for oid, sc in ranked[:limit]
        ]

    def get_timeline(self, entity_id: str) -> list[TimelineItem]:
        mentions = self.get_references(entity_id)
        items: list[TimelineItem] = []
        for mention in mentions:
            step = self.steps.get((mention.run_id, mention.step_id))
            if step:
                items.append(TimelineItem(step=step, mention=mention))
        return sorted(
            items,
            key=lambda it: (it.step.run_id, it.step.index, it.mention.span.start),
        )

    def get_context(
        self,
        entity_id: str,
        *,
        max_mentions: int = 20,
        max_related: int = 10,
        window: int = 1,
    ) -> EntityContext:
        entity = self.get_entity(entity_id)
        mentions = tuple(self.get_references(entity_id)[:max_mentions])
        definition = self.get_definition(entity_id)
        related = tuple(self.get_related(entity_id, limit=max_related))
        timeline = tuple(self.get_timeline(entity_id))

        steps_by_run: dict[str, list[Step]] = defaultdict(list)
        for (rid, _), s in self.steps.items():
            steps_by_run[rid].append(s)
        for v in steps_by_run.values():
            v.sort(key=lambda s: s.index)

        snippets: list[ContextSnippet] = []
        for mention in mentions:
            step = self.steps.get((mention.run_id, mention.step_id))
            if not step:
                continue
            same_run = steps_by_run.get(mention.run_id, [])
            before = tuple(
                s for s in same_run
                if step.index - window <= s.index < step.index
            )
            after = tuple(
                s for s in same_run
                if step.index < s.index <= step.index + window
            )
            snippets.append(ContextSnippet(
                focus_step=step, focus_mention=mention,
                before=before, after=after,
            ))

        return EntityContext(
            entity=entity, definition=definition,
            mentions=mentions, related=related,
            timeline=timeline, snippets=tuple(snippets),
        )

    def mention_at(self, run_id: str, step_id: str, offset: int) -> Mention | None:
        mention_ids = self._mention_ids_by_step.get((run_id, step_id), [])
        candidates = [
            self.mentions[mid]
            for mid in mention_ids
            if self.mentions[mid].span.contains(offset)
        ]
        if not candidates:
            return None
        return sorted(
            candidates, key=lambda m: (m.span.end - m.span.start, -m.confidence),
        )[0]

    def _mention_sort_key(self, mention: Mention) -> tuple[str, int, int]:
        step = self.steps.get((mention.run_id, mention.step_id))
        step_index = step.index if step else 10**12
        return (mention.run_id, step_index, mention.span.start)
