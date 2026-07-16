"""Read-only query facade for the trajectory index."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .models import (
    ContextSnippet,
    Reference,
    RelatedSymbol,
    Relation,
    SearchResult,
    Step,
    Symbol,
    SymbolContext,
    TimelineItem,
    normalize_name,
)


class TrajectoryQueryMixin:
    """Search and projection operations over an initialized index."""

    symbols: dict[str, Symbol]
    references: dict[str, Reference]
    relations: dict[str, Relation]
    steps: dict[tuple[str, str], Step]
    _ref_ids_by_symbol: dict[str, list[str]]
    _ref_ids_by_step: dict[tuple[str, str], list[str]]
    _relation_ids_by_symbol: dict[str, list[str]]
    _ref_step_index: Any

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
            norm_names = {normalize_name(name) for name in symbol.all_names}
            if norm_query in norm_names:
                score = max(score, 1.0)
                matched.add("name_exact")
            for norm_name in norm_names:
                if norm_query in norm_name or norm_name in norm_query:
                    score = max(score, 0.75)
                    matched.add("name_partial")
            if symbol.summary and norm_query in normalize_name(symbol.summary):
                score = max(score, 0.5)
                matched.add("summary")
            if score > 0:
                scored[symbol_id] = (score, matched)

        ranked = sorted(
            scored.items(),
            key=lambda item: (
                -item[1][0],
                self.symbols[item[0]].canonical_name,
            ),
        )
        results: list[SearchResult] = []
        for symbol_id, (score, matched) in ranked[:limit]:
            references: tuple[Reference, ...] = ()
            related: tuple[RelatedSymbol, ...] = ()
            if include_references:
                references = tuple(self.get_references(symbol_id)[:5])
            if include_related:
                related = tuple(self.get_related(symbol_id, limit=5))
            results.append(
                SearchResult(
                    symbol=self.symbols[symbol_id],
                    score=score,
                    matched_fields=tuple(sorted(matched)),
                    references=references,
                    related=related,
                )
            )
        return results

    def get_symbol(self, symbol_id: str) -> Symbol:
        try:
            return self.symbols[symbol_id]
        except KeyError as exc:
            raise KeyError(f"Symbol not found: {symbol_id}") from exc

    def get_references(self, symbol_id: str) -> list[Reference]:
        self.get_symbol(symbol_id)
        reference_ids = self._ref_ids_by_symbol.get(symbol_id, [])
        references = [self.references[ref_id] for ref_id in reference_ids]
        return sorted(references, key=self._ref_sort_key)

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
        relations_by_other: dict[str, list[Relation]] = defaultdict(list)
        visited = {symbol_id}
        queue: deque[tuple[str, int]] = deque([(symbol_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for relation_id in self._relation_ids_by_symbol.get(current_id, []):
                relation = self.relations[relation_id]
                if relation_types and relation.type not in relation_types:
                    continue
                other_id = (
                    relation.to_symbol_id
                    if relation.from_symbol_id == current_id
                    else relation.from_symbol_id
                )
                if other_id == symbol_id:
                    continue
                scores[other_id] += relation.weight / (depth + 1)
                relations_by_other[other_id].append(relation)
                if other_id not in visited:
                    visited.add(other_id)
                    queue.append((other_id, depth + 1))

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                self.symbols[item[0]].canonical_name,
            ),
        )
        return [
            RelatedSymbol(
                symbol=self.symbols[other_id],
                score=score,
                relations=tuple(relations_by_other[other_id]),
            )
            for other_id, score in ranked[:limit]
        ]

    def get_timeline(self, symbol_id: str) -> list[TimelineItem]:
        items: list[TimelineItem] = []
        for reference in self.get_references(symbol_id):
            step = self.steps.get((reference.run_id, reference.step_id))
            if step is not None:
                items.append(TimelineItem(step=step, reference=reference))
        return sorted(
            items,
            key=lambda item: (
                item.step.run_id,
                item.step.index,
                item.reference.location.start,
            ),
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
        references = tuple(self.get_references(symbol_id)[:max_references])
        definition = self.get_definition(symbol_id)
        related = tuple(self.get_related(symbol_id, limit=max_related))
        timeline = tuple(self.get_timeline(symbol_id))

        steps_by_run: dict[str, list[Step]] = defaultdict(list)
        for (run_id, _), step in self.steps.items():
            steps_by_run[run_id].append(step)
        for steps in steps_by_run.values():
            steps.sort(key=lambda step: step.index)

        snippets: list[ContextSnippet] = []
        for reference in references:
            focus_step = self.steps.get((reference.run_id, reference.step_id))
            if focus_step is None:
                continue
            same_run = steps_by_run.get(reference.run_id, [])
            before = tuple(
                item
                for item in same_run
                if focus_step.index - window <= item.index < focus_step.index
            )
            after = tuple(
                item
                for item in same_run
                if focus_step.index < item.index <= focus_step.index + window
            )
            snippets.append(
                ContextSnippet(
                    focus_step=focus_step,
                    focus_ref=reference,
                    before=before,
                    after=after,
                )
            )

        return SymbolContext(
            symbol=symbol,
            definition=definition,
            references=references,
            related=related,
            timeline=timeline,
            snippets=tuple(snippets),
        )

    def reference_at(
        self,
        run_id: str,
        step_id: str,
        offset: int,
    ) -> Reference | None:
        reference_ids = self._ref_ids_by_step.get((run_id, step_id), [])
        candidates = [
            self.references[reference_id]
            for reference_id in reference_ids
            if self.references[reference_id].location.contains(offset)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda reference: (
                reference.location.end - reference.location.start
            ),
        )

    def _ref_sort_key(self, reference: Reference) -> tuple[str, int, int]:
        return (
            reference.run_id,
            self._ref_step_index(reference),
            reference.location.start,
        )


__all__ = ["TrajectoryQueryMixin"]
