# code-health: ignore-file[AM025] -- IFG projections normalize raw JSON evidence into typed graph rows
"""Pure IFG projections from atomic facts into symbols and graph rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from policy_engine.repository_index import RepositoryIndex
from policy_engine.source_parser import (
    SymbolExtractionInput,
    SymbolFact,
    extract_symbols_from_repository_files,
    extract_symbols_from_source_units,
)

from .types import (
    IfgActionFileEdgeRow,
    IfgActionRow,
    IfgActionSymbolEdgeRow,
    IfgExtractionRows,
    IfgFileSymbolEdgeRow,
    IfgGraphEdgeRow,
    IfgNodeRow,
    IfgPathCandidateRow,
    IfgSourceUnitRow,
    IfgSymbolMentionRow,
    IfgSymbolRow,
    IfgSymbolSymbolEdgeRow,
)
from .utils import _aggregate_confidence, _stable_id


def build_ifg_graph(
    *,
    actions: Sequence[IfgActionRow],
    source_units: Sequence[IfgSourceUnitRow],
    file_edges: Sequence[IfgActionFileEdgeRow],
    action_symbol_edges: Sequence[IfgActionSymbolEdgeRow],
    file_symbol_edges: Sequence[IfgFileSymbolEdgeRow],
    symbol_symbol_edges: Sequence[IfgSymbolSymbolEdgeRow],
    symbols: Sequence[IfgSymbolRow],
    extractor_version: str,
) -> IfgExtractionRows:
    """Build the typed IFG node/edge layer from extracted facts."""

    nodes: list[IfgNodeRow] = []
    graph_edges: list[IfgGraphEdgeRow] = []
    action_ids = {action.action_id for action in actions}

    for action in actions:
        nodes.append(_action_node(action))

    source_node_by_id: dict[str, IfgNodeRow] = {}
    for unit in source_units:
        node = _source_unit_node(unit)
        source_node_by_id[unit.source_unit_id] = node
        nodes.append(node)
        if unit.action_id in action_ids:
            graph_edges.append(
                _graph_edge(
                    session_id=unit.session_id,
                    extractor_version=unit.extractor_version,
                    from_node_id=_action_node_id(unit.action_id),
                    to_node_id=node.node_id,
                    relation=unit.relation,
                    turn=unit.turn,
                    event_id=unit.event_id,
                    source=unit.origin,
                    confidence=_unit_confidence(unit),
                    metadata={
                        "source_unit_id": unit.source_unit_id,
                        "kind": unit.kind,
                        "origin": unit.origin,
                        "path": unit.path,
                    },
                    raw_evidence=unit.raw_evidence,
                    stable_parts=("action_source", unit.source_unit_id),
                )
            )

    file_node_by_path: dict[str, IfgNodeRow] = {}
    for path, edges in _group_file_edges(file_edges).items():
        node = _file_node(path, edges, extractor_version=extractor_version)
        file_node_by_path[path] = node
        nodes.append(node)

    symbol_node_by_id: dict[str, IfgNodeRow] = {}
    for symbol in symbols:
        node = _symbol_node(symbol)
        symbol_node_by_id[symbol.symbol_id] = node
        nodes.append(node)

    for edge in file_edges:
        file_node = file_node_by_path.get(edge.path)
        if file_node is None:
            continue
        graph_edges.append(
            _graph_edge(
                session_id=edge.session_id,
                extractor_version=edge.extractor_version,
                from_node_id=_action_node_id(edge.action_id),
                to_node_id=file_node.node_id,
                relation=edge.relation,
                turn=edge.turn,
                event_id=edge.event_id,
                source=edge.source,
                confidence=edge.confidence,
                metadata={"path": edge.path},
                raw_evidence=edge.raw_evidence,
                stable_parts=("action_file", edge.edge_id),
            )
        )

    for unit in source_units:
        if unit.path is None:
            continue
        source_node = source_node_by_id.get(unit.source_unit_id)
        file_node = file_node_by_path.get(unit.path)
        if source_node is None or file_node is None:
            continue
        graph_edges.append(
            _graph_edge(
                session_id=unit.session_id,
                extractor_version=unit.extractor_version,
                from_node_id=source_node.node_id,
                to_node_id=file_node.node_id,
                relation=unit.relation,
                turn=unit.turn,
                event_id=unit.event_id,
                source=unit.origin,
                confidence=_unit_confidence(unit),
                metadata={
                    "source_unit_id": unit.source_unit_id,
                    "path": unit.path,
                    "kind": unit.kind,
                },
                raw_evidence=unit.raw_evidence,
                stable_parts=("source_file", unit.source_unit_id, unit.path),
            )
        )

    for edge in action_symbol_edges:
        symbol_node = symbol_node_by_id.get(edge.symbol_id)
        if symbol_node is None:
            continue
        source_unit_id = _metadata_str(edge.metadata, "source_unit_id")
        source_node = source_node_by_id.get(source_unit_id) if source_unit_id else None
        graph_edges.append(
            _graph_edge(
                session_id=edge.session_id,
                extractor_version=edge.extractor_version,
                from_node_id=(
                    source_node.node_id
                    if source_node is not None
                    else _action_node_id(edge.action_id)
                ),
                to_node_id=symbol_node.node_id,
                relation=edge.relation,
                turn=edge.turn,
                event_id=edge.event_id,
                source=edge.source,
                confidence=edge.confidence,
                metadata=edge.metadata,
                raw_evidence=edge.raw_evidence,
                stable_parts=("action_symbol", edge.edge_id),
            )
        )

    for edge in file_symbol_edges:
        file_node = file_node_by_path.get(edge.path)
        symbol_node = symbol_node_by_id.get(edge.symbol_id)
        if file_node is None or symbol_node is None:
            continue
        graph_edges.append(
            _graph_edge(
                session_id=edge.session_id,
                extractor_version=edge.extractor_version,
                from_node_id=file_node.node_id,
                to_node_id=symbol_node.node_id,
                relation=edge.relation,
                turn=edge.turn,
                event_id=edge.event_id,
                source=edge.source,
                confidence=edge.confidence,
                metadata=edge.metadata,
                raw_evidence=edge.raw_evidence,
                stable_parts=("file_symbol", edge.edge_id),
            )
        )

    for edge in symbol_symbol_edges:
        from_node = symbol_node_by_id.get(edge.from_symbol_id)
        to_node = symbol_node_by_id.get(edge.to_symbol_id)
        if from_node is None or to_node is None:
            continue
        graph_edges.append(
            _graph_edge(
                session_id=edge.session_id,
                extractor_version=edge.extractor_version,
                from_node_id=from_node.node_id,
                to_node_id=to_node.node_id,
                relation=edge.relation,
                turn=edge.turn,
                event_id=edge.event_id,
                source=edge.source,
                confidence=edge.confidence,
                metadata=edge.metadata,
                raw_evidence=edge.raw_evidence,
                stable_parts=("symbol_symbol", edge.edge_id),
            )
        )

    return IfgExtractionRows(
        actions=(),
        file_edges=(),
        source_units=(),
        symbol_mentions=(),
        graph_nodes=tuple(nodes),
        graph_edges=tuple(graph_edges),
    )


def build_ifg_symbols(
    *,
    source_units: Sequence[IfgSourceUnitRow],
    symbol_mentions: Sequence[IfgSymbolMentionRow],
    file_edges: Sequence[IfgActionFileEdgeRow] = (),
    extractor_version: str,
    repository_index: RepositoryIndex | None = None,
) -> IfgExtractionRows:
    """Extract code symbols from source units and resolve text mentions to them."""

    resolved_paths = frozenset(edge.path for edge in file_edges)
    fragment_result = extract_symbols_from_source_units(
        _symbol_inputs_from_source_units(
            source_units,
            resolved_paths=resolved_paths,
        ),
        extractor_version=extractor_version,
    )
    repository_inputs = _repository_symbol_inputs(
        source_units,
        file_edges,
        repository_index=repository_index,
    )
    repository_result = (
        repository_index.extract_symbols(
            repository_inputs,
            extractor_version=extractor_version,
        )
        if repository_index is not None
        else extract_symbols_from_repository_files(
            repository_inputs,
            extractor_version=extractor_version,
        )
    )
    symbol_facts = fragment_result.symbols + repository_result.symbols

    symbols: list[IfgSymbolRow] = []
    action_symbol_edges: list[IfgActionSymbolEdgeRow] = []
    file_symbol_edges: list[IfgFileSymbolEdgeRow] = []
    fact_groups: list[tuple[IfgSymbolRow, list[SymbolFact]]] = []
    for _stable_key, facts in _group_symbol_facts(symbol_facts).items():
        symbol = _symbol_from_facts(facts, extractor_version=extractor_version)
        symbols.append(symbol)
        fact_groups.append((symbol, facts))
        for fact in facts:
            if fact.metadata.get("link_action", True):
                action_symbol_edges.append(_action_symbol_edge_from_fact(fact, symbol))
            file_symbol_edges.append(_file_symbol_edge_from_fact(fact, symbol))

    resolved_edges, unresolved_mentions = _resolve_symbol_mentions(
        symbol_mentions,
        symbols,
        file_edges=file_edges,
    )
    action_symbol_edges.extend(resolved_edges)

    symbol_symbol_edges = _symbol_symbol_edges_from_facts(fact_groups)
    return IfgExtractionRows(
        actions=(),
        file_edges=(),
        source_units=(),
        symbol_mentions=tuple(unresolved_mentions),
        symbols=tuple(symbols),
        action_symbol_edges=tuple(action_symbol_edges),
        file_symbol_edges=tuple(file_symbol_edges),
        symbol_symbol_edges=tuple(symbol_symbol_edges),
        errors=fragment_result.errors + repository_result.errors,
    )


def unique_nodes(nodes: Sequence[IfgNodeRow]) -> tuple[IfgNodeRow, ...]:
    return tuple({node.node_id: node for node in nodes}.values())


def unique_graph_edges(
    edges: Sequence[IfgGraphEdgeRow],
) -> tuple[IfgGraphEdgeRow, ...]:
    return tuple({edge.edge_id: edge for edge in edges}.values())


def unique_actions(actions: Sequence[IfgActionRow]) -> tuple[IfgActionRow, ...]:
    return tuple({action.action_id: action for action in actions}.values())


def unique_file_edges(
    edges: Sequence[IfgActionFileEdgeRow],
) -> tuple[IfgActionFileEdgeRow, ...]:
    return tuple({edge.edge_id: edge for edge in edges}.values())


def unique_source_units(
    units: Sequence[IfgSourceUnitRow],
) -> tuple[IfgSourceUnitRow, ...]:
    return tuple({unit.source_unit_id: unit for unit in units}.values())


def unique_path_candidates(
    candidates: Sequence[IfgPathCandidateRow],
) -> tuple[IfgPathCandidateRow, ...]:
    return tuple(
        {candidate.candidate_id: candidate for candidate in candidates}.values()
    )


def resolve_path_candidates(
    *,
    anchor_edges: Sequence[IfgActionFileEdgeRow],
    candidates: Sequence[IfgPathCandidateRow],
    extractor_version: str,
) -> tuple[IfgActionFileEdgeRow, ...]:
    """Attach bash support evidence only to files anchored by file operations."""

    anchors = unique_file_edges(anchor_edges)
    anchored_paths = {edge.path for edge in anchors}
    resolved: list[IfgActionFileEdgeRow] = list(anchors)
    for candidate in candidates:
        if (
            candidate.normalized_path not in anchored_paths
            or candidate.path_kind == "pattern"
        ):
            continue
        resolution = "trajectory_anchor"
        relation = "supports"
        edge_id = _stable_id(
            "edge",
            candidate.action_id,
            candidate.normalized_path,
            relation,
            candidate.source,
            resolution,
            extractor_version,
        )
        resolved.append(
            IfgActionFileEdgeRow(
                edge_id=edge_id,
                session_id=candidate.session_id,
                action_id=candidate.action_id,
                path=candidate.normalized_path,
                relation=relation,
                turn=candidate.turn,
                event_id=candidate.event_id,
                source=candidate.source,
                confidence=candidate.confidence,
                is_anchor=False,
                extractor_version=extractor_version,
                content_hash=None,
                before_hash=None,
                after_hash=None,
                content_state=None,
                line_range={},
                span={},
                metadata={
                    **dict(candidate.metadata),
                    "anchor": False,
                    "candidate_id": candidate.candidate_id,
                    "source_unit_id": candidate.source_unit_id,
                    "entity_kind": "file",
                    "evidence": candidate.source,
                    "evidence_role": "support",
                    "observed_relation": candidate.relation,
                    "existence": "anchored_by_fileop",
                    "resolution": resolution,
                },
                raw_evidence=candidate.raw_evidence,
            )
        )
    return unique_file_edges(resolved)


def unique_symbol_mentions(
    mentions: Sequence[IfgSymbolMentionRow],
) -> tuple[IfgSymbolMentionRow, ...]:
    return tuple({mention.mention_id: mention for mention in mentions}.values())


def unique_symbols(symbols: Sequence[IfgSymbolRow]) -> tuple[IfgSymbolRow, ...]:
    return tuple({symbol.symbol_id: symbol for symbol in symbols}.values())


def unique_action_symbol_edges(
    edges: Sequence[IfgActionSymbolEdgeRow],
) -> tuple[IfgActionSymbolEdgeRow, ...]:
    return tuple({edge.edge_id: edge for edge in edges}.values())


def unique_file_symbol_edges(
    edges: Sequence[IfgFileSymbolEdgeRow],
) -> tuple[IfgFileSymbolEdgeRow, ...]:
    return tuple({edge.edge_id: edge for edge in edges}.values())


def unique_symbol_symbol_edges(
    edges: Sequence[IfgSymbolSymbolEdgeRow],
) -> tuple[IfgSymbolSymbolEdgeRow, ...]:
    return tuple({edge.edge_id: edge for edge in edges}.values())


def _symbol_inputs_from_source_units(
    source_units: Sequence[IfgSourceUnitRow],
    *,
    resolved_paths: frozenset[str],
) -> tuple[SymbolExtractionInput, ...]:
    return tuple(
        SymbolExtractionInput(
            session_id=unit.session_id,
            extractor_version=unit.extractor_version,
            action_id=unit.action_id,
            source_unit_id=unit.source_unit_id,
            path=unit.path,
            relation=unit.relation,
            turn=unit.turn,
            event_id=unit.event_id,
            tool_name=unit.tool_name,
            content_hash=unit.content_hash,
            unit_hash=unit.unit_hash,
            content_text=unit.content_text,
            metadata={
                "source_unit_id": unit.source_unit_id,
                "source_unit_kind": unit.kind,
                "origin": unit.origin,
                **dict(unit.metadata),
            },
            raw_evidence=unit.raw_evidence,
        )
        for unit in source_units
        if unit.path is not None
        and unit.path in resolved_paths
        and unit.tool_name in {"read", "write", "edit"}
        and unit.content_text
        and bool(unit.metadata.get("parse_symbols", True))
    )


def _repository_symbol_inputs(
    source_units: Sequence[IfgSourceUnitRow],
    file_edges: Sequence[IfgActionFileEdgeRow],
    *,
    repository_index: RepositoryIndex | None = None,
) -> tuple[SymbolExtractionInput, ...]:
    units_by_id = {unit.source_unit_id: unit for unit in source_units}
    units_by_path: dict[str, IfgSourceUnitRow] = {}
    for unit in source_units:
        if unit.path and unit.tool_name in {"read", "write", "edit"}:
            units_by_path.setdefault(unit.path, unit)

    inputs: list[SymbolExtractionInput] = []
    seen_paths: set[str] = set()
    for edge in file_edges:
        if (
            not edge.is_anchor
            or edge.path in seen_paths
            or not _repository_contains_file(
                edge.path,
                repository_index=repository_index,
            )
        ):
            continue
        source_unit_id = edge.metadata.get("source_unit_id")
        unit = (
            units_by_id.get(source_unit_id) if isinstance(source_unit_id, str) else None
        )
        if unit is None:
            unit = units_by_path.get(edge.path)
        if unit is None:
            continue
        seen_paths.add(edge.path)
        inputs.append(
            SymbolExtractionInput(
                session_id=unit.session_id,
                extractor_version=unit.extractor_version,
                action_id=edge.action_id,
                source_unit_id=unit.source_unit_id,
                path=edge.path,
                relation=edge.relation,
                turn=edge.turn,
                event_id=edge.event_id,
                tool_name=unit.tool_name,
                content_hash=unit.content_hash,
                unit_hash=unit.unit_hash,
                content_text=None,
                metadata={
                    "source_unit_id": unit.source_unit_id,
                    "source_unit_kind": unit.kind,
                    "origin": unit.origin,
                    "repository_validation": True,
                    **dict(unit.metadata),
                },
                raw_evidence=unit.raw_evidence,
            )
        )
    return tuple(inputs)


def _repository_contains_file(
    path: str,
    *,
    repository_index: RepositoryIndex | None,
) -> bool:
    if repository_index is not None:
        return repository_index.contains_file(path)
    return Path(path).is_file()


def _resolve_symbol_mentions(
    mentions: Sequence[IfgSymbolMentionRow],
    symbols: Sequence[IfgSymbolRow],
    *,
    file_edges: Sequence[IfgActionFileEdgeRow],
) -> tuple[list[IfgActionSymbolEdgeRow], list[IfgSymbolMentionRow]]:
    symbols_by_name = _symbols_by_name(symbols)
    action_paths = _action_paths(file_edges)
    edges: list[IfgActionSymbolEdgeRow] = []
    unresolved: list[IfgSymbolMentionRow] = []

    for mention in mentions:
        candidates = symbols_by_name.get(mention.symbol_text, ())
        targets = _mention_targets(
            mention,
            candidates,
            context_paths=action_paths.get(mention.action_id, frozenset()),
        )
        if not targets:
            unresolved.append(mention)
            continue
        resolution = (
            "path" if mention.path or action_paths.get(mention.action_id) else "unique"
        )
        if len(targets) > 1:
            resolution = f"{resolution}:ambiguous"
        for symbol in targets:
            edges.append(
                _action_symbol_edge_from_mention(
                    mention,
                    symbol,
                    resolution=resolution,
                )
            )
    return edges, unresolved


def _symbols_by_name(
    symbols: Sequence[IfgSymbolRow],
) -> dict[str, tuple[IfgSymbolRow, ...]]:
    grouped: dict[str, list[IfgSymbolRow]] = {}
    for symbol in symbols:
        if symbol.kind == "mention" or symbol.source == "symbol_mentions":
            continue
        grouped.setdefault(symbol.qualified_name, []).append(symbol)
        short_name = symbol.metadata.get("short_name")
        if (
            isinstance(short_name, str)
            and short_name
            and short_name != symbol.qualified_name
        ):
            grouped.setdefault(short_name, []).append(symbol)
    return {name: tuple(items) for name, items in grouped.items()}


def _mention_targets(
    mention: IfgSymbolMentionRow,
    candidates: Sequence[IfgSymbolRow],
    *,
    context_paths: frozenset[str],
) -> tuple[IfgSymbolRow, ...]:
    if not candidates:
        return ()
    paths = set(context_paths)
    if mention.path:
        paths.add(mention.path)
    if paths:
        same_path = tuple(symbol for symbol in candidates if symbol.path in paths)
        if same_path:
            return same_path[:5]
    if len(candidates) == 1:
        return tuple(candidates)
    defining = tuple(symbol for symbol in candidates if _symbol_defines_code(symbol))
    if len(defining) == 1:
        return defining
    return ()


def _symbol_defines_code(symbol: IfgSymbolRow) -> bool:
    counts = symbol.metadata.get("relation_counts")
    if not isinstance(counts, Mapping):
        return False
    return bool(counts.get("defines") or counts.get("exports"))


def _action_paths(
    file_edges: Sequence[IfgActionFileEdgeRow],
) -> dict[str, frozenset[str]]:
    paths: dict[str, set[str]] = {}
    for edge in file_edges:
        paths.setdefault(edge.action_id, set()).add(edge.path)
    return {action_id: frozenset(values) for action_id, values in paths.items()}


def _group_file_edges(
    edges: Sequence[IfgActionFileEdgeRow],
) -> dict[str, list[IfgActionFileEdgeRow]]:
    by_path: dict[str, list[IfgActionFileEdgeRow]] = {}
    for edge in edges:
        by_path.setdefault(edge.path, []).append(edge)
    return by_path


def _group_symbol_facts(
    facts: Sequence[SymbolFact],
) -> dict[str, list[SymbolFact]]:
    by_key: dict[str, list[SymbolFact]] = {}
    for fact in facts:
        by_key.setdefault(fact.stable_key, []).append(fact)
    return by_key


def _symbol_from_facts(
    facts: Sequence[SymbolFact],
    *,
    extractor_version: str,
) -> IfgSymbolRow:
    first = facts[0]
    source_unit_ids = tuple(dict.fromkeys(fact.source_unit_id for fact in facts))
    sources = tuple(dict.fromkeys(fact.source for fact in facts))
    relations = Counter(fact.file_relation for fact in facts)
    validations = tuple(
        dict.fromkeys(
            str(value) for fact in facts if (value := fact.metadata.get("validation"))
        )
    )
    content_scopes = tuple(
        dict.fromkeys(
            str(value)
            for fact in facts
            if (value := fact.metadata.get("content_scope"))
        )
    )
    return IfgSymbolRow(
        symbol_id=_symbol_id(first.session_id, extractor_version, first.stable_key),
        session_id=first.session_id,
        extractor_version=extractor_version,
        kind=first.kind,
        qualified_name=first.qualified_name,
        path=first.path,
        stable_key=first.stable_key,
        first_seen_turn=min(fact.turn for fact in facts),
        last_seen_turn=max(fact.turn for fact in facts),
        observation_count=len(source_unit_ids),
        source=sources[0] if sources else first.source,
        confidence=_aggregate_confidence(fact.confidence for fact in facts),
        metadata={
            "kind": first.kind,
            "path": first.path,
            "sources": list(sources),
            "relation_counts": dict(relations),
            "fact_count": len(facts),
            **dict(first.metadata),
            "validations": list(validations),
            "validation": (
                "repository_present"
                if "repository_present" in validations
                else "trajectory_observed"
            ),
            "content_scopes": list(content_scopes),
        },
        raw_evidence={
            "source_unit_ids": list(source_unit_ids[:100]),
            "sources": list(sources[:100]),
            "qualified_name": first.qualified_name,
        },
    )


def _action_symbol_edge_from_fact(
    fact: SymbolFact,
    symbol: IfgSymbolRow,
) -> IfgActionSymbolEdgeRow:
    return IfgActionSymbolEdgeRow(
        edge_id=_stable_id(
            "action_symbol_edge",
            fact.action_id,
            symbol.symbol_id,
            fact.action_relation,
            fact.source_unit_id,
            fact.extractor_version,
        ),
        session_id=fact.session_id,
        extractor_version=fact.extractor_version,
        action_id=fact.action_id,
        symbol_id=symbol.symbol_id,
        relation=fact.action_relation,
        turn=fact.turn,
        event_id=fact.event_id,
        source=fact.source,
        confidence=fact.confidence,
        metadata={
            "path": fact.path,
            "kind": fact.kind,
            "qualified_name": fact.qualified_name,
            "source_unit_id": fact.source_unit_id,
            **dict(fact.metadata),
        },
        raw_evidence=fact.raw_evidence,
    )


def _file_symbol_edge_from_fact(
    fact: SymbolFact,
    symbol: IfgSymbolRow,
) -> IfgFileSymbolEdgeRow:
    return IfgFileSymbolEdgeRow(
        edge_id=_stable_id(
            "file_symbol_edge",
            fact.session_id,
            fact.path,
            symbol.symbol_id,
            fact.file_relation,
            fact.extractor_version,
        ),
        session_id=fact.session_id,
        extractor_version=fact.extractor_version,
        path=fact.path,
        symbol_id=symbol.symbol_id,
        relation=fact.file_relation,
        turn=fact.turn,
        event_id=fact.event_id,
        source=fact.source,
        confidence=fact.confidence,
        metadata={
            "kind": fact.kind,
            "qualified_name": fact.qualified_name,
            "source_unit_id": fact.source_unit_id,
            **dict(fact.metadata),
        },
        raw_evidence=fact.raw_evidence,
    )


def _symbol_symbol_edges_from_facts(
    fact_groups: Sequence[tuple[IfgSymbolRow, Sequence[SymbolFact]]],
) -> tuple[IfgSymbolSymbolEdgeRow, ...]:
    by_path: dict[str, list[tuple[IfgSymbolRow, Sequence[SymbolFact]]]] = {}
    for symbol, facts in fact_groups:
        if symbol.path is None:
            continue
        by_path.setdefault(symbol.path, []).append((symbol, facts))

    edges: list[IfgSymbolSymbolEdgeRow] = []
    for path, path_groups in by_path.items():
        local_symbols = [
            (symbol, facts)
            for symbol, facts in path_groups
            if symbol.kind != "import"
            and any(fact.file_relation in {"defines", "exports"} for fact in facts)
        ]
        # Import declarations already have precise file→symbol edges. Linking
        # every local symbol to every import creates a quadratic, inferred graph
        # with no evidence that a particular symbol uses that import.
        for target_symbol, target_facts in path_groups:
            relation = _symbol_relation_for_facts(target_facts)
            if relation is None or relation not in {
                "calls",
                "uses",
                "extends",
                "references",
            }:
                continue
            target_fact = target_facts[0]
            for source_symbol, source_facts in _containing_source_groups(
                local_symbols,
                target_fact,
            ):
                if source_symbol.symbol_id == target_symbol.symbol_id:
                    continue
                edges.append(
                    _symbol_symbol_edge_from_facts(
                        source_symbol=source_symbol,
                        target_symbol=target_symbol,
                        source_fact=source_facts[0],
                        target_fact=target_fact,
                        path=path,
                        relation=relation,
                        source="ast-grep:symbol-relation",
                        confidence=target_fact.confidence,
                        inference="same_file_symbol_relation",
                    )
                )
    return tuple(edges)


def _symbol_relation_for_facts(facts: Sequence[SymbolFact]) -> str | None:
    for fact in facts:
        if fact.symbol_relation:
            return fact.symbol_relation
    return None


def _containing_source_groups(
    source_groups: Sequence[tuple[IfgSymbolRow, Sequence[SymbolFact]]],
    target_fact: SymbolFact,
) -> tuple[tuple[IfgSymbolRow, Sequence[SymbolFact]], ...]:
    target_span = _fact_span(target_fact)
    if target_span is None:
        return ()
    containing: list[tuple[int, tuple[IfgSymbolRow, Sequence[SymbolFact]]]] = []
    for source_symbol, source_facts in source_groups:
        source_span = _smallest_containing_span(source_facts, target_span)
        if source_span is None:
            continue
        span_size = source_span[1] - source_span[0]
        containing.append((span_size, (source_symbol, source_facts)))
    if not containing:
        return ()
    containing.sort(key=lambda item: item[0])
    return (containing[0][1],)


def _smallest_containing_span(
    facts: Sequence[SymbolFact],
    target_span: tuple[int, int],
) -> tuple[int, int] | None:
    spans = [
        span
        for fact in facts
        if (span := _fact_span(fact)) is not None
        and span[0] <= target_span[0]
        and target_span[1] <= span[1]
    ]
    if not spans:
        return None
    return min(spans, key=lambda span: span[1] - span[0])


def _fact_span(fact: SymbolFact) -> tuple[int, int] | None:
    span = fact.metadata.get("span")
    if not isinstance(span, Mapping):
        return None
    start = span.get("start_index")
    end = span.get("end_index")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return start, end


def _symbol_symbol_edge_from_facts(
    *,
    source_symbol: IfgSymbolRow,
    target_symbol: IfgSymbolRow,
    source_fact: SymbolFact,
    target_fact: SymbolFact,
    path: str,
    relation: str = "imports",
    source: str = "ast-grep:file-imports",
    confidence: str = "medium",
    inference: str = "same_file_import_dependency",
) -> IfgSymbolSymbolEdgeRow:
    return IfgSymbolSymbolEdgeRow(
        edge_id=_stable_id(
            "symbol_symbol_edge",
            source_symbol.symbol_id,
            target_symbol.symbol_id,
            relation,
            source_fact.extractor_version,
        ),
        session_id=source_fact.session_id,
        extractor_version=source_fact.extractor_version,
        from_symbol_id=source_symbol.symbol_id,
        to_symbol_id=target_symbol.symbol_id,
        relation=relation,
        turn=min(source_fact.turn, target_fact.turn),
        event_id=target_fact.event_id,
        source=source,
        confidence=confidence,
        metadata={
            "path": path,
            "from_kind": source_symbol.kind,
            "from_qualified_name": source_symbol.qualified_name,
            "to_kind": target_symbol.kind,
            "to_qualified_name": target_symbol.qualified_name,
            "source_unit_id": source_fact.source_unit_id,
            "target_source_unit_id": target_fact.source_unit_id,
        },
        raw_evidence={
            "path": path,
            "inference": inference,
            "source_symbol_id": source_symbol.symbol_id,
            "target_symbol_id": target_symbol.symbol_id,
            "source_fact": source_fact.raw_evidence,
            "target_fact": target_fact.raw_evidence,
        },
    )


def _action_symbol_edge_from_mention(
    mention: IfgSymbolMentionRow,
    symbol: IfgSymbolRow,
    *,
    resolution: str,
) -> IfgActionSymbolEdgeRow:
    return IfgActionSymbolEdgeRow(
        edge_id=_stable_id(
            "action_symbol_edge",
            mention.action_id,
            symbol.symbol_id,
            "mentions",
            mention.mention_id,
            mention.extractor_version,
        ),
        session_id=mention.session_id,
        extractor_version=mention.extractor_version,
        action_id=mention.action_id,
        symbol_id=symbol.symbol_id,
        relation="mentions",
        turn=mention.turn,
        event_id=mention.event_id,
        source=mention.source,
        confidence=mention.confidence,
        metadata={
            "symbol_text": mention.symbol_text,
            "mention_id": mention.mention_id,
            "source_unit_id": mention.source_unit_id,
            "resolution": resolution,
            "path": mention.path,
            **dict(mention.metadata),
        },
        raw_evidence=mention.raw_evidence,
    )


def _action_node(action: IfgActionRow) -> IfgNodeRow:
    display_name = action.template or action.command or action.tool_name
    return IfgNodeRow(
        node_id=_action_node_id(action.action_id),
        session_id=action.session_id,
        extractor_version=action.extractor_version,
        node_type="action",
        stable_key=action.action_id,
        display_name=display_name,
        first_seen_turn=action.turn,
        last_seen_turn=action.turn,
        observation_count=1,
        source=action.source,
        confidence=action.confidence,
        metadata={
            "action_id": action.action_id,
            "tool_name": action.tool_name,
            "tool_call_id": action.tool_call_id,
            "segment_index": action.segment_index,
            "command": action.command,
            "action_kind": action.action_kind,
            "family": action.family,
            "template": action.template,
        },
        raw_evidence=action.raw_evidence,
    )


def _source_unit_node(unit: IfgSourceUnitRow) -> IfgNodeRow:
    display = f"{unit.kind}:{unit.path or unit.origin}"
    return IfgNodeRow(
        node_id=_source_unit_node_id(unit.source_unit_id),
        session_id=unit.session_id,
        extractor_version=unit.extractor_version,
        node_type="source_unit",
        stable_key=unit.source_unit_id,
        display_name=display,
        first_seen_turn=unit.turn,
        last_seen_turn=unit.turn,
        observation_count=1,
        source=unit.origin,
        confidence=_unit_confidence(unit),
        metadata={
            "source_unit_id": unit.source_unit_id,
            "kind": unit.kind,
            "origin": unit.origin,
            "path": unit.path,
            "relation": unit.relation,
            "content_state": unit.content_state,
            **dict(unit.metadata),
        },
        raw_evidence=unit.raw_evidence,
    )


def _file_node(
    path: str,
    edges: Sequence[IfgActionFileEdgeRow],
    *,
    extractor_version: str,
) -> IfgNodeRow:
    first = edges[0]
    relation_counts = Counter(edge.relation for edge in edges)
    return IfgNodeRow(
        node_id=_file_node_id(first.session_id, extractor_version, path),
        session_id=first.session_id,
        extractor_version=extractor_version,
        node_type="file",
        stable_key=path,
        display_name=path,
        first_seen_turn=min(edge.turn for edge in edges),
        last_seen_turn=max(edge.turn for edge in edges),
        observation_count=len(edges),
        source="action_file_edges",
        confidence=_aggregate_confidence(edge.confidence for edge in edges),
        metadata={"relation_counts": dict(relation_counts)},
        raw_evidence={
            "edge_ids": [edge.edge_id for edge in edges[:100]],
            "relation_counts": dict(relation_counts),
        },
    )


def _symbol_node(symbol: IfgSymbolRow) -> IfgNodeRow:
    return IfgNodeRow(
        node_id=_symbol_node_id(symbol.symbol_id),
        session_id=symbol.session_id,
        extractor_version=symbol.extractor_version,
        node_type="symbol",
        stable_key=symbol.stable_key,
        display_name=symbol.qualified_name,
        first_seen_turn=symbol.first_seen_turn,
        last_seen_turn=symbol.last_seen_turn,
        observation_count=symbol.observation_count,
        source=symbol.source,
        confidence=symbol.confidence,
        metadata={
            "symbol_id": symbol.symbol_id,
            "kind": symbol.kind,
            "qualified_name": symbol.qualified_name,
            "path": symbol.path,
            **dict(symbol.metadata),
        },
        raw_evidence=symbol.raw_evidence,
    )


def _graph_edge(
    *,
    session_id: str,
    extractor_version: str,
    from_node_id: str,
    to_node_id: str,
    relation: str,
    turn: int,
    event_id: int | None,
    source: str,
    confidence: str,
    metadata: Mapping[str, object],
    raw_evidence: Mapping[str, object],
    stable_parts: Sequence[str],
) -> IfgGraphEdgeRow:
    return IfgGraphEdgeRow(
        edge_id=_stable_id(
            "graph_edge",
            session_id,
            extractor_version,
            *stable_parts,
        ),
        session_id=session_id,
        extractor_version=extractor_version,
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        relation=relation,
        turn=turn,
        event_id=event_id,
        source=source,
        confidence=confidence,
        metadata=dict(metadata),
        raw_evidence=dict(raw_evidence),
    )


def _unit_confidence(unit: IfgSourceUnitRow) -> str:
    raw = unit.metadata.get("confidence")
    return raw if isinstance(raw, str) and raw else "medium"


def _metadata_str(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _action_node_id(action_id: str) -> str:
    return f"action:{action_id}"


def _source_unit_node_id(source_unit_id: str) -> str:
    return f"source:{source_unit_id}"


def _file_node_id(session_id: str, extractor_version: str, path: str) -> str:
    return f"file:{_stable_id(session_id, extractor_version, 'file', path)}"


def _symbol_id(session_id: str, extractor_version: str, stable_key: str) -> str:
    return f"symbol:{_stable_id(session_id, extractor_version, stable_key)}"


def _symbol_node_id(symbol_id: str) -> str:
    return f"node:{symbol_id}"
