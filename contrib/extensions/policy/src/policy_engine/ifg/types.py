# code-health: ignore-file[AM025] -- IFG DTOs carry raw trajectory evidence as JSON
"""Typed rows used by policy IFG extraction and persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IfgToolEvent:
    """Source-neutral normalized tool event consumed by the IFG extractor."""

    session_id: str
    turn: int
    event_id: int | None
    tool_call_id: str | None
    phase: str
    tool_name: str
    args: object
    result: object
    processed: object
    state: object
    cwd: str | None
    ts: float
    source: str
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgNodeRow:
    node_id: str
    session_id: str
    extractor_version: str
    node_type: str
    stable_key: str
    display_name: str
    first_seen_turn: int
    last_seen_turn: int
    observation_count: int
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgGraphEdgeRow:
    edge_id: str
    session_id: str
    extractor_version: str
    from_node_id: str
    to_node_id: str
    relation: str
    turn: int
    event_id: int | None
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgActionRow:
    action_id: str
    session_id: str
    turn: int
    event_id: int | None
    tool_call_id: str | None
    tool_name: str
    segment_index: int | None
    command: str | None
    action_kind: str
    family: str
    template: str | None
    source: str
    confidence: str
    extractor_version: str
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgSourceUnitRow:
    """One parseable or mention-bearing source fragment observed in a tool event.

    Source units are the canonical bridge between tool I/O and code symbols.
    Examples: ``bash.args.cmd`` segment, ``read.result`` file content,
    ``edit.args.new_string``, ``write.args.content``, or a bash search result line.
    """

    source_unit_id: str
    session_id: str
    extractor_version: str
    action_id: str
    kind: str
    origin: str
    path: str | None
    relation: str
    turn: int
    event_id: int | None
    tool_name: str
    language: str | None
    content_hash: str | None
    previous_content_hash: str | None
    result_content_hash: str | None
    unit_hash: str | None
    content_state: str
    line_range: Mapping[str, object]
    span: Mapping[str, object]
    content_text: str | None
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgActionFileEdgeRow:
    edge_id: str
    session_id: str
    action_id: str
    path: str
    relation: str
    turn: int
    event_id: int | None
    source: str
    confidence: str
    is_anchor: bool
    extractor_version: str
    content_hash: str | None
    before_hash: str | None
    after_hash: str | None
    content_state: str | None
    line_range: Mapping[str, object]
    span: Mapping[str, object]
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgPathCandidateRow:
    """A bash-derived path that may support an independently anchored file."""

    candidate_id: str
    session_id: str
    extractor_version: str
    action_id: str
    source_unit_id: str | None
    path_text: str
    normalized_path: str
    path_kind: str
    relation: str
    turn: int
    event_id: int | None
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgSymbolMentionRow:
    """Unresolved symbol-like text emitted by a source unit."""

    mention_id: str
    session_id: str
    extractor_version: str
    action_id: str
    source_unit_id: str | None
    symbol_text: str
    turn: int
    event_id: int | None
    source: str
    confidence: str
    path: str | None
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgSymbolRow:
    symbol_id: str
    session_id: str
    extractor_version: str
    kind: str
    qualified_name: str
    path: str | None
    stable_key: str
    first_seen_turn: int
    last_seen_turn: int
    observation_count: int
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgActionSymbolEdgeRow:
    edge_id: str
    session_id: str
    extractor_version: str
    action_id: str
    symbol_id: str
    relation: str
    turn: int
    event_id: int | None
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgFileSymbolEdgeRow:
    edge_id: str
    session_id: str
    extractor_version: str
    path: str
    symbol_id: str
    relation: str
    turn: int
    event_id: int | None
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgSymbolSymbolEdgeRow:
    edge_id: str
    session_id: str
    extractor_version: str
    from_symbol_id: str
    to_symbol_id: str
    relation: str
    turn: int
    event_id: int | None
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IfgExtractionRows:
    actions: tuple[IfgActionRow, ...]
    file_edges: tuple[IfgActionFileEdgeRow, ...]
    source_units: tuple[IfgSourceUnitRow, ...] = ()
    path_candidates: tuple[IfgPathCandidateRow, ...] = ()
    symbol_mentions: tuple[IfgSymbolMentionRow, ...] = ()
    symbols: tuple[IfgSymbolRow, ...] = ()
    action_symbol_edges: tuple[IfgActionSymbolEdgeRow, ...] = ()
    file_symbol_edges: tuple[IfgFileSymbolEdgeRow, ...] = ()
    symbol_symbol_edges: tuple[IfgSymbolSymbolEdgeRow, ...] = ()
    graph_nodes: tuple[IfgNodeRow, ...] = ()
    graph_edges: tuple[IfgGraphEdgeRow, ...] = ()
    errors: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class IfgBackfillResult:
    session_id: str
    extractor_version: str
    source_events: int
    source_units: int
    graph_nodes: int
    graph_edges: int
    actions: int
    files: int
    file_edges: int
    path_candidates: int
    unresolved_path_candidates: int
    symbols: int
    action_symbol_edges: int
    file_symbol_edges: int
    symbol_symbol_edges: int
    symbol_mentions: int
    unresolved_symbol_mentions: int
    errors: int
    deleted: int
    action_kinds: Mapping[str, int]
