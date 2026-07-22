# code-health: ignore-file[AM025] -- extraction stores raw tool evidence as JSON
"""Pure IFG extraction from normalized tool events."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from sqlalchemy.engine import RowMapping

from policy_engine.repository_index import RepositoryIndex
from policy_engine.ifg_regions import parse_source_region
from policy_engine.source_parser import (
    BashSegment,
    parse_bash_segments,
)
from policy_engine.source_semantics import (
    BashPathReference,
    analyze_bash_segment,
)

from .normalize import tool_event_from_policy_row
from .project import (
    build_ifg_graph,
    build_ifg_symbols,
    resolve_path_candidates,
    unique_action_symbol_edges,
    unique_actions,
    unique_file_edges,
    unique_file_symbol_edges,
    unique_graph_edges,
    unique_nodes,
    unique_path_candidates,
    unique_source_units,
    unique_symbol_mentions,
    unique_symbol_symbol_edges,
    unique_symbols,
)
from .schema import IFG_EXTRACTOR_VERSION
from .types import (
    IfgActionFileEdgeRow,
    IfgActionRow,
    IfgExtractionRows,
    IfgPathCandidateRow,
    IfgSourceUnitRow,
    IfgSymbolMentionRow,
    IfgToolEvent,
)
from .utils import (
    _first_str,
    _mapping_str,
    _nested_mapping_str,
    _path_info,
    _row_is_error,
    _stable_id,
    _text_hash,
    _tool_result_text,
)

_STRUCTURED_FILE_TOOLS = frozenset({"read", "write", "edit"})
_SEARCH_MATCH_RE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+)(?::(?P<column>\d+))?:(?P<text>.*)$"
)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]{2,}\b")
_MAX_IDENTIFIER_MENTIONS = 40
_SYMBOL_MENTION_STOPWORDS = frozenset(
    {
        "and",
        "as",
        "async",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "def",
        "default",
        "delete",
        "do",
        "else",
        "except",
        "export",
        "extends",
        "finally",
        "for",
        "from",
        "function",
        "if",
        "import",
        "in",
        "interface",
        "let",
        "new",
        "pass",
        "raise",
        "return",
        "switch",
        "try",
        "type",
        "var",
        "while",
        "with",
        "yield",
    }
)


@dataclass(frozen=True, slots=True)
class _SearchMatch:
    path: str
    line: int
    column: int | None
    text: str
    raw_line: str


@dataclass(frozen=True, slots=True)
class _PreparedText:
    text: str | None
    line_range: Mapping[str, object]
    span: Mapping[str, object]
    content_state: str


def extract_ifg_from_policy_tool_rows(
    rows: Sequence[RowMapping],
    *,
    extractor_version: str = IFG_EXTRACTOR_VERSION,
    repository_index: RepositoryIndex | None = None,
) -> IfgExtractionRows:
    return extract_ifg_from_tool_events(
        tuple(tool_event_from_policy_row(row) for row in rows),
        extractor_version=extractor_version,
        repository_index=repository_index,
    )


def extract_ifg_from_tool_events(
    events: Sequence[IfgToolEvent],
    *,
    extractor_version: str = IFG_EXTRACTOR_VERSION,
    repository_index: RepositoryIndex | None = None,
) -> IfgExtractionRows:
    actions: list[IfgActionRow] = []
    file_edges: list[IfgActionFileEdgeRow] = []
    source_units: list[IfgSourceUnitRow] = []
    path_candidates: list[IfgPathCandidateRow] = []
    symbol_mentions: list[IfgSymbolMentionRow] = []
    errors: list[Mapping[str, object]] = []
    for event in events:
        try:
            extracted = _extract_event(event, extractor_version=extractor_version)
        except Exception as exc:
            errors.append(_error_row(event, exc))
            continue
        actions.extend(extracted.actions)
        file_edges.extend(extracted.file_edges)
        source_units.extend(extracted.source_units)
        path_candidates.extend(extracted.path_candidates)
        symbol_mentions.extend(extracted.symbol_mentions)
        errors.extend(extracted.errors)

    unique_units = unique_source_units(source_units)
    unique_candidates = unique_path_candidates(path_candidates)
    unique_edges = resolve_path_candidates(
        anchor_edges=unique_file_edges(file_edges),
        candidates=unique_candidates,
        extractor_version=extractor_version,
    )
    symbol_projection = build_ifg_symbols(
        source_units=unique_units,
        symbol_mentions=unique_symbol_mentions(symbol_mentions),
        file_edges=unique_edges,
        extractor_version=extractor_version,
        repository_index=repository_index,
    )
    errors.extend(symbol_projection.errors)
    symbols = unique_symbols(symbol_projection.symbols)
    action_symbol_edges = unique_action_symbol_edges(
        symbol_projection.action_symbol_edges
    )
    file_symbol_edges = unique_file_symbol_edges(symbol_projection.file_symbol_edges)
    symbol_symbol_edges = unique_symbol_symbol_edges(
        symbol_projection.symbol_symbol_edges
    )
    graph = build_ifg_graph(
        actions=unique_actions(actions),
        source_units=unique_units,
        file_edges=unique_edges,
        action_symbol_edges=action_symbol_edges,
        file_symbol_edges=file_symbol_edges,
        symbol_symbol_edges=symbol_symbol_edges,
        symbols=symbols,
        extractor_version=extractor_version,
    )
    return IfgExtractionRows(
        actions=tuple(actions),
        file_edges=unique_edges,
        source_units=tuple(source_units),
        path_candidates=unique_candidates,
        symbol_mentions=tuple(symbol_mentions),
        symbols=symbols,
        action_symbol_edges=action_symbol_edges,
        file_symbol_edges=file_symbol_edges,
        symbol_symbol_edges=symbol_symbol_edges,
        graph_nodes=unique_nodes(graph.graph_nodes),
        graph_edges=unique_graph_edges(graph.graph_edges),
        errors=tuple(errors),
    )


def _extract_event(
    event: IfgToolEvent,
    *,
    extractor_version: str,
) -> IfgExtractionRows:
    tool_name = event.tool_name
    if tool_name in _STRUCTURED_FILE_TOOLS:
        return _extract_structured_file_tool(
            event,
            extractor_version=extractor_version,
        )
    if tool_name == "bash":
        return _extract_bash_tool(event, extractor_version=extractor_version)
    return IfgExtractionRows(actions=(), file_edges=())


def _extract_structured_file_tool(
    event: IfgToolEvent,
    *,
    extractor_version: str,
) -> IfgExtractionRows:
    args = event.args if isinstance(event.args, Mapping) else {}
    processed = event.processed if isinstance(event.processed, Mapping) else {}
    path = _first_str(args.get("path"), processed.get("path"))
    tool_name = event.tool_name
    action_kind = {"read": "read", "write": "write", "edit": "edit"}[tool_name]
    action = _action_row(
        event,
        segment_index=None,
        command=None,
        action_kind=action_kind,
        family=_family_for_action(action_kind),
        template=tool_name,
        source=f"tool.{tool_name}",
        confidence="high",
        raw_evidence=event.raw_evidence,
        extractor_version=extractor_version,
    )
    if not path:
        return IfgExtractionRows(actions=(action,), file_edges=())

    source_units = _structured_source_units(
        event,
        action,
        path=path,
        extractor_version=extractor_version,
    )
    edge = _file_edge_for_structured_tool(
        event,
        action,
        path=path,
        relation=action_kind,
        source_units=source_units,
        extractor_version=extractor_version,
    )
    return IfgExtractionRows(
        actions=(action,),
        file_edges=(edge,),
        source_units=source_units,
        symbol_mentions=(),
    )


def _structured_source_units(
    event: IfgToolEvent,
    action: IfgActionRow,
    *,
    path: str,
    extractor_version: str,
) -> tuple[IfgSourceUnitRow, ...]:
    args = event.args if isinstance(event.args, Mapping) else {}
    units: list[IfgSourceUnitRow] = []
    if event.tool_name == "read":
        prepared = _prepare_numbered_result(_tool_result_text(event.result))
        units.append(
            _source_unit_row(
                event,
                action,
                kind="read_result",
                origin="result.text",
                path=path,
                relation="read",
                prepared=prepared,
                extractor_version=extractor_version,
                metadata={
                    "parse_symbols": True,
                    "confidence": "high",
                    "content_scope": _prepared_content_scope(prepared),
                },
            )
        )
    elif event.tool_name == "write":
        content = args.get("content")
        prepared = _prepare_raw_text(content if isinstance(content, str) else None)
        units.append(
            _source_unit_row(
                event,
                action,
                kind="write_content",
                origin="args.content",
                path=path,
                relation="write",
                prepared=prepared,
                extractor_version=extractor_version,
                metadata={
                    "parse_symbols": True,
                    "confidence": "high",
                    "content_scope": "full_file",
                },
            )
        )
    elif event.tool_name == "edit":
        old_string = args.get("old_string")
        if isinstance(old_string, str) and old_string:
            units.append(
                _source_unit_row(
                    event,
                    action,
                    kind="edit_old",
                    origin="args.old_string",
                    path=path,
                    relation="read",
                    prepared=_prepare_raw_text(old_string),
                    extractor_version=extractor_version,
                    metadata={
                        "parse_symbols": True,
                        "confidence": "medium",
                        "content_scope": "fragment",
                    },
                )
            )
        new_string = args.get("new_string")
        if isinstance(new_string, str) and new_string:
            units.append(
                _source_unit_row(
                    event,
                    action,
                    kind="edit_new",
                    origin="args.new_string",
                    path=path,
                    relation="edit",
                    prepared=_prepare_raw_text(new_string),
                    extractor_version=extractor_version,
                    metadata={
                        "parse_symbols": True,
                        "confidence": "high",
                        "content_scope": "fragment",
                    },
                )
            )
        result_text = _tool_result_text(event.result)
        if result_text:
            units.append(
                _source_unit_row(
                    event,
                    action,
                    kind="edit_result_snippet",
                    origin="result.text",
                    path=path,
                    relation="edit",
                    prepared=_prepare_numbered_result(result_text),
                    extractor_version=extractor_version,
                    metadata={
                        "parse_symbols": True,
                        "confidence": "high",
                        "content_scope": "fragment",
                    },
                )
            )
    return tuple(units)


def _extract_bash_tool(
    event: IfgToolEvent,
    *,
    extractor_version: str,
) -> IfgExtractionRows:
    cmd = _command_text(event.args)
    if not cmd:
        return IfgExtractionRows(actions=(), file_edges=())
    actions: list[IfgActionRow] = []
    path_candidates: list[IfgPathCandidateRow] = []
    source_units: list[IfgSourceUnitRow] = []
    symbol_mentions: list[IfgSymbolMentionRow] = []
    search_result_action: IfgActionRow | None = None
    search_result_event: IfgToolEvent | None = None
    fallback_result_action: IfgActionRow | None = None
    fallback_result_event: IfgToolEvent | None = None
    current_cwd = event.cwd

    for segment_index, segment in enumerate(parse_bash_segments(cmd)):
        if not segment.argv:
            continue
        segment_event = _event_with_cwd(event, current_cwd)
        analysis = analyze_bash_segment(segment)
        raw_evidence = {
            "tool": "bash",
            "full_command": cmd,
            "segment": _segment_evidence(segment),
            "template_tokens": list(analysis.template_tokens),
            "path_refs": [
                {
                    "path": ref.path,
                    "path_kind": ref.path_kind,
                    "relation": ref.relation,
                    "source": ref.source,
                    "confidence": ref.confidence,
                }
                for ref in analysis.path_refs
            ],
        }
        action = _action_row(
            segment_event,
            segment_index=segment_index,
            command=analysis.command,
            action_kind=analysis.action_kind,
            family=analysis.family,
            template=analysis.template,
            source="bash.ast-grep",
            confidence=analysis.confidence,
            raw_evidence=raw_evidence,
            extractor_version=extractor_version,
        )
        actions.append(action)
        unit = _source_unit_row(
            segment_event,
            action,
            kind="bash_segment",
            origin="args.cmd",
            path=None,
            relation=analysis.action_kind,
            prepared=_prepare_raw_text(segment.text),
            extractor_version=extractor_version,
            metadata={
                "parse_symbols": False,
                "confidence": analysis.confidence,
                "command": analysis.command,
                "template": analysis.template,
                "segment_index": segment_index,
            },
        )
        source_units.append(unit)
        if fallback_result_action is None and analysis.action_kind not in {
            "control",
            "filter",
        }:
            fallback_result_action = action
            fallback_result_event = segment_event
        if (
            search_result_action is None
            and analysis.family == "read"
            and analysis.action_kind != "filter"
        ):
            search_result_action = action
            search_result_event = segment_event
        for ref in analysis.path_refs:
            path_candidates.append(
                _path_candidate_row(
                    segment_event,
                    action,
                    source_unit_id=unit.source_unit_id,
                    ref=ref,
                    extractor_version=extractor_version,
                )
            )
        for mention in analysis.symbol_mentions:
            symbol_mentions.append(
                _symbol_mention_row(
                    segment_event,
                    action,
                    source_unit_id=unit.source_unit_id,
                    symbol_text=mention.text,
                    source=mention.source,
                    confidence=mention.confidence,
                    path=None,
                    raw_evidence={"segment": segment.text, "mention": mention.text},
                    extractor_version=extractor_version,
                )
            )
        current_cwd = _updated_shell_cwd(segment, current_cwd)

    if search_result_action is None:
        search_result_action = fallback_result_action
        search_result_event = fallback_result_event
    if search_result_action is not None and search_result_event is not None:
        result_rows = _search_result_rows(
            search_result_event,
            search_result_action,
            extractor_version=extractor_version,
        )
        path_candidates.extend(result_rows.path_candidates)
        source_units.extend(result_rows.source_units)
        symbol_mentions.extend(result_rows.symbol_mentions)
    return IfgExtractionRows(
        actions=tuple(actions),
        file_edges=(),
        source_units=tuple(source_units),
        path_candidates=tuple(path_candidates),
        symbol_mentions=tuple(symbol_mentions),
    )


def _file_edge_for_structured_tool(
    event: IfgToolEvent,
    action: IfgActionRow,
    *,
    path: str,
    relation: str,
    source_units: Sequence[IfgSourceUnitRow],
    extractor_version: str,
) -> IfgActionFileEdgeRow:
    unit = _representative_source_unit(source_units)
    return _file_edge_row(
        event,
        action,
        path=path,
        relation=relation,
        source="args.path",
        confidence="high",
        raw_evidence={"path": path, "tool": event.tool_name},
        extractor_version=extractor_version,
        content_hash=_event_content_hash(event),
        before_hash=_event_previous_content_hash(event),
        after_hash=_event_content_hash(event),
        content_state=unit.content_state if unit is not None else None,
        line_range=unit.line_range if unit is not None else {},
        span=unit.span if unit is not None else {},
        metadata={
            "anchor": True,
            "entity_kind": "file",
            "evidence": "tool_contract",
            "existence": _structured_file_existence(event),
            "source_unit_ids": [item.source_unit_id for item in source_units],
            "result_content_hash": _event_result_content_hash(event),
        },
    )


def _representative_source_unit(
    source_units: Sequence[IfgSourceUnitRow],
) -> IfgSourceUnitRow | None:
    for unit in source_units:
        if unit.kind in {"edit_result_snippet", "read_result", "write_content"}:
            return unit
    return source_units[0] if source_units else None


def _search_result_rows(
    event: IfgToolEvent,
    action: IfgActionRow,
    *,
    extractor_version: str,
) -> IfgExtractionRows:
    stdout = _bash_stdout_text(event)
    if not stdout:
        return IfgExtractionRows(actions=(), file_edges=())
    matches = _parse_search_matches(stdout)
    if not matches:
        return IfgExtractionRows(actions=(), file_edges=())
    path_candidates: list[IfgPathCandidateRow] = []
    source_units: list[IfgSourceUnitRow] = []
    symbol_mentions: list[IfgSymbolMentionRow] = []
    for path, path_matches in _group_search_matches(matches).items():
        path_info = _bash_result_path_info(path, cwd=event.cwd)
        normalized_path = str(path_info["normalized_path"])
        content_text = "\n".join(match.text for match in path_matches)
        matched_lines = [match.line for match in path_matches]
        columns = [match.column for match in path_matches if match.column is not None]
        line_range: dict[str, object] = {
            "matched_lines": matched_lines,
            "start_line": min(matched_lines),
            "end_line": max(matched_lines),
            "partial": True,
        }
        if columns:
            line_range["matched_columns"] = columns
        unit = _source_unit_row(
            event,
            action,
            kind="bash_search_result",
            origin="result.stdout",
            path=normalized_path,
            relation="supports",
            prepared=_PreparedText(
                text=content_text,
                line_range=line_range,
                span={"source": "bash.result.search_match"},
                content_state="matched_lines",
            ),
            extractor_version=extractor_version,
            metadata={
                "parse_symbols": False,
                "confidence": "medium",
                "content_scope": "matched_lines",
                "evidence_role": "support",
                "match_count": len(path_matches),
                **path_info,
            },
        )
        source_units.append(unit)
        path_candidates.append(
            _path_candidate_row(
                event,
                action,
                source_unit_id=unit.source_unit_id,
                ref=BashPathReference(
                    path=normalized_path,
                    path_kind="file",
                    relation="read",
                    source="bash.result.search_match",
                    confidence="medium",
                ),
                extractor_version=extractor_version,
                metadata={
                    "unit_hash": unit.unit_hash,
                    "match_count": len(path_matches),
                },
                raw_evidence={
                    "path": path,
                    "lines": [match.raw_line for match in path_matches[:20]],
                },
            )
        )
        for symbol_text in _symbol_mentions_from_text(content_text):
            symbol_mentions.append(
                _symbol_mention_row(
                    event,
                    action,
                    source_unit_id=unit.source_unit_id,
                    symbol_text=symbol_text,
                    source="bash.result.search_match",
                    confidence="low",
                    path=normalized_path,
                    raw_evidence={
                        "path": path,
                        "matched_lines": [
                            match.raw_line for match in path_matches[:20]
                        ],
                    },
                    extractor_version=extractor_version,
                )
            )
    return IfgExtractionRows(
        actions=(),
        file_edges=(),
        source_units=tuple(source_units),
        path_candidates=tuple(path_candidates),
        symbol_mentions=tuple(symbol_mentions),
    )


def _source_unit_row(
    event: IfgToolEvent,
    action: IfgActionRow,
    *,
    kind: str,
    origin: str,
    path: str | None,
    relation: str,
    prepared: _PreparedText,
    extractor_version: str,
    metadata: Mapping[str, object],
) -> IfgSourceUnitRow:
    path_info = _path_info(path, cwd=event.cwd) if path else {}
    normalized_path = str(path_info["normalized_path"]) if path else None
    unit_hash = _text_hash(prepared.text) if prepared.text is not None else None
    source_unit_id = _stable_id(
        "source_unit",
        action.action_id,
        kind,
        origin,
        normalized_path or "",
        unit_hash or "",
        extractor_version,
    )
    return IfgSourceUnitRow(
        source_unit_id=source_unit_id,
        session_id=action.session_id,
        extractor_version=extractor_version,
        action_id=action.action_id,
        kind=kind,
        origin=origin,
        path=normalized_path,
        relation=relation,
        turn=action.turn,
        event_id=event.event_id,
        tool_name=event.tool_name,
        language=_language_hint(normalized_path),
        content_hash=_event_content_hash(event),
        previous_content_hash=_event_previous_content_hash(event),
        result_content_hash=_event_result_content_hash(event),
        unit_hash=unit_hash,
        content_state=prepared.content_state,
        line_range=prepared.line_range,
        span=prepared.span,
        content_text=prepared.text,
        metadata={**path_info, **dict(metadata)},
        raw_evidence={
            "tool_name": event.tool_name,
            "origin": origin,
            "kind": kind,
            "args": event.args,
            "processed": event.processed,
            "state": event.state,
            "text_preview": prepared.text[:1000] if prepared.text else None,
        },
    )


def _symbol_mentions_from_text(text: str) -> tuple[str, ...]:
    mentions: dict[str, None] = {}
    for identifier in _IDENTIFIER_RE.findall(text):
        if identifier in _SYMBOL_MENTION_STOPWORDS:
            continue
        mentions.setdefault(identifier, None)
        if len(mentions) >= _MAX_IDENTIFIER_MENTIONS:
            break
    return tuple(mentions)


def _symbol_mention_row(
    event: IfgToolEvent,
    action: IfgActionRow,
    *,
    source_unit_id: str | None,
    symbol_text: str,
    source: str,
    confidence: str,
    path: str | None,
    raw_evidence: Mapping[str, object],
    extractor_version: str,
) -> IfgSymbolMentionRow:
    mention_id = _stable_id(
        "symbol_mention",
        action.action_id,
        source_unit_id or "",
        symbol_text,
        source,
        path or "",
        extractor_version,
    )
    return IfgSymbolMentionRow(
        mention_id=mention_id,
        session_id=action.session_id,
        extractor_version=extractor_version,
        action_id=action.action_id,
        source_unit_id=source_unit_id,
        symbol_text=symbol_text,
        turn=action.turn,
        event_id=event.event_id,
        source=source,
        confidence=confidence,
        path=path,
        metadata={"path": path, "source_unit_id": source_unit_id},
        raw_evidence=dict(raw_evidence),
    )


def _action_row(
    event: IfgToolEvent,
    *,
    segment_index: int | None,
    command: str | None,
    action_kind: str,
    family: str,
    template: str | None,
    source: str,
    confidence: str,
    raw_evidence: Mapping[str, object],
    extractor_version: str,
) -> IfgActionRow:
    action_id = _stable_id(
        "action",
        event.session_id,
        str(event.tool_call_id or f"event:{event.event_id}"),
        event.tool_name,
        "" if segment_index is None else str(segment_index),
        extractor_version,
    )
    return IfgActionRow(
        action_id=action_id,
        session_id=event.session_id,
        turn=event.turn,
        event_id=event.event_id,
        tool_call_id=str(event.tool_call_id) if event.tool_call_id else None,
        tool_name=event.tool_name,
        segment_index=segment_index,
        command=command,
        action_kind=action_kind,
        family=family,
        template=template,
        source=source,
        confidence=confidence,
        extractor_version=extractor_version,
        raw_evidence=dict(raw_evidence),
    )


def _file_edge_row(
    event: IfgToolEvent,
    action: IfgActionRow,
    *,
    path: str,
    relation: str,
    source: str,
    confidence: str,
    raw_evidence: Mapping[str, object],
    extractor_version: str,
    content_hash: str | None = None,
    before_hash: str | None = None,
    after_hash: str | None = None,
    content_state: str | None = None,
    line_range: Mapping[str, object] | None = None,
    span: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> IfgActionFileEdgeRow:
    path_info = _path_info(path, cwd=event.cwd)
    normalized_path = str(path_info["normalized_path"])
    edge_id = _stable_id(
        "edge",
        action.action_id,
        normalized_path,
        relation,
        source,
        extractor_version,
    )
    return IfgActionFileEdgeRow(
        edge_id=edge_id,
        session_id=action.session_id,
        action_id=action.action_id,
        path=normalized_path,
        relation=relation,
        turn=action.turn,
        event_id=event.event_id,
        source=source,
        confidence=confidence,
        is_anchor=True,
        extractor_version=extractor_version,
        content_hash=content_hash,
        before_hash=before_hash,
        after_hash=after_hash,
        content_state=content_state,
        line_range=dict(line_range or {}),
        span=dict(span or {}),
        metadata={**path_info, **dict(metadata or {})},
        raw_evidence=dict(raw_evidence),
    )


def _path_candidate_row(
    event: IfgToolEvent,
    action: IfgActionRow,
    *,
    source_unit_id: str | None,
    ref: BashPathReference,
    extractor_version: str,
    metadata: Mapping[str, object] | None = None,
    raw_evidence: Mapping[str, object] | None = None,
) -> IfgPathCandidateRow:
    path_info = _path_info(ref.path, cwd=event.cwd)
    normalized_path = str(path_info["normalized_path"])
    candidate_id = _stable_id(
        "path_candidate",
        action.action_id,
        source_unit_id or "",
        normalized_path,
        ref.path_kind,
        ref.relation,
        ref.source,
        extractor_version,
    )
    return IfgPathCandidateRow(
        candidate_id=candidate_id,
        session_id=action.session_id,
        extractor_version=extractor_version,
        action_id=action.action_id,
        source_unit_id=source_unit_id,
        path_text=ref.path,
        normalized_path=normalized_path,
        path_kind=ref.path_kind,
        relation=ref.relation,
        turn=action.turn,
        event_id=event.event_id,
        source=ref.source,
        confidence=ref.confidence,
        metadata={**path_info, **dict(metadata or {})},
        raw_evidence={
            "path": ref.path,
            "path_kind": ref.path_kind,
            "relation": ref.relation,
            "source": ref.source,
            "action_template": action.template,
            **dict(raw_evidence or {}),
        },
    )


def _structured_file_existence(event: IfgToolEvent) -> str:
    if _row_is_error(event.result, event.processed):
        return "unknown"
    if event.tool_name == "read":
        return "observed_at_event"
    return "present_after_event"


def _prepared_content_scope(prepared: _PreparedText) -> str:
    return "fragment" if prepared.line_range.get("partial", True) else "full_file"


def _prepare_raw_text(text: str | None) -> _PreparedText:
    if not text:
        return _PreparedText(
            text=None,
            line_range={},
            span={},
            content_state="content_unavailable",
        )
    line_count = len(text.splitlines())
    return _PreparedText(
        text=text,
        line_range={
            "start_line": 1,
            "end_line": max(line_count, 1),
            "partial": False,
        },
        span={"source": "raw_text"},
        content_state="source_text",
    )


def _prepare_numbered_result(text: str | None) -> _PreparedText:
    if not text:
        return _prepare_raw_text(None)
    parsed = parse_source_region(text)
    if parsed is None:
        return _prepare_raw_text(None)
    return _PreparedText(
        text=parsed.content_text,
        line_range=parsed.line_range,
        span={
            "source": (
                "tool.result.line_numbered_snippet"
                if parsed.numbered
                else "tool.result.text"
            )
        },
        content_state="source_text",
    )


def _content_state(
    *,
    text: str | None,
    content_hash: str | None,
    previous_hash: str | None,
    relation: str,
) -> str:
    if text is not None and content_hash:
        return "content_available"
    if text is not None:
        return "source_text"
    if content_hash or previous_hash:
        return "hash_only"
    if relation in {"write", "edit"}:
        return "content_unavailable"
    return "unavailable"


def _event_with_cwd(event: IfgToolEvent, cwd: str | None) -> IfgToolEvent:
    if cwd == event.cwd:
        return event
    return replace(event, cwd=cwd)


def _updated_shell_cwd(segment: BashSegment, cwd: str | None) -> str | None:
    if not segment.argv or segment.argv[0] != "cd" or len(segment.argv) < 2:
        return cwd
    target = segment.argv[1]
    if not target or target.startswith("-"):
        return cwd
    clean = target.strip().strip("\"'")
    if clean.startswith("/"):
        return posixpath.normpath(clean)
    if cwd and cwd.startswith("/"):
        return posixpath.normpath(posixpath.join(cwd, clean))
    return posixpath.normpath(clean)


def _bash_stdout_text(event: IfgToolEvent) -> str | None:
    text = _tool_result_text(event.result)
    if not text:
        return None
    marker = "\nStdout:\n"
    if marker not in text:
        return text
    stdout = text.split(marker, 1)[1]
    stderr_marker = "\nStderr:\n"
    if stderr_marker in stdout:
        stdout = stdout.split(stderr_marker, 1)[0]
    return stdout or None


def _parse_search_matches(stdout: str) -> tuple[_SearchMatch, ...]:
    matches: list[_SearchMatch] = []
    for line in stdout.splitlines():
        parsed = _parse_search_match_line(line)
        if parsed is not None:
            matches.append(parsed)
    return tuple(matches)


def _parse_search_match_line(line: str) -> _SearchMatch | None:
    match = _SEARCH_MATCH_RE.match(line)
    if match is None:
        return None
    path = match.group("path").strip()
    if not _looks_like_result_path(path):
        return None
    column_raw = match.group("column")
    return _SearchMatch(
        path=path,
        line=int(match.group("line")),
        column=int(column_raw) if column_raw is not None else None,
        text=match.group("text"),
        raw_line=line,
    )


def _looks_like_result_path(path: str) -> bool:
    if not path or path.startswith(("http://", "https://")):
        return False
    if path in {".", ".."}:
        return False
    return path.startswith(("/", "./", "../")) or "/" in path or "." in path


def _group_search_matches(
    matches: Sequence[_SearchMatch],
) -> dict[str, list[_SearchMatch]]:
    grouped: dict[str, list[_SearchMatch]] = {}
    for match in matches:
        grouped.setdefault(match.path, []).append(match)
    return grouped


def _bash_result_path_info(path: str, *, cwd: str | None) -> Mapping[str, object]:
    if path.startswith("/"):
        return _path_info(path, cwd=cwd)
    if cwd and cwd.startswith("/"):
        normalized = posixpath.normpath(posixpath.join(cwd, path))
        return {
            "original_path": path,
            "normalized_path": normalized,
            "path_kind": "cwd_relative",
            "cwd": cwd,
        }
    return _path_info(path, cwd=cwd)


def _event_content_hash(event: IfgToolEvent) -> str | None:
    return _first_str(
        _mapping_str(event.processed, "content_hash"),
        _mapping_str(event.result, "content_hash"),
        _mapping_str(event.state, "content_hash"),
        _nested_mapping_str(event.result, "extras", "content_hash"),
        _nested_mapping_str(event.state, "processed", "content_hash"),
    )


def _event_previous_content_hash(event: IfgToolEvent) -> str | None:
    return _first_str(
        _mapping_str(event.processed, "previous_content_hash"),
        _mapping_str(event.result, "previous_content_hash"),
        _mapping_str(event.state, "previous_content_hash"),
        _nested_mapping_str(event.result, "extras", "previous_content_hash"),
        _nested_mapping_str(event.state, "processed", "previous_content_hash"),
    )


def _event_result_content_hash(event: IfgToolEvent) -> str | None:
    return _first_str(
        _mapping_str(event.processed, "result_content_hash"),
        _mapping_str(event.result, "result_content_hash"),
        _mapping_str(event.state, "result_content_hash"),
        _nested_mapping_str(event.result, "extras", "result_content_hash"),
        _nested_mapping_str(event.state, "processed", "result_content_hash"),
    )


def _language_hint(path: str | None) -> str | None:
    if path is None:
        return None
    lowered = path.lower()
    if lowered.endswith((".ts", ".mts", ".cts", ".d.ts")):
        return "typescript"
    if lowered.endswith(".tsx"):
        return "tsx"
    if lowered.endswith((".js", ".mjs", ".cjs")):
        return "javascript"
    if lowered.endswith(".jsx"):
        return "jsx"
    if lowered.endswith((".py", ".pyi")):
        return "python"
    return None


def _family_for_action(action_kind: str) -> str:
    if action_kind in {"write", "edit", "create", "delete"}:
        return "write"
    if action_kind in {"read", "reference", "filter"}:
        return "read"
    if action_kind in {"control", "exec", "test"}:
        return action_kind
    return "read"


def _command_text(args: object) -> str:
    if not isinstance(args, Mapping):
        return ""
    raw = args.get("cmd") or args.get("command")
    return raw if isinstance(raw, str) else ""


def _error_row(event: IfgToolEvent, exc: Exception) -> Mapping[str, object]:
    return {
        "turn": event.turn,
        "event_id": event.event_id,
        "tool_call_id": event.tool_call_id,
        "tool_name": event.tool_name,
        "error": f"{type(exc).__name__}: {exc}",
        "raw_evidence": event.raw_evidence,
    }


def _segment_evidence(segment: BashSegment) -> Mapping[str, object]:
    return {
        "argv": list(segment.argv),
        "text": segment.text,
        "parser": segment.parser,
        "pipeline_index": segment.pipeline_index,
        "depth": segment.depth,
        "redirects": [
            {
                "kind": redirect.kind,
                "operator": redirect.operator,
                "destination": redirect.destination,
                "descriptor": redirect.descriptor,
                "text": redirect.text,
            }
            for redirect in segment.redirects
        ],
    }
