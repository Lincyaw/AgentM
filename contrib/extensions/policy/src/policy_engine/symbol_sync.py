"""Sync repository index symbols to the PG data plane.

Uses PgQuerySource for writes (shared connection), not standalone psycopg.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loguru import logger

from .ifg.repository_index import RepositoryIndex
from .ifg.source_parser import (
    SymbolExtractionInput,
    SymbolFact,
)
from .pg_query import PgQuerySource

_EXTRACTOR_VERSION = "policy-symbol-sync-v1"


@dataclass(slots=True)
class SymbolRow:
    session_id: str
    path: str
    symbol_name: str
    symbol_kind: str | None
    relation: str
    start_line: int | None
    end_line: int | None
    source: str


def extract_symbols_for_paths(
    repo_index: RepositoryIndex,
    *,
    session_id: str,
    paths: Sequence[str],
) -> tuple[SymbolRow, ...]:
    inputs = tuple(
        SymbolExtractionInput(
            session_id=session_id,
            extractor_version=_EXTRACTOR_VERSION,
            action_id="",
            source_unit_id="",
            path=path,
            relation="read",
            turn=0,
            event_id=None,
            tool_name="repository_index",
            content_hash=None,
            unit_hash=None,
            content_text=None,
            metadata={"content_scope": "full_file"},
            raw_evidence={},
        )
        for path in paths
        if repo_index.contains_file(path)
    )
    if not inputs:
        return ()
    result = repo_index.extract_symbols(inputs, extractor_version=_EXTRACTOR_VERSION)
    return tuple(_fact_to_row(fact, session_id) for fact in result.symbols)


def _fact_to_row(fact: SymbolFact, session_id: str) -> SymbolRow:
    span = fact.metadata.get("span") if isinstance(fact.metadata, Mapping) else None
    start_line = None
    end_line = None
    if isinstance(span, Mapping):
        sl = span.get("start_line")
        el = span.get("end_line")
        if isinstance(sl, int):
            start_line = sl
        if isinstance(el, int):
            end_line = el
    return SymbolRow(
        session_id=session_id,
        path=fact.path,
        symbol_name=fact.qualified_name,
        symbol_kind=fact.kind,
        relation=fact.file_relation,
        start_line=start_line,
        end_line=end_line,
        source=fact.source,
    )


def write_symbols(
    source: PgQuerySource,
    rows: Sequence[SymbolRow],
    *,
    session_id: str,
    paths: Sequence[str],
) -> int:
    if not rows:
        return 0
    try:
        source.execute(
            "DELETE FROM policy.file_symbols WHERE session_id = %s AND path = ANY(%s)",
            (session_id, list(paths)),
        )
        source.executemany(
            "INSERT INTO policy.file_symbols "
            "(session_id, path, symbol_name, symbol_kind, "
            " relation, start_line, end_line, source) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    r.session_id,
                    r.path,
                    r.symbol_name,
                    r.symbol_kind,
                    r.relation,
                    r.start_line,
                    r.end_line,
                    r.source,
                )
                for r in rows
            ],
        )
        return len(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("symbol_sync: PG write failed: {}", exc)
        return 0
