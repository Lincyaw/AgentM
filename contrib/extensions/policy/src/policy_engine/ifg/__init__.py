"""Information Flow Graph extraction for policy-observed tool events."""

from __future__ import annotations

from .extract import extract_ifg_from_policy_tool_rows, extract_ifg_from_tool_events
from .project import build_ifg_graph, build_ifg_symbols
from .schema import (
    IFG_EXTRACTOR_VERSION,
    IFG_TABLES,
    delete_ifg_session,
    ensure_ifg_schema,
)
from .service import (
    backfill_ifg_from_policy_events,
    backfill_ifg_from_trajectory_turns,
    persist_ifg_tool_events,
    rebuild_ifg_projection,
)
from .types import (
    IfgActionFileEdgeRow,
    IfgActionRow,
    IfgActionSymbolEdgeRow,
    IfgBackfillResult,
    IfgExtractionRows,
    IfgFileSymbolEdgeRow,
    IfgGraphEdgeRow,
    IfgNodeRow,
    IfgPathCandidateRow,
    IfgSourceUnitRow,
    IfgSymbolMentionRow,
    IfgSymbolRow,
    IfgSymbolSymbolEdgeRow,
    IfgToolEvent,
)

__all__ = [
    "IFG_TABLES",
    "IFG_EXTRACTOR_VERSION",
    "IfgActionFileEdgeRow",
    "IfgActionRow",
    "IfgActionSymbolEdgeRow",
    "IfgBackfillResult",
    "IfgExtractionRows",
    "IfgFileSymbolEdgeRow",
    "IfgGraphEdgeRow",
    "IfgNodeRow",
    "IfgPathCandidateRow",
    "IfgSourceUnitRow",
    "IfgSymbolMentionRow",
    "IfgSymbolRow",
    "IfgSymbolSymbolEdgeRow",
    "IfgToolEvent",
    "backfill_ifg_from_policy_events",
    "backfill_ifg_from_trajectory_turns",
    "build_ifg_graph",
    "build_ifg_symbols",
    "delete_ifg_session",
    "ensure_ifg_schema",
    "extract_ifg_from_policy_tool_rows",
    "extract_ifg_from_tool_events",
    "persist_ifg_tool_events",
    "rebuild_ifg_projection",
]
