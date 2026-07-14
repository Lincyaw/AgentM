"""Trajectory-index query atom — read a persisted index, expose its query tools.

The companion to ``atom.py``: where that one *builds* an index live from the
running session, this one *loads* a durable ``index.json`` (written by a prior
build / eval run) and registers the same one set of read tools over it —
``search_symbols`` / ``get_symbol_context`` / ``get_insights`` (from
``query_tools``). Mount it wherever an already-built index needs to be queried
(the auditor is the primary consumer): the parent writes the index artifact and
passes its path; this atom loads it, no in-memory dict shipped through config.
"""

from __future__ import annotations

from pathlib import Path

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, Field

from .ir.index import TrajectoryIndex
from .query_tools import INDEX_SERVICE_KEY, register_query_tools


class TrajectoryIndexQueryConfig(BaseModel):
    index_path: str = Field(
        default="",
        description="Path to a persisted index.json (written by TrajectoryIndex.dump). "
        "Loaded read-only at install; empty mounts the tools over an empty index.",
    )


MANIFEST = ExtensionManifest(
    name="trajectory_index_query",
    description=(
        "Query a persisted trajectory semantic index: search symbols, get symbol "
        "context, and review the analysis passes' insights (grounding flags, "
        "unsupported claims, unmet constraints)."
    ),
    registers=(
        "tool:search_symbols",
        "tool:get_symbol_context",
        "tool:get_insights",
    ),
    config_schema=TrajectoryIndexQueryConfig,
)


def install(api: ExtensionAPI, config: TrajectoryIndexQueryConfig) -> None:
    index = TrajectoryIndex()
    if config.index_path and Path(config.index_path).is_file():
        try:
            index = TrajectoryIndex.load(config.index_path)
        except Exception:
            logger.exception(
                "trajectory_index_query: failed to load index from {}", config.index_path
            )
    elif config.index_path:
        logger.warning(
            "trajectory_index_query: index_path {} does not exist; serving empty index",
            config.index_path,
        )
    api.set_service(INDEX_SERVICE_KEY, index)
    register_query_tools(api)
