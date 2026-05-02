"""Scenario-local wrapper that lazily loads the RCA DuckDB tool package."""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="duckdb_sql",
    description="Lazy wrapper for the RCA DuckDB SQL tools.",
    registers=(),
)


def install(api: Any, config: dict[str, Any]) -> Any:
    from agentm_rca.tools.duckdb_sql import install as package_install

    return package_install(api, config)

