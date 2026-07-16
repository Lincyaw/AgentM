"""Read-only tools for the content-addressed atom catalog."""

from __future__ import annotations

import json
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import ExtensionAPI, FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest

from ._root import resolve_root


class ToolCatalogBrowseConfig(BaseModel):
    root: str | None = None


MANIFEST = ExtensionManifest(
    name="tool_catalog_browse",
    description="Browse immutable atom snapshots and their attributed runs.",
    registers=(
        "tool:catalog_list_versions",
        "tool:catalog_get_manifest",
        "tool:catalog_runs_for",
        "tool:catalog_get_source",
        "tool:list_atoms",
    ),
    config_schema=ToolCatalogBrowseConfig,
    api_version=1,
    tier=1,
)

_ATOM_PARAMS: Final = {
    "type": "object",
    "properties": {
        "atom": {
            "type": "string",
            "description": "Atom name.",
        }
    },
    "required": ["atom"],
    "additionalProperties": False,
}

_VERSION_PARAMS: Final = {
    "type": "object",
    "properties": {
        "atom": {"type": "string", "description": "Atom name."},
        "version": {
            "type": "string",
            "description": "12-character content hash.",
        },
    },
    "required": ["atom", "version"],
    "additionalProperties": False,
}

_RUNS_FOR_PARAMS: Final = {
    "type": "object",
    "properties": {
        "fingerprint": {
            "description": "Exact atom-set fingerprint mapping or 'atom@version'.",
            "oneOf": [{"type": "object"}, {"type": "string"}],
        }
    },
    "required": ["fingerprint"],
    "additionalProperties": False,
}

_EMPTY_PARAMS: Final = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: ToolCatalogBrowseConfig) -> None:
    root = resolve_root(api, config.root)

    async def _list_versions(args: dict[str, Any]) -> ToolResult:
        atom = str(args["atom"])
        try:
            return _json_result(api.catalog.list_versions(atom, root))
        except Exception as exc:  # noqa: BLE001
            logger.debug("catalog list_versions failed: {}", exc)
            return _error(f"Failed to list versions for {atom}: {exc}")

    async def _get_manifest(args: dict[str, Any]) -> ToolResult:
        atom = str(args["atom"])
        version = str(args["version"])
        try:
            return _json_result(api.catalog.get_manifest_at(atom, version, root))
        except Exception as exc:  # noqa: BLE001
            logger.debug("catalog get_manifest failed: {}", exc)
            return _error(f"Failed to load manifest for {atom}@{version}: {exc}")

    async def _runs_for(args: dict[str, Any]) -> ToolResult:
        try:
            return _json_result(api.catalog.runs_for(args["fingerprint"], root))
        except Exception as exc:  # noqa: BLE001
            logger.debug("catalog runs_for failed: {}", exc)
            return _error(f"Failed to resolve catalog runs: {exc}")

    async def _get_source(args: dict[str, Any]) -> ToolResult:
        atom = str(args["atom"])
        version = str(args["version"])
        try:
            source = api.catalog.get_source_at(atom, version, root)
            return _json_result(source.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("catalog get_source failed: {}", exc)
            return _error(f"Failed to load source for {atom}@{version}: {exc}")

    async def _list_atoms(_args: dict[str, Any]) -> ToolResult:
        return _json_result(
            [
                {
                    "name": atom.name,
                    "tier": atom.tier,
                    "api_version": atom.api_version,
                    "current_hash": atom.current_hash,
                    "source_path": atom.source_path,
                }
                for atom in api.list_atoms()
            ]
        )

    api.register_tool(
        FunctionTool(
            name="catalog_list_versions",
            description="List validated content hashes for one atom.",
            parameters=_ATOM_PARAMS,
            fn=_list_versions,
        )
    )
    api.register_tool(
        FunctionTool(
            name="catalog_get_manifest",
            description="Return the frozen manifest for one atom version.",
            parameters=_VERSION_PARAMS,
            fn=_get_manifest,
        )
    )
    api.register_tool(
        FunctionTool(
            name="catalog_runs_for",
            description="List trace ids attributed to an exact atom fingerprint.",
            parameters=_RUNS_FOR_PARAMS,
            fn=_runs_for,
        )
    )
    api.register_tool(
        FunctionTool(
            name="catalog_get_source",
            description="Return UTF-8 source for one immutable atom version.",
            parameters=_VERSION_PARAMS,
            fn=_get_source,
        )
    )
    api.register_tool(
        FunctionTool(
            name="list_atoms",
            description="List atoms loaded in the current session.",
            parameters=_EMPTY_PARAMS,
            fn=_list_atoms,
        )
    )


def _json_result(payload: Any) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, default=str, indent=2, sort_keys=True),
            )
        ],
        extras=payload,
    )


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
