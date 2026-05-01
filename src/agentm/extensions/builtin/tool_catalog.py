"""Read-only browser of `.agentm/catalog` for the agent.

Tier-1 MVP scope: browse versions, manifests, and recorded runs. This atom
intentionally does not expose `compare()` or `propose_change()`; those land in
Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="tool_catalog",
    description=(
        "Browse catalog versions, manifests, and recorded runs. MVP is read-only; "
        "compare()/propose_change() are deferred to Phase 2."
    ),
    registers=(
        "tool:catalog_list_versions",
        "tool:catalog_get_manifest",
        "tool:catalog_runs_for",
    ),
    config_schema={
        "type": "object",
        "properties": {"root": {"type": "string"}},
        "additionalProperties": False,
    },
    api_version=1,
    affects=(),
    tier=1,
)

_LIST_VERSIONS_PARAMS = {
    "type": "object",
    "properties": {
        "atom": {
            "type": "string",
            "description": "Atom name, for example 'tool_read'.",
        }
    },
    "required": ["atom"],
    "additionalProperties": False,
}

_GET_MANIFEST_PARAMS = {
    "type": "object",
    "properties": {
        "atom": {"type": "string"},
        "version": {
            "type": "string",
            "description": "Catalog version hash.",
        },
    },
    "required": ["atom", "version"],
    "additionalProperties": False,
}

_RUNS_FOR_PARAMS = {
    "type": "object",
    "properties": {
        "fingerprint": {
            "description": "Exact atom-set fingerprint mapping or 'atom@version' string.",
            "oneOf": [{"type": "object"}, {"type": "string"}],
        }
    },
    "required": ["fingerprint"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # ``root`` is the *project* root that contains ``.agentm/catalog/``.
    # ``agentm.core._internal.catalog._layout.catalog_root`` prepends that segment, so
    # callers must not include it here. Defaults to the session cwd.
    raw_root = config.get("root")
    if raw_root is None:
        root = Path(api.cwd)
    else:
        root = Path(str(raw_root))
        if not root.is_absolute():
            root = Path(api.cwd) / root

    async def _list_versions_tool(args: dict[str, Any]) -> ToolResult:
        versions = api.catalog.list_versions(str(args["atom"]), root)
        return _json_result(versions)

    async def _get_manifest_tool(args: dict[str, Any]) -> ToolResult:
        atom = str(args["atom"])
        version = str(args["version"])
        try:
            return _json_result(api.catalog.get_manifest_at(atom, version, root))
        except Exception as exc:
            return _error(
                f"Failed to load manifest for {atom}@{version}: {exc}"
            )

    async def _runs_for_tool(args: dict[str, Any]) -> ToolResult:
        try:
            trace_ids = api.catalog.runs_for(args["fingerprint"], root)
            return _json_result(trace_ids)
        except Exception as exc:
            return _error(f"Failed to resolve catalog runs: {exc}")

    api.register_tool(
        FunctionTool(
            name="catalog_list_versions",
            description="List known catalog versions for one atom.",
            parameters=_LIST_VERSIONS_PARAMS,
            fn=_list_versions_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="catalog_get_manifest",
            description="Load the full manifest for one cataloged atom version.",
            parameters=_GET_MANIFEST_PARAMS,
            fn=_get_manifest_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="catalog_runs_for",
            description="List trace ids recorded for an exact atom-set fingerprint.",
            parameters=_RUNS_FOR_PARAMS,
            fn=_runs_for_tool,
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
        details=payload,
    )


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
