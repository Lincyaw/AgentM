"""Read-only git-backed catalog browsing tools for the agent."""

from __future__ import annotations

import json
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

from ._git_log import list_history
from ._paths import catalog_root, resolve_catalog_path

MANIFEST = ExtensionManifest(
    name="tool_catalog_browse",
    description="Browse git-backed resource history and loaded atom metadata.",
    registers=(
        "tool:catalog_list_versions",
        "tool:catalog_get_manifest",
        "tool:catalog_runs_for",
        "tool:get_source_at",
        "tool:list_history",
        "tool:list_atoms",
    ),
    config_schema=None,
    api_version=1,
    affects=(),
    tier=1,
)

_LIST_VERSIONS_PARAMS: Final = {
    "type": "object",
    "properties": {
        "atom": {
            "type": "string",
            "description": "Atom name or repo-relative path.",
        }
    },
    "required": ["atom"],
    "additionalProperties": False,
}

_GET_MANIFEST_PARAMS: Final = {
    "type": "object",
    "properties": {
        "atom": {"type": "string"},
        "version": {
            "type": "string",
            "description": "Git commit SHA.",
        },
    },
    "required": ["atom", "version"],
    "additionalProperties": False,
}

_RUNS_FOR_PARAMS: Final = {
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

_GET_SOURCE_AT_PARAMS: Final = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Atom name, repo-relative path, or absolute path inside the repo.",
        },
        "sha": {
            "type": "string",
            "description": "Git commit SHA to read from.",
        },
    },
    "required": ["path", "sha"],
    "additionalProperties": False,
}

_LIST_HISTORY_PARAMS: Final = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Atom name, repo-relative path, or absolute path inside the repo.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "default": 20,
            "description": "Maximum number of commits to return, newest first.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}

_LIST_ATOMS_PARAMS: Final = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    root = catalog_root(api, config)

    async def _list_versions_tool(args: dict[str, Any]) -> ToolResult:
        versions = api.catalog.list_versions(str(args["atom"]), root)
        return _json_result(versions)

    async def _get_manifest_tool(args: dict[str, Any]) -> ToolResult:
        atom = str(args["atom"])
        version = str(args["version"])
        try:
            return _json_result(api.catalog.get_manifest_at(atom, version, root))
        except Exception as exc:
            return _error(f"Failed to load manifest for {atom}@{version}: {exc}")

    async def _runs_for_tool(args: dict[str, Any]) -> ToolResult:
        try:
            trace_ids = api.catalog.runs_for(args["fingerprint"], root)
            return _json_result(trace_ids)
        except Exception as exc:
            return _error(f"Failed to resolve catalog runs: {exc}")

    async def _get_source_at_tool(args: dict[str, Any]) -> ToolResult:
        resolved = resolve_catalog_path(api, str(args["path"]), root)
        sha = str(args["sha"])
        try:
            source = api.catalog.get_source_at(resolved.git_path, sha, root)
            return _json_result(source.decode("utf-8"))
        except Exception as exc:
            return _error(f"Failed to load source for {resolved.git_path}@{sha}: {exc}")

    async def _list_history_tool(args: dict[str, Any]) -> ToolResult:
        resolved = resolve_catalog_path(api, str(args["path"]), root)
        limit = int(args.get("limit", 20))
        try:
            history = list_history(resolved.git_path, limit=limit, root=root)
            return _json_result(history)
        except Exception as exc:
            return _error(f"Failed to list history for {resolved.git_path}: {exc}")

    async def _list_atoms_tool(args: dict[str, Any]) -> ToolResult:
        atoms = api.list_atoms()
        payload = [
            {
                "name": a.name,
                "tier": a.tier,
                "api_version": a.api_version,
                "current_hash": a.current_hash,
                "source_path": a.source_path,
            }
            for a in atoms
        ]
        return _json_result(payload)

    api.register_tool(
        FunctionTool(
            name="catalog_list_versions",
            description="List known git versions for one managed atom or path.",
            parameters=_LIST_VERSIONS_PARAMS,
            fn=_list_versions_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="catalog_get_manifest",
            description="AST-parse the historical MANIFEST payload for one atom at a commit.",
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
    api.register_tool(
        FunctionTool(
            name="get_source_at",
            description="Return UTF-8 source text for a managed path at an arbitrary git SHA.",
            parameters=_GET_SOURCE_AT_PARAMS,
            fn=_get_source_at_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="list_history",
            description="Return recent git history entries {sha, author, timestamp, message} for one managed path.",
            parameters=_LIST_HISTORY_PARAMS,
            fn=_list_history_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="list_atoms",
            description=(
                "List every atom currently loaded in this running session "
                "with its name, tier, api_version, current git hash (when "
                "managed), and on-disk source path."
            ),
            parameters=_LIST_ATOMS_PARAMS,
            fn=_list_atoms_tool,
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
