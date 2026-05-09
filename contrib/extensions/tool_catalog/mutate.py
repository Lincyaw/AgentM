"""Self-modification tools for the git-backed tool catalog package."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import (
    ExtensionAPI,
    InstallAtomResult,
    ReloadResult,
    UnloadAtomResult,
)
from agentm.harness.resource_writer import WriteResult

from ._paths import catalog_root, resolve_catalog_path

MANIFEST = ExtensionManifest(
    name="tool_catalog_mutate",
    description="Mutate managed resources and live atoms through audited catalog tools.",
    registers=(
        "tool:rollback_resource",
        "tool:install_atom",
        "tool:unload_atom",
        "tool:reload_atom",
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

_ROLLBACK_RESOURCE_PARAMS: Final = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Atom name, repo-relative path, or absolute path inside the repo.",
        },
        "target_sha": {
            "type": "string",
            "description": "Historical git commit SHA whose bytes should be restored.",
        },
        "rationale": {
            "type": "string",
            "description": "Why the rollback is being requested.",
        },
        "force": {
            "type": "boolean",
            "default": False,
            "description": "Override a prior regressed decision for the target SHA.",
        },
    },
    "required": ["path", "target_sha", "rationale"],
    "additionalProperties": False,
}

_INSTALL_ATOM_PARAMS: Final = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Atom name; must equal MANIFEST.name in the source and be a "
                "valid Python identifier."
            ),
        },
        "source": {
            "type": "string",
            "description": (
                "Full Python source for the new atom. Must define "
                "MANIFEST: ExtensionManifest and install(api, config). "
                "Tier must be 1; agent-installed atoms cannot ship at tier 2."
            ),
        },
        "config": {
            "type": "object",
            "description": "Config dict passed to install(api, config).",
        },
        "target_path": {
            "type": "string",
            "description": (
                "Optional repo-relative or absolute path where the source is "
                "written. When omitted the harness writes to "
                ".agentm/atoms/<name>.py so the new atom stays isolated from "
                "the framework's builtin tree."
            ),
        },
        "rationale": {
            "type": "string",
            "description": "Why this atom is being installed.",
        },
    },
    "required": ["name", "source"],
    "additionalProperties": False,
}

_UNLOAD_ATOM_PARAMS: Final = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Bare atom name (MANIFEST.name) to unload.",
        },
        "rationale": {
            "type": "string",
            "description": "Why this atom is being unloaded.",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}

_RELOAD_ATOM_PARAMS: Final = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Bare atom name (MANIFEST.name) to replace in the running "
                "session. The atom must already be loaded."
            ),
        },
        "source": {
            "type": "string",
            "description": (
                "Full Python source that will replace the atom's current "
                "implementation. Must define MANIFEST: ExtensionManifest "
                "and install(api, config); section 11 contract is enforced. "
                "Reload is transactional: a failure rolls the session "
                "back to the prior version with no observable change."
            ),
        },
        "rationale": {
            "type": "string",
            "description": "Why this atom is being reloaded.",
        },
    },
    "required": ["name", "source"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    root = catalog_root(api, config)

    async def _rollback_resource_tool(args: dict[str, Any]) -> ToolResult:
        resolved = resolve_catalog_path(api, str(args["path"]), root)
        target_sha = str(args["target_sha"])
        rationale = str(args["rationale"])
        force = bool(args.get("force", False))

        prior_decision = _regression_decision(
            api,
            resolved.atom_name,
            target_sha,
            root=root,
        )
        if prior_decision is not None and not force:
            _append_decision(
                api,
                resolved.atom_name,
                target_sha,
                {
                    "at": _now_iso(),
                    "kind": "rollback_blocked",
                    "by": "tool_catalog",
                    "rationale": (
                        "rollback refused because target version is marked regressed"
                    ),
                    "requested_rationale": rationale,
                    "target_sha": target_sha,
                },
                root=root,
            )
            payload = {
                "error": "rollback_blocked",
                "path": resolved.display_path,
                "target_sha": target_sha,
                "reason": "target version is marked regressed; pass force=true to override",
                "decision": prior_decision,
            }
            return _json_error(payload)

        try:
            source_bytes = api.catalog.get_source_at(resolved.git_path, target_sha, root)
        except Exception as exc:
            return _error(
                f"Failed to load rollback source for {resolved.git_path}@{target_sha}: {exc}"
            )

        rollback_rationale = f"rollback to {target_sha[:8]}: {rationale}"
        try:
            result_payload: dict[str, Any]
            if resolved.atom_name is not None:
                reload_result = api.reload_atom(
                    resolved.atom_name,
                    source_bytes.decode("utf-8"),
                    rationale=rollback_rationale,
                )
                result_payload = _serialize_result(reload_result)
                if not bool(result_payload.get("ok", False)):
                    return _json_error(result_payload)
            else:
                writer = api.get_resource_writer()
                write_result = await writer.write(
                    resolved.writer_path,
                    source_bytes,
                    rationale=rollback_rationale,
                )
                result_payload = _serialize_result(write_result)
                if write_result.error is not None:
                    return _json_error(result_payload)
        except Exception as exc:
            return _error(f"Rollback failed for {resolved.display_path}: {exc}")

        _append_decision(
            api,
            resolved.atom_name,
            target_sha,
            {
                "at": _now_iso(),
                "kind": "rollback",
                "by": "tool_catalog",
                "rationale": rationale,
                "forced": force,
                "target_sha": target_sha,
            },
            root=root,
        )
        return _json_result(result_payload)

    async def _install_atom_tool(args: dict[str, Any]) -> ToolResult:
        result = api.install_atom(
            name=str(args["name"]),
            source=str(args["source"]),
            target_path=(str(args["target_path"]) if "target_path" in args else None),
            config=dict(args.get("config") or {}),
            rationale=(str(args["rationale"]) if "rationale" in args else None),
            agent_initiated=True,
        )
        payload = _serialize_result(result)
        if not bool(payload.get("ok", False)):
            return _json_error(payload)
        return _json_result(payload)

    async def _unload_atom_tool(args: dict[str, Any]) -> ToolResult:
        result = api.unload_atom(
            str(args["name"]),
            rationale=(str(args["rationale"]) if "rationale" in args else None),
            agent_initiated=True,
        )
        payload = _serialize_result(result)
        if not bool(payload.get("ok", False)):
            return _json_error(payload)
        return _json_result(payload)

    async def _reload_atom_tool(args: dict[str, Any]) -> ToolResult:
        result = api.reload_atom(
            str(args["name"]),
            str(args["source"]),
            rationale=(str(args["rationale"]) if "rationale" in args else None),
            agent_initiated=True,
        )
        payload = _serialize_result(result)
        if not bool(payload.get("ok", False)):
            return _json_error(payload)
        return _json_result(payload)

    api.register_tool(
        FunctionTool(
            name="rollback_resource",
            description=(
                "Roll a managed atom or resource back to bytes from target_sha. "
                "Returns structured ReloadResult or WriteResult details."
            ),
            parameters=_ROLLBACK_RESOURCE_PARAMS,
            fn=_rollback_resource_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="install_atom",
            description=(
                "Install a brand-new atom into the running session from a "
                "Python source string. The source must define MANIFEST and "
                "install(api, config) and pass the section 11 single-file contract. "
                "Tier-2 atoms are refused. Returns structured InstallAtomResult."
            ),
            parameters=_INSTALL_ATOM_PARAMS,
            fn=_install_atom_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="unload_atom",
            description=(
                "Remove a previously-installed atom from the running session. "
                "On-disk source and git history are kept. Provider atoms and "
                "constitution-path atoms are refused. Returns UnloadAtomResult."
            ),
            parameters=_UNLOAD_ATOM_PARAMS,
            fn=_unload_atom_tool,
        )
    )
    api.register_tool(
        FunctionTool(
            name="reload_atom",
            description=(
                "Replace a currently-loaded atom's source in the running "
                "session. The reload is transactional: the new source is "
                "validated against the section 11 contract, the module is "
                "re-executed, and ``install`` is re-invoked; on any failure "
                "the prior version is restored and the session sees no "
                "change. Use this -- not the generic ``write`` or ``edit`` "
                "tools -- to evolve an atom's behavior, otherwise the "
                "on-disk source drifts away from the live module. Returns "
                "structured ReloadResult."
            ),
            parameters=_RELOAD_ATOM_PARAMS,
            fn=_reload_atom_tool,
        )
    )


def _regression_decision(
    api: ExtensionAPI,
    atom_name: str | None,
    target_sha: str,
    *,
    root: Any,
) -> dict[str, Any] | None:
    if atom_name is None:
        return None
    for payload in api.catalog.read_atom_decisions(atom_name, target_sha, root):
        if payload.get("regressed") is True or payload.get("kind") == "regressed":
            return payload
    return None


def _append_decision(
    api: ExtensionAPI,
    atom_name: str | None,
    target_sha: str,
    record: dict[str, Any],
    *,
    root: Any,
) -> None:
    if atom_name is None:
        return
    api.catalog.append_atom_decision(atom_name, target_sha, record, root)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _serialize_result(payload: Any) -> dict[str, Any]:
    if isinstance(
        payload, (ReloadResult, WriteResult, InstallAtomResult, UnloadAtomResult)
    ):
        return asdict(payload)
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported result payload type: {type(payload).__name__}")


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


def _json_error(payload: Any) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, default=str, indent=2, sort_keys=True),
            )
        ],
        extras=payload,
        is_error=True,
    )


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
