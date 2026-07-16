"""Audited self-modification tools for atom lifecycle and rollback."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    InstallAtomResult,
    ReloadResult,
    TextContent,
    ToolResult,
    UnloadAtomResult,
)
from agentm.extensions import ExtensionManifest

from ._root import resolve_root


class ToolCatalogMutateConfig(BaseModel):
    root: str | None = None


MANIFEST = ExtensionManifest(
    name="tool_catalog_mutate",
    description="Install, unload, reload, and roll back live atoms.",
    registers=(
        "tool:rollback_atom",
        "tool:install_atom",
        "tool:unload_atom",
        "tool:reload_atom",
    ),
    config_schema=ToolCatalogMutateConfig,
    api_version=1,
    tier=1,
)

_ROLLBACK_ATOM_PARAMS: Final = {
    "type": "object",
    "properties": {
        "atom": {
            "type": "string",
            "description": "Loaded atom name.",
        },
        "version": {
            "type": "string",
            "description": "Catalog content hash to restore.",
        },
        "rationale": {
            "type": "string",
            "description": "Why the rollback is being requested.",
        },
        "force": {
            "type": "boolean",
            "default": False,
            "description": "Override a regressed decision for the target version.",
        },
    },
    "required": ["atom", "version", "rationale"],
    "additionalProperties": False,
}

_INSTALL_ATOM_PARAMS: Final = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Atom name; must equal MANIFEST.name.",
        },
        "source": {
            "type": "string",
            "description": (
                "Full Python source defining MANIFEST: ExtensionManifest "
                "and install(api, config). Tier must be 1."
            ),
        },
        "config": {
            "type": "object",
            "description": "Config passed to install(api, config).",
        },
        "target_path": {
            "type": "string",
            "description": (
                "Optional workspace path. Defaults to .agentm/atoms/<name>.py."
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
        "name": {"type": "string", "description": "Loaded atom name."},
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
        "name": {"type": "string", "description": "Loaded atom name."},
        "source": {
            "type": "string",
            "description": (
                "Full replacement source. Validation, write verification, "
                "activation, catalog freeze, and rollback are transactional."
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


def install(api: ExtensionAPI, config: ToolCatalogMutateConfig) -> None:
    root = resolve_root(api, config.root)

    async def _rollback_atom(args: dict[str, Any]) -> ToolResult:
        atom = str(args["atom"])
        version = str(args["version"])
        rationale = str(args["rationale"])
        force = bool(args.get("force", False))

        prior_decision = _regression_decision(atom, version, root=root)
        if prior_decision is not None and not force:
            _append_decision(
                atom,
                version,
                {
                    "at": _now_iso(),
                    "kind": "rollback_blocked",
                    "by": "tool_catalog",
                    "rationale": (
                        "rollback refused because target version is marked regressed"
                    ),
                    "requested_rationale": rationale,
                    "target_version": version,
                },
                root=root,
            )
            return _json_error(
                {
                    "error": "rollback_blocked",
                    "atom": atom,
                    "version": version,
                    "reason": (
                        "target version is marked regressed; "
                        "pass force=true to override"
                    ),
                    "decision": prior_decision,
                }
            )

        try:
            source = api.catalog.get_source_at(atom, version, root).decode("utf-8")
            result = api.reload_atom(
                atom,
                source,
                rationale=f"rollback to {version}: {rationale}",
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("catalog rollback failed: {}", exc)
            return _error(f"Rollback failed for {atom}@{version}: {exc}")

        payload = _serialize_result(result)
        if not result.ok:
            return _json_error(payload)
        if result.new_hash != version:
            return _json_error(
                {
                    "error": "rollback_identity_mismatch",
                    "atom": atom,
                    "requested_version": version,
                    "activated_version": result.new_hash,
                }
            )
        _append_decision(
            atom,
            version,
            {
                "at": _now_iso(),
                "kind": "rollback",
                "by": "tool_catalog",
                "rationale": rationale,
                "forced": force,
                "target_version": version,
            },
            root=root,
        )
        return _json_result(payload)

    async def _install_atom(args: dict[str, Any]) -> ToolResult:
        result = api.install_atom(
            name=str(args["name"]),
            source=str(args["source"]),
            target_path=(
                str(args["target_path"]) if "target_path" in args else None
            ),
            config=dict(args.get("config") or {}),
            rationale=(str(args["rationale"]) if "rationale" in args else None),
            agent_initiated=True,
        )
        return _result_or_error(result)

    async def _unload_atom(args: dict[str, Any]) -> ToolResult:
        result = api.unload_atom(str(args["name"]), agent_initiated=True)
        return _result_or_error(result)

    async def _reload_atom(args: dict[str, Any]) -> ToolResult:
        result = api.reload_atom(
            str(args["name"]),
            str(args["source"]),
            rationale=(str(args["rationale"]) if "rationale" in args else None),
            agent_initiated=True,
        )
        return _result_or_error(result)

    api.register_tool(
        FunctionTool(
            name="rollback_atom",
            description=(
                "Reload a live atom from an immutable catalog snapshot. "
                "The activated content hash must equal the requested version."
            ),
            parameters=_ROLLBACK_ATOM_PARAMS,
            fn=_rollback_atom,
        )
    )
    api.register_tool(
        FunctionTool(
            name="install_atom",
            description="Install and freeze a new tier-1 atom transactionally.",
            parameters=_INSTALL_ATOM_PARAMS,
            fn=_install_atom,
        )
    )
    api.register_tool(
        FunctionTool(
            name="unload_atom",
            description="Unload a non-provider, non-constitution atom.",
            parameters=_UNLOAD_ATOM_PARAMS,
            fn=_unload_atom,
        )
    )
    api.register_tool(
        FunctionTool(
            name="reload_atom",
            description=(
                "Validate, write, activate, and freeze replacement atom source "
                "transactionally."
            ),
            parameters=_RELOAD_ATOM_PARAMS,
            fn=_reload_atom,
        )
    )


def _decisions_path(atom: str, version: str, root: Path) -> Path:
    from agentm.core.abi import atom_decisions_path

    return atom_decisions_path(atom, version, root=root)


def _regression_decision(
    atom: str,
    version: str,
    *,
    root: Path,
) -> dict[str, Any] | None:
    path = _decisions_path(atom, version, root)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("regressed") is True or payload.get("kind") == "regressed":
            return payload
    return None


def _append_decision(
    atom: str,
    version: str,
    record: dict[str, Any],
    *,
    root: Path,
) -> None:
    path = _decisions_path(atom, version, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _serialize_result(
    payload: ReloadResult | InstallAtomResult | UnloadAtomResult,
) -> dict[str, Any]:
    return asdict(payload)


def _result_or_error(
    result: ReloadResult | InstallAtomResult | UnloadAtomResult,
) -> ToolResult:
    payload = _serialize_result(result)
    return _json_result(payload) if result.ok else _json_error(payload)


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
