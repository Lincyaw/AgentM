"""Git-backed catalog browser and rollback tools for the agent."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import AtomInfo, ExtensionAPI, ReloadResult
from agentm.harness.resource_writer import WriteResult

MANIFEST = ExtensionManifest(
    name="tool_catalog",
    description=(
        "Browse git-backed resource history, inspect manifests/source at prior "
        "commits, and roll managed resources back to an earlier version."
    ),
    registers=(
        "tool:catalog_list_versions",
        "tool:catalog_get_manifest",
        "tool:catalog_runs_for",
        "tool:get_source_at",
        "tool:list_history",
        "tool:rollback_resource",
        "tool:install_atom",
        "tool:unload_atom",
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
            "description": "Atom name or repo-relative path.",
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
            "description": "Git commit SHA.",
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

_GET_SOURCE_AT_PARAMS = {
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

_LIST_HISTORY_PARAMS = {
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

_ROLLBACK_RESOURCE_PARAMS = {
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

_INSTALL_ATOM_PARAMS = {
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

_UNLOAD_ATOM_PARAMS = {
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    raw_root = config.get("root")
    if raw_root is None:
        root = Path(api.cwd)
    else:
        root = Path(str(raw_root))
        if not root.is_absolute():
            root = Path(api.cwd) / root
    root = root.resolve()

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

    async def _get_source_at_tool(args: dict[str, Any]) -> ToolResult:
        resolved = _resolve_catalog_path(api, str(args["path"]), root)
        sha = str(args["sha"])
        try:
            source = api.catalog.get_source_at(resolved.git_path, sha, root)
            return _json_result(source.decode("utf-8"))
        except Exception as exc:
            return _error(f"Failed to load source for {resolved.git_path}@{sha}: {exc}")

    async def _list_history_tool(args: dict[str, Any]) -> ToolResult:
        resolved = _resolve_catalog_path(api, str(args["path"]), root)
        limit = int(args.get("limit", 20))
        try:
            history = _list_history(resolved.git_path, limit=limit, root=root)
            return _json_result(history)
        except Exception as exc:
            return _error(f"Failed to list history for {resolved.git_path}: {exc}")

    async def _rollback_resource_tool(args: dict[str, Any]) -> ToolResult:
        resolved = _resolve_catalog_path(api, str(args["path"]), root)
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
            return _error(f"Failed to load rollback source for {resolved.git_path}@{target_sha}: {exc}")

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
            target_path=(
                str(args["target_path"]) if "target_path" in args else None
            ),
            config=dict(args.get("config") or {}),
            rationale=(
                str(args["rationale"]) if "rationale" in args else None
            ),
            agent_initiated=True,
        )
        payload = _serialize_result(result)
        if not bool(payload.get("ok", False)):
            return _json_error(payload)
        return _json_result(payload)

    async def _unload_atom_tool(args: dict[str, Any]) -> ToolResult:
        result = api.unload_atom(
            str(args["name"]),
            rationale=(
                str(args["rationale"]) if "rationale" in args else None
            ),
            agent_initiated=True,
        )
        payload = _serialize_result(result)
        if not bool(payload.get("ok", False)):
            return _json_error(payload)
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
                "install(api, config) and pass the §11 single-file contract. "
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


class _ResolvedCatalogPath:
    def __init__(
        self,
        *,
        atom_name: str | None,
        display_path: str,
        git_path: str,
        writer_path: str,
    ) -> None:
        self.atom_name = atom_name
        self.display_path = display_path
        self.git_path = git_path
        self.writer_path = writer_path


def _resolve_catalog_path(
    api: ExtensionAPI,
    raw_path: str,
    root: Path,
) -> _ResolvedCatalogPath:
    atoms = api.list_atoms()
    atom = _atom_by_name(atoms, raw_path)
    if atom is not None and atom.source_path is not None:
        source_path = Path(atom.source_path).resolve()
        git_path = PurePosixPath(source_path.relative_to(root)).as_posix()
        return _ResolvedCatalogPath(
            atom_name=atom.name,
            display_path=raw_path,
            git_path=git_path,
            writer_path=str(source_path),
        )

    # Path-form input: classify and try to map back to an atom so the
    # rollback goes through reload_atom (transactional + emits
    # ExtensionReloadEvent) instead of falling through to plain
    # writer.write. The tool's own schema promises that
    # "atom name, repo-relative path, or absolute path" all work.
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path {raw_path!r} is outside repo root {root}") from exc
        git_path = PurePosixPath(relative).as_posix()
        writer_path = str(resolved)
    else:
        resolved = (root / candidate).resolve()
        git_path = PurePosixPath(candidate).as_posix()
        writer_path = raw_path

    matched = _atom_by_source_path(atoms, resolved)
    if matched is not None:
        return _ResolvedCatalogPath(
            atom_name=matched.name,
            display_path=raw_path,
            git_path=git_path,
            writer_path=str(Path(matched.source_path).resolve())
            if matched.source_path is not None
            else writer_path,
        )

    return _ResolvedCatalogPath(
        atom_name=None,
        display_path=raw_path,
        git_path=git_path,
        writer_path=writer_path,
    )


def _atom_by_name(atoms: list[AtomInfo], name: str) -> AtomInfo | None:
    for atom in atoms:
        if atom.name == name:
            return atom
    return None


def _atom_by_source_path(atoms: list[AtomInfo], resolved: Path) -> AtomInfo | None:
    for atom in atoms:
        if atom.source_path is None:
            continue
        try:
            if Path(atom.source_path).resolve() == resolved:
                return atom
        except OSError:
            continue
    return None


def _list_history(path: str, *, limit: int, root: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "git",
            "log",
            "--format=%H%x00%an%x00%aI%x00%s",
            "-n",
            str(limit),
            "--",
            path,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "<no output>"
        raise RuntimeError(stderr)
    out: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line:
            continue
        sha, author, timestamp, message = line.split("\x00", 3)
        out.append(
            {
                "sha": sha,
                "author": author,
                "timestamp": timestamp,
                "message": message,
            }
        )
    return out


def _regression_decision(
    api: ExtensionAPI,
    atom_name: str | None,
    target_sha: str,
    *,
    root: Path,
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
    root: Path,
) -> None:
    if atom_name is None:
        return
    api.catalog.append_atom_decision(atom_name, target_sha, record, root)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _serialize_result(payload: Any) -> dict[str, Any]:
    if isinstance(payload, (ReloadResult, WriteResult)):
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
        details=payload,
    )


def _json_error(payload: Any) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, default=str, indent=2, sort_keys=True),
            )
        ],
        details=payload,
        is_error=True,
    )


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
