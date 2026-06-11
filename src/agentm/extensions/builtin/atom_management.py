"""Grouped atom-management tool atom: ``install_atom``, ``unload_atom``,
and ``list_atoms``.

Merges the former single-tool atoms ``tool_install_atom``,
``tool_unload_atom``, and ``tool_list_atoms`` into one §11-compliant
module. The LLM-facing tool names are unchanged.

install_atom — lightweight wrapper around :meth:`ExtensionAPI.install_atom`.
The agent hands the SDK a single Python module's source; it lands at
``<cwd>/.agentm/atoms/<name>.py``, gets registered live, and persists
across process restarts.

unload_atom — wraps :meth:`ExtensionAPI.unload_atom`. Drops a loaded
atom's handlers/tools/commands from the running session. On-disk source
is preserved.

list_atoms — wraps :meth:`ExtensionAPI.list_atoms`. Returns one line per
loaded atom with name, tier, and source path.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest

# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

class AtomManagementConfig(BaseModel):
    pass

MANIFEST = ExtensionManifest(
    name="atom_management",
    description=(
        "Register the install_atom, unload_atom, and list_atoms tools "
        "for agent self-modification of the atom set."
    ),
    registers=("tool:install_atom", "tool:unload_atom", "tool:list_atoms"),
    config_schema=AtomManagementConfig,
    requires=(),
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])

def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)

# ---------------------------------------------------------------------------
# Tool parameter schemas
# ---------------------------------------------------------------------------

_INSTALL_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Atom name; must match the source's MANIFEST.name. "
                "Becomes the file stem at <cwd>/.agentm/atoms/<name>.py."
            ),
        },
        "source": {
            "type": "string",
            "description": (
                "Full Python module text. Must contain a top-level "
                "``MANIFEST = ExtensionManifest(...)`` and an "
                "``install(api, config)`` function. §11 single-file "
                "contract applies — no imports of other atom modules, "
                "no agentm.core._internal, no agentm.core.runtime.session."
            ),
        },
        "rationale": {
            "type": "string",
            "description": (
                "Why this atom is being installed. Surfaces in "
                "observability so future operators can audit "
                "agent-initiated installs."
            ),
        },
        "config": {
            "type": "object",
            "description": (
                "Optional config dict passed to the atom's install() "
                "call. Defaults to {} when omitted."
            ),
            "additionalProperties": True,
        },
    },
    "required": ["name", "source", "rationale"],
    "additionalProperties": False,
}

_UNLOAD_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Atom name as registered (matches MANIFEST.name).",
        },
        "rationale": {
            "type": "string",
            "description": "Why this atom is being removed.",
        },
    },
    "required": ["name", "rationale"],
    "additionalProperties": False,
}

_LIST_PARAMETERS: Final = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: AtomManagementConfig) -> None:
    del config

    # --- install_atom tool ------------------------------------------------

    async def _install_execute(args: dict[str, Any]) -> ToolResult:
        name = str(args["name"])
        source = str(args["source"])
        rationale = str(args["rationale"])
        atom_config = args.get("config")
        if atom_config is not None and not isinstance(atom_config, dict):
            return _error("`config` must be an object/mapping when provided.")
        try:
            result = api.install_atom(
                name=name,
                source=source,
                target_path=None,  # default: <cwd>/.agentm/atoms/<name>.py
                config=atom_config,
                rationale=rationale,
                agent_initiated=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface to the model.
            return _error(f"install_atom raised: {exc}")
        if not result.ok:
            return _error(
                f"install_atom rejected '{name}': {result.error or 'unknown error'}"
            )
        path = getattr(result, "target_path", None) or "(default)"
        new_hash = getattr(result, "new_hash", None)
        return _ok(
            f"Installed atom '{name}' at {path}. "
            f"new_hash={new_hash}. Active in this session and persisted "
            "for future sessions via <cwd>/.agentm/atoms/."
        )

    api.register_tool(
        FunctionTool(
            name="install_atom",
            description=(
                "Install a new agent-authored atom. Source must be a "
                "single §11-compliant Python module with a top-level "
                "MANIFEST and install() function. The atom is registered "
                "in the live session and on disk so it survives restarts."
            ),
            parameters=_INSTALL_PARAMETERS,
            fn=_install_execute,
            metadata={"meta_op": "install_atom"},
        )
    )

    # --- unload_atom tool -------------------------------------------------

    async def _unload_execute(args: dict[str, Any]) -> ToolResult:
        name = str(args["name"])
        # ``rationale`` is required by the tool schema for human/audit clarity
        # but the kernel does not currently consume it; the trajectory captures
        # the args verbatim.
        _ = str(args["rationale"])
        try:
            result = api.unload_atom(name, agent_initiated=True)
        except Exception as exc:  # noqa: BLE001
            return _error(f"unload_atom raised: {exc}")
        if not result.ok:
            return _error(
                f"unload_atom refused to drop '{name}': "
                f"{result.error or 'unknown error'}"
            )
        return _ok(
            f"Unloaded atom '{name}' from the running session. "
            "On-disk source is untouched; delete the file separately if "
            "you want it removed permanently."
        )

    api.register_tool(
        FunctionTool(
            name="unload_atom",
            description=(
                "Remove a loaded atom from the running session. "
                "Reverses install_atom's live registration but does not "
                "delete the source file."
            ),
            parameters=_UNLOAD_PARAMETERS,
            fn=_unload_execute,
            metadata={"meta_op": "unload_atom"},
        )
    )

    # --- list_atoms tool --------------------------------------------------

    async def _list_execute(args: dict[str, Any]) -> ToolResult:
        del args
        try:
            atoms = api.list_atoms()
        except Exception as exc:  # noqa: BLE001
            return _error(f"list_atoms raised: {exc}")
        if not atoms:
            return _ok("(no atoms loaded)")
        lines = [f"{len(atoms)} atom(s) loaded:"]
        for entry in sorted(atoms, key=lambda a: a.name):
            source = entry.source_path or "(builtin)"
            lines.append(f"  {entry.name}  tier={entry.tier}  src={source}")
        return _ok("\n".join(lines))

    api.register_tool(
        FunctionTool(
            name="list_atoms",
            description=(
                "List every atom currently registered in this session, "
                "with name, tier, and source path. Useful before install "
                "or unload to avoid collisions and confirm state."
            ),
            parameters=_LIST_PARAMETERS,
            fn=_list_execute,
            metadata={"meta_op": "list_atoms"},
        )
    )
