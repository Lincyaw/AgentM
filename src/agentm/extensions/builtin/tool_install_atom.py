"""Tool atom: agent self-install of new atoms.

Lightweight wrapper around :meth:`ExtensionAPI.install_atom`. The
existing ``tool_propose_change`` is the heavy, evidence-driven path
(eval baseline + proposed, promotion gate, tier checks); this atom is
the *light* path: agent decides it needs a new helper, hands the SDK a
single Python module's source, the source lands at
``<cwd>/.agentm/atoms/<name>.py`` (default :meth:`install_atom` target),
gets registered live, and persists across process restarts because the
scenario loader now merges that directory on every session start.

§11 single-file contract enforcement happens *inside* ``install_atom``
— invalid sources surface as ``ok=False`` with the validator's error
verbatim. Constitution-protected paths and tier-2 atoms are also
rejected there. This atom is intentionally a thin pass-through; it adds
no policy of its own except converting the structured result into a
human-readable ``ToolResult``.

Pair with ``tool_unload_atom`` for the symmetric remove operation.
"""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_install_atom",
    description=(
        "Expose ExtensionAPI.install_atom as a tool the model can call. "
        "Writes a new atom's source to <cwd>/.agentm/atoms/<name>.py, "
        "validates §11, and registers handlers/tools live. Persists "
        "across restarts via the user-atom auto-discovery layer."
    ),
    registers=("tool:install_atom",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=(),
)


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
                "no agentm.core._internal, no agentm.harness.session."
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config

    async def _execute(args: dict[str, Any]) -> ToolResult:
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
            fn=_execute,
            metadata={"meta_op": "install_atom"},
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
