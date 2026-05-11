"""Tool atom: agent self-removes a previously installed atom.

Symmetric counterpart to :mod:`tool_install_atom`. Wraps
:meth:`ExtensionAPI.unload_atom` so the model can ask the harness to
forget an atom it added earlier. The on-disk source at
``<cwd>/.agentm/atoms/<name>.py`` is **not** deleted — only the live
registration is dropped — so a subsequent session start will re-pick
up the atom via the user-atom auto-discovery layer. Deleting the file
is a separate ``write``/``bash`` operation the model can chain when
the intent is permanent removal.

Refusals (per ``unload_atom`` contract):

- atom name not currently loaded
- atom is the active provider (would leave the loop without a stream_fn)
- atom source is in the constitution layer
"""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_unload_atom",
    description=(
        "Expose ExtensionAPI.unload_atom as a tool. Drops a loaded "
        "atom's handlers/tools/commands from the running session. "
        "Source file on disk is preserved; rerunning the session "
        "re-loads it unless the file is deleted separately."
    ),
    registers=("tool:unload_atom",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=(),
)


_PARAMETERS: Final = {
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config

    async def _execute(args: dict[str, Any]) -> ToolResult:
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
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"meta_op": "unload_atom"},
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
