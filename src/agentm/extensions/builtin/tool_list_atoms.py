"""Tool atom: list every atom currently registered in the session.

Wraps :meth:`ExtensionAPI.list_atoms` so the model can introspect its
own toolbelt — useful before calling ``install_atom`` (avoid name
collisions) or ``unload_atom`` (confirm the atom is actually loaded).

Each entry is rendered as one line ``name | tier | source_path`` so a
typical assistant turn doesn't blow the message budget on a JSON dump.
The model can request the full schema by piping output through other
tools (read on the source path, etc.).
"""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_list_atoms",
    description=(
        "Expose ExtensionAPI.list_atoms as a tool. Returns one line "
        "per loaded atom with name, tier, and source path."
    ),
    registers=("tool:list_atoms",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=(),
)


_PARAMETERS: Final = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config

    async def _execute(args: dict[str, Any]) -> ToolResult:
        del args
        try:
            atoms = api.list_atoms()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content=[
                    TextContent(type="text", text=f"list_atoms raised: {exc}")
                ],
                is_error=True,
            )
        if not atoms:
            return ToolResult(
                content=[TextContent(type="text", text="(no atoms loaded)")]
            )
        lines = [f"{len(atoms)} atom(s) loaded:"]
        for entry in sorted(atoms, key=lambda a: a.name):
            source = entry.source_path or "(builtin)"
            lines.append(f"  {entry.name}  tier={entry.tier}  src={source}")
        return ToolResult(
            content=[TextContent(type="text", text="\n".join(lines))]
        )

    api.register_tool(
        FunctionTool(
            name="list_atoms",
            description=(
                "List every atom currently registered in this session, "
                "with name, tier, and source path. Useful before install "
                "or unload to avoid collisions and confirm state."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"meta_op": "list_atoms"},
        )
    )
