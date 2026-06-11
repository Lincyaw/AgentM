"""Atom under evolution: ``normalize_json``.

v1 — DELIBERATELY WEAK. Uses naive ``str.replace`` to swap single quotes
to double quotes and strip trailing commas. Fails on tasks 02–05 of the
eval suite (nested objects, unicode escapes, numeric coercion, arrays of
objects). The tuner's job is to evolve this into a working JSON
normalizer.

Local scenario atom (loaded via ``local: tool_normalize_json`` from the
scenario manifest); registered under the synthetic module name
``agentm._scenarios.format_fix.tool_normalize_json``.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="tool_normalize_json",
    description=(
        "Canonicalize malformed JSON. v1 is deliberately weak so the "
        "format_fix tuner has something to evolve."
    ),
    registers=("tool:normalize_json",),
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "raw": {"type": "string", "description": "Malformed JSON to normalize."},
    },
    "required": ["raw"],
    "additionalProperties": False,
}

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config

    async def _execute(args: dict[str, Any]) -> ToolResult:
        raw = str(args["raw"])
        # v1: naive str.replace. Quotes single -> double; strip trailing
        # commas before closing brackets. Won't handle nested cases,
        # unicode escapes, or numeric coercion — those are the eval
        # gaps the tuner must close.
        text = raw.replace("'", '"')
        text = text.replace(",}", "}").replace(",]", "]")
        return ToolResult(content=[TextContent(type="text", text=text)])

    api.register_tool(
        FunctionTool(
            name="normalize_json",
            description="Best-effort canonicalization of malformed JSON.",
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )
