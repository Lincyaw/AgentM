"""Verifier-scenario atom: serve per-fault-kind documentation files.

Each known ``fault_kind`` has a markdown file under
``contrib/scenarios/verifier/fault_kinds/<kind>.md`` describing how the
injection works, what signals it produces, and how it tends to
propagate. The verifier reads only the doc for the actual fault kind
of the current case — progressive disclosure rather than dumping all
26 entries into the system prompt.

Exposes one tool, ``get_fault_kind_doc(fault_kind)``, which returns
the file body as plain text. Unknown kinds return an error result;
the agent should fall back to fault-agnostic reasoning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.core.abi import FunctionTool, ToolResult
from agentm.core.abi.messages import TextContent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest


_DOCS_DIR = Path(__file__).resolve().parent / "fault_kinds"


MANIFEST = ExtensionManifest(
    name="verifier_fault_docs",
    description=(
        "Expose get_fault_kind_doc() — on-demand lookup of the per-"
        "fault-kind reference (mechanism / signatures / propagation). "
        "The agent calls it after get_injection_spec returns fault_kind."
    ),
    registers=("tool:get_fault_kind_doc",),
)


def _available_kinds() -> list[str]:
    if not _DOCS_DIR.is_dir():
        return []
    return sorted(p.stem for p in _DOCS_DIR.glob("*.md"))


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    async def _get_doc(args: dict[str, Any]) -> ToolResult:
        kind = (args.get("fault_kind") or "").strip()
        kinds = _available_kinds()
        if not kind or kind not in kinds:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "unknown_fault_kind",
                                "requested": kind,
                                "available": kinds,
                            }
                        ),
                    )
                ],
                is_error=True,
            )
        body = (_DOCS_DIR / f"{kind}.md").read_text()
        return ToolResult(content=[TextContent(type="text", text=body)])

    api.register_tool(
        FunctionTool(
            name="get_fault_kind_doc",
            description=(
                "Return the reference doc for a given fault_kind: how "
                "the injection works, what signals it produces, and how "
                "it tends to propagate. Call after get_injection_spec "
                "returns the fault_kind for this case."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fault_kind": {
                        "type": "string",
                        "description": (
                            "Canonical fault_kind string (e.g., "
                            "'pod_failure', 'network_loss', 'http_slow')."
                        ),
                    },
                },
                "required": ["fault_kind"],
                "additionalProperties": False,
            },
            fn=_get_doc,
        )
    )


__all__ = ["MANIFEST", "install"]
