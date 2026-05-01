"""RCA hypothesis-management tool atoms."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from agentm_rca.stores import HypothesisStore

MANIFEST = ExtensionManifest(
    name="hypothesis_tools",
    description="Register RCA hypothesis store tool atoms.",
    registers=("tool:update_hypothesis", "tool:remove_hypothesis"),
)

_UPDATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "description": {"type": "string"},
        "status": {
            "type": "string",
            "enum": [
                "formed",
                "investigating",
                "confirmed",
                "rejected",
                "refined",
                "inconclusive",
            ],
            "default": "formed",
        },
        "evidence_summary": {"type": ["string", "null"]},
        "parent_id": {"type": ["string", "null"]},
    },
    "required": ["id", "description"],
    "additionalProperties": False,
}

_REMOVE_PARAMETERS = {
    "type": "object",
    "properties": {"id": {"type": "string"}},
    "required": ["id"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    store = _expect_store(config)

    async def _update(args: dict[str, Any]) -> ToolResult:
        entry = store.update(
            id=str(args["id"]),
            description=str(args["description"]),
            status=str(args.get("status", "formed")),
            evidence_summary=_maybe_str(args.get("evidence_summary")),
            parent_id=_maybe_str(args.get("parent_id")),
        )
        api.session.append_entry("hypothesis", asdict(entry))
        text = f"Hypothesis {entry.id} updated: {entry.status} -- {entry.description}"
        return _ok(text, details=asdict(entry))

    async def _remove(args: dict[str, Any]) -> ToolResult:
        hypothesis_id = str(args["id"])
        existed = store.remove(hypothesis_id)
        if existed:
            api.session.append_entry(
                "hypothesis",
                {"id": hypothesis_id, "status": "removed"},
            )
        return _ok(
            f"Hypothesis {hypothesis_id} removed"
            if existed
            else f"Hypothesis {hypothesis_id} not found"
        )

    api.register_tool(
        FunctionTool(
            name="update_hypothesis",
            description="Create or update a hypothesis in the RCA investigation.",
            parameters=_UPDATE_PARAMETERS,
            fn=_update,
        )
    )
    api.register_tool(
        FunctionTool(
            name="remove_hypothesis",
            description="Remove a hypothesis from the RCA investigation.",
            parameters=_REMOVE_PARAMETERS,
            fn=_remove,
        )
    )


def _expect_store(config: dict[str, Any]) -> HypothesisStore:
    store = config.get("store")
    if not isinstance(store, HypothesisStore):
        raise TypeError("hypothesis_tools.install requires config['store']=HypothesisStore")
    return store


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _ok(text: str, *, details: Any = None) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        details=details,
    )


def format_hypothesis_payload(payload: Any) -> str:
    return json.dumps(payload, default=str, indent=2, sort_keys=True)
