"""Tool atom for the ``extensions.builtin.tool_hypothesis_store`` §7.1 row."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


_VALID_STATUSES = frozenset(
    {"formed", "investigating", "confirmed", "rejected", "refined", "inconclusive"}
)


@dataclass(slots=True)
class Hypothesis:
    id: str
    description: str
    status: str = "formed"
    evidence: list[str] | None = None
    parent_id: str | None = None


MANIFEST = ExtensionManifest(
    name="tool_hypothesis_store",
    description="Register in-memory RCA hypothesis store tools.",
    registers=(
        "tool:add_hypothesis",
        "tool:update_hypothesis",
        "tool:list_hypotheses",
    ),
    config_schema=None,
)

_ADD_PARAMETERS = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "description": {"type": "string"},
        "status": {"type": "string", "default": "formed"},
        "evidence_summary": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
    },
    "required": ["id", "description"],
    "additionalProperties": False,
}

_UPDATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "description": {"type": "string"},
        "status": {"type": "string"},
        "evidence_summary": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
    },
    "required": ["id"],
    "additionalProperties": False,
}

_LIST_PARAMETERS = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    store: dict[str, Hypothesis] = {}

    async def _add(args: dict[str, Any]) -> ToolResult:
        try:
            hypothesis = _build_new(args)
            store[hypothesis.id] = hypothesis
            api.session.append_entry("hypothesis", asdict(hypothesis))
            return _json_result(asdict(hypothesis))
        except Exception as exc:
            return _error(f"Failed to add hypothesis: {exc}")

    async def _update(args: dict[str, Any]) -> ToolResult:
        hypothesis_id = str(args["id"])
        try:
            existing = store.get(hypothesis_id)
            if existing is None:
                return _error(f"Unknown hypothesis: {hypothesis_id}")
            updated = _apply_update(existing, args)
            store[hypothesis_id] = updated
            api.session.append_entry("hypothesis", asdict(updated))
            return _json_result(asdict(updated))
        except Exception as exc:
            return _error(f"Failed to update hypothesis {hypothesis_id!r}: {exc}")

    async def _list(_args: dict[str, Any]) -> ToolResult:
        items = [asdict(store[key]) for key in sorted(store)]
        return _json_result(items)

    api.register_tool(
        FunctionTool(
            name="add_hypothesis",
            description="Create a tracked RCA hypothesis.",
            parameters=_ADD_PARAMETERS,
            fn=_add,
        )
    )
    api.register_tool(
        FunctionTool(
            name="update_hypothesis",
            description="Update a tracked RCA hypothesis.",
            parameters=_UPDATE_PARAMETERS,
            fn=_update,
        )
    )
    api.register_tool(
        FunctionTool(
            name="list_hypotheses",
            description="List all tracked RCA hypotheses.",
            parameters=_LIST_PARAMETERS,
            fn=_list,
        )
    )


def _build_new(args: dict[str, Any]) -> Hypothesis:
    status = str(args.get("status", "formed"))
    _validate_status(status)
    evidence_summary = args.get("evidence_summary")
    evidence = [str(evidence_summary)] if evidence_summary else []
    return Hypothesis(
        id=str(args["id"]),
        description=str(args["description"]),
        status=status,
        evidence=evidence,
        parent_id=_maybe_str(args.get("parent_id")),
    )


def _apply_update(existing: Hypothesis, args: dict[str, Any]) -> Hypothesis:
    status = str(args.get("status", existing.status))
    _validate_status(status)
    evidence = list(existing.evidence or [])
    evidence_summary = args.get("evidence_summary")
    if evidence_summary:
        evidence.append(str(evidence_summary))
    return Hypothesis(
        id=existing.id,
        description=str(args.get("description", existing.description)),
        status=status,
        evidence=evidence,
        parent_id=_maybe_str(args.get("parent_id", existing.parent_id)),
    )


def _validate_status(status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _json_result(payload: Any) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, default=str, indent=2, sort_keys=True),
            )
        ]
    )


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
