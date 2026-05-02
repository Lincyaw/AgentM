"""Lightweight hypothesis-tracking tools for the RCA scenario.

Provides two atoms (``update_hypothesis``, ``remove_hypothesis``) backed by
an in-memory dict scoped to the extension install. The agent uses these to
keep an explicit list of working hypotheses while it investigates. The
store is intentionally simple: no persistence, no ordering guarantees
beyond insertion order, no concurrent-writer protection (single-session).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi.messages import TextContent
from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="hypothesis_tools",
    description="Track RCA hypotheses in an in-memory list of dicts.",
    registers=("tool:update_hypothesis", "tool:remove_hypothesis"),
)

_STATUSES = (
    "formed",
    "investigating",
    "confirmed",
    "rejected",
    "refined",
    "inconclusive",
)


@dataclass
class _Entry:
    id: str
    description: str
    status: str = "formed"
    evidence_summary: str | None = None
    parent_id: str | None = None


@dataclass
class _Store:
    entries: dict[str, _Entry] = field(default_factory=dict)


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _err(msg: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        is_error=True,
    )


def install(api: ExtensionAPI, _config: dict[str, Any]) -> None:
    store = _Store()

    async def _update(args: dict[str, Any]) -> ToolResult:
        hid = str(args.get("id", "")).strip()
        desc = str(args.get("description", "")).strip()
        if not hid or not desc:
            return _err("id and description are required")
        status = str(args.get("status", "formed"))
        if status not in _STATUSES:
            return _err(f"status must be one of {_STATUSES}")
        entry = _Entry(
            id=hid,
            description=desc,
            status=status,
            evidence_summary=_maybe_str(args.get("evidence_summary")),
            parent_id=_maybe_str(args.get("parent_id")),
        )
        store.entries[hid] = entry
        return _ok(
            json.dumps(
                {
                    "id": entry.id,
                    "status": entry.status,
                    "description": entry.description,
                    "evidence_summary": entry.evidence_summary,
                    "parent_id": entry.parent_id,
                    "total": len(store.entries),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    async def _remove(args: dict[str, Any]) -> ToolResult:
        hid = str(args.get("id", "")).strip()
        existed = store.entries.pop(hid, None) is not None
        return _ok(
            json.dumps(
                {"id": hid, "removed": existed, "remaining": len(store.entries)},
                ensure_ascii=False,
            )
        )

    api.register_tool(
        FunctionTool(
            name="update_hypothesis",
            description=(
                "Create or update a working hypothesis. Status must be one of "
                f"{list(_STATUSES)}. Use this to record what you currently "
                "believe and why — keep evidence_summary terse."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": list(_STATUSES)},
                    "evidence_summary": {"type": ["string", "null"]},
                    "parent_id": {"type": ["string", "null"]},
                },
                "required": ["id", "description"],
                "additionalProperties": False,
            },
            fn=_update,
        )
    )
    api.register_tool(
        FunctionTool(
            name="remove_hypothesis",
            description="Remove a hypothesis by id.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            fn=_remove,
        )
    )


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["MANIFEST", "install"]
