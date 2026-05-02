"""Compatibility wrappers that persist RCA hypotheses as shared artifacts."""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi.messages import TextContent
from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.extensions.builtin.artifact_store import ArtifactStoreHandle
from agentm.harness.events import SessionReadyEvent
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="hypothesis_tools",
    description="Persist RCA hypotheses as append-only shared artifacts.",
    registers=("tool:update_hypothesis", "tool:remove_hypothesis", "event:session_ready"),
)

_STATUSES = (
    "formed",
    "investigating",
    "confirmed",
    "rejected",
    "refined",
    "inconclusive",
)


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _err(msg: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        is_error=True,
    )


def install(api: ExtensionAPI, _config: dict[str, Any]) -> None:
    store = ArtifactStoreHandle(api, {})

    async def _on_session_ready(event: SessionReadyEvent) -> None:
        await store.on_session_ready(event)

    async def _update(args: dict[str, Any]) -> ToolResult:
        hid = str(args.get("id", "")).strip()
        desc = str(args.get("description", "")).strip()
        if not hid or not desc:
            return _err("id and description are required")
        status = str(args.get("status", "formed"))
        if status not in _STATUSES:
            return _err(f"status must be one of {_STATUSES}")
        evidence_summary = _maybe_str(args.get("evidence_summary"))
        parent_id = _maybe_str(args.get("parent_id"))
        payload = {
            "id": hid,
            "status": status,
            "description": desc,
            "evidence_summary": evidence_summary,
            "parent_id": parent_id,
        }
        result = await store.write_artifact(
            kind="hypothesis",
            title=f"{hid} {status}",
            body=json.dumps(payload, ensure_ascii=False, indent=2),
            tags=["hypothesis", status],
            parent_artifact_ids=[parent_id] if parent_id else [],
        )
        payload["artifact_id"] = result["artifact_id"]
        payload["path"] = result["path"]
        return _ok(json.dumps(payload, ensure_ascii=False, indent=2))

    async def _remove(args: dict[str, Any]) -> ToolResult:
        hid = str(args.get("id", "")).strip()
        result = await store.write_artifact(
            kind="hypothesis",
            title=f"{hid} removed",
            body=json.dumps({"id": hid, "removed": True}, ensure_ascii=False, indent=2),
            tags=["hypothesis", "removed"],
        )
        return _ok(
            json.dumps(
                {
                    "id": hid,
                    "removed": True,
                    "artifact_id": result["artifact_id"],
                    "path": result["path"],
                },
                ensure_ascii=False,
            )
        )

    api.on("session_ready", _on_session_ready)
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
            description="Record that a hypothesis was removed by id.",
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
