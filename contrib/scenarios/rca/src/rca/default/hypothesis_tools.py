"""Compatibility wrappers that persist RCA hypotheses as shared artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    SessionReadyEvent,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="hypothesis_tools",
    description="Persist RCA hypotheses as append-only shared artifacts.",
    registers=(
        "tool:add_hypothesis",
        "tool:update_hypothesis",
        "tool:remove_hypothesis",
        "tool:list_hypotheses",
        f"event:{SessionReadyEvent.CHANNEL}",
    ),
)

# Suggested (not enforced) status vocabulary surfaced in the tool
# descriptions. Status is free-text + LLM-decided — the line between e.g.
# "refined" and "investigating", or "inconclusive" and "rejected", is a
# judgment call we let the model phrase, so these are hints, not a validated
# enum (no preset enums for subjective dimensions).
_SUGGESTED_STATUSES = (
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
    # Defer the artifact_store lookup to ``session_ready`` — by then every
    # extension's ``install()`` has run, and the canonical store instance
    # owned by the ``artifact_store`` extension is registered. Constructing
    # our own would create a parallel state-divergent store (see
    # PR #65 follow-up cleanup).
    # Why: service-registry contracts are by-shape, not by-class — keep the
    # by-name lookup and a duck-type guard so the artifact-store atom stays
    # replaceable (§S4 pluggability). Importing the concrete ArtifactStore
    # class here would re-create the cross-atom coupling we removed.
    store_cell: list[Any] = [None]

    async def _on_session_ready(_event: SessionReadyEvent) -> None:
        store = api.get_service("artifact_store")
        if store is None or not hasattr(store, "write_artifact"):
            raise RuntimeError(
                "hypothesis_tools requires the 'artifact_store' extension to "
                "be loaded for the same session"
            )
        store_cell[0] = store

    def _store() -> Any:
        store = store_cell[0]
        if store is None:
            raise RuntimeError("hypothesis_tools called before session_ready")
        return store

    async def _write_hypothesis(
        *,
        hid: str,
        desc: str,
        status: str,
        evidence_summary: str | None,
        parent_id: str | None,
    ) -> ToolResult:
        payload = {
            "id": hid,
            "status": status,
            "description": desc,
            "evidence_summary": evidence_summary,
            "parent_id": parent_id,
        }
        result = await _store().write_artifact(
            kind="hypothesis",
            title=f"{hid} {status}",
            body=json.dumps(payload, ensure_ascii=False, indent=2),
            tags=["hypothesis", status],
            parent_artifact_ids=[parent_id] if parent_id else [],
        )
        payload["artifact_id"] = result["artifact_id"]
        payload["path"] = result["path"]
        return _ok(json.dumps(payload, ensure_ascii=False, indent=2))

    async def _add(args: dict[str, Any]) -> ToolResult:
        hid = str(args.get("id", "")).strip()
        desc = str(args.get("description", "")).strip()
        if not hid or not desc:
            return _err("id and description are required")
        return await _write_hypothesis(
            hid=hid,
            desc=desc,
            status="formed",
            evidence_summary=_maybe_str(args.get("evidence_summary")),
            parent_id=_maybe_str(args.get("parent_id")),
        )

    async def _update(args: dict[str, Any]) -> ToolResult:
        hid = str(args.get("id", "")).strip()
        desc = str(args.get("description", "")).strip()
        if not hid or not desc:
            return _err("id and description are required")
        status = str(args.get("status", "formed"))
        return await _write_hypothesis(
            hid=hid,
            desc=desc,
            status=status,
            evidence_summary=_maybe_str(args.get("evidence_summary")),
            parent_id=_maybe_str(args.get("parent_id")),
        )

    async def _remove(args: dict[str, Any]) -> ToolResult:
        hid = str(args.get("id", "")).strip()
        if not hid:
            return _err("id is required")
        result = await _store().write_artifact(
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

    async def _list(args: dict[str, Any]) -> ToolResult:
        try:
            limit = max(1, int(args.get("limit", 50)))
        except (TypeError, ValueError):
            return _err("limit must be an integer")
        artifact_listing = await _store().list_artifacts(
            {"kind": "hypothesis", "limit": limit}
        )
        if artifact_listing.is_error:
            return artifact_listing
        artifacts = artifact_listing.extras.get("artifacts", [])
        if not isinstance(artifacts, list):
            return _err("artifact store returned an invalid listing")
        hid_filter = _maybe_str(args.get("id"))
        status_filter = _maybe_str(args.get("status"))
        items: list[dict[str, Any]] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = _maybe_str(artifact.get("path"))
            if path is None:
                continue
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if hid_filter is not None and payload.get("id") != hid_filter:
                continue
            if status_filter is not None and payload.get("status") != status_filter:
                continue
            payload["artifact_id"] = artifact.get("id")
            payload["path"] = path
            items.append(payload)
        return _ok(json.dumps({"hypotheses": items}, ensure_ascii=False, indent=2))

    api.on(SessionReadyEvent.CHANNEL, _on_session_ready)
    api.register_tool(
        FunctionTool(
            name="add_hypothesis",
            description=(
                "Compatibility alias for recording a newly formed hypothesis. "
                "Creates a new append-only shared artifact with status='formed'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "evidence_summary": {"type": ["string", "null"]},
                    "parent_id": {"type": ["string", "null"]},
                },
                "required": ["id", "description"],
                "additionalProperties": False,
            },
            fn=_add,
        )
    )
    api.register_tool(
        FunctionTool(
            name="update_hypothesis",
            description=(
                "Create or update a working hypothesis. Status is free-text — "
                f"common values: {list(_SUGGESTED_STATUSES)}, but use whatever "
                "phrasing fits. Use this to record what you currently believe "
                "and why — keep evidence_summary terse."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string"},
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
    api.register_tool(
        FunctionTool(
            name="list_hypotheses",
            description=(
                "Compatibility listing view over hypothesis artifacts. "
                "Returns the latest matching hypothesis snapshots without "
                "loading unrelated artifact kinds."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": ["string", "null"]},
                    "status": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional free-text status filter (e.g. "
                            "'confirmed', 'removed')."
                        ),
                    },
                    "limit": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
            fn=_list,
        )
    )

def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

__all__ = ["MANIFEST", "install"]
