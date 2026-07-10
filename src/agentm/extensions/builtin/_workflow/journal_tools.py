"""``workflow_lineage`` / ``workflow_invalidate`` tool implementations.

The recovery-loop surface over the workflow journal
(reliability-substrate.md §4.3): inspect the journaled node graph, then flag
a wrong node so the next run redoes it and everything derived from it. The
entry file ``workflow.py`` only registers these; the implementation lives
here per the package's entry-file/private-implementation pattern.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agentm.core.abi import ARTIFACT_STORE_SERVICE, ExtensionAPI, TextContent, ToolResult

from agentm.extensions.builtin._workflow.journal import (
    JournalEntry,
    _ArtifactStore,
    journal_key_exists,
    load_journal_entries,
    write_invalidation,
)
from agentm.extensions.builtin._workflow.lineage import ancestors, derive_lineage
from agentm.extensions.builtin._workflow.runner import _error


class WorkflowLineageParams(BaseModel):
    key: str | None = Field(
        default=None,
        description=(
            "Journal key of one node: restrict the output to that node plus "
            "its upstream ancestors. Omit to get the full graph."
        ),
    )


class WorkflowInvalidateParams(BaseModel):
    key: str = Field(
        description=(
            "Journal key of the wrong agent() result (find it with "
            "workflow_lineage)."
        )
    )
    reason: str = Field(description="Why the result is wrong.")
    feedback: str | None = Field(
        default=None,
        description=(
            "Guidance injected into the re-run prompt so the new attempt "
            "does not repeat the failure. Without it the node re-runs with "
            "an identical prompt and may reproduce the same wrong output."
        ),
    )
    carry_previous: bool = Field(
        default=False,
        description=(
            "Include the invalidated result in the re-run prompt for "
            "reference (default: fresh solve without anchoring on it)."
        ),
    )


_SNIPPET_CHARS = 200


def _entry_view(entry: JournalEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "prompt_head": (entry.prompt or "")[:_SNIPPET_CHARS],
        "result_head": entry.result[:_SNIPPET_CHARS],
        "invalidated": entry.invalidated,
        "recorded_at": entry.timestamp,
    }


def _ok_text(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


class JournalTools:
    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api

    def _store(self) -> _ArtifactStore | None:
        return self._api.get_service(ARTIFACT_STORE_SERVICE)

    async def lineage(self, args: dict[str, Any]) -> ToolResult:
        try:
            params = WorkflowLineageParams.model_validate(args)
        except ValidationError as exc:
            return _error(f"workflow_lineage: invalid arguments: {exc}")
        store = self._store()
        if store is None:
            return _error(
                "workflow_lineage: no artifact store in this session, so "
                "there is no workflow journal to inspect"
            )
        entries = await load_journal_entries(store)
        if not entries:
            return _ok_text(
                "The workflow journal is empty — no agent() results have "
                "been recorded in this session tree yet."
            )
        graph = derive_lineage(entries)
        selected = entries
        if params.key is not None and params.key.strip():
            key = params.key.strip()
            if key not in {entry.key for entry in entries}:
                known = ", ".join(sorted(entry.key for entry in entries)[:10])
                return _error(
                    f"workflow_lineage: no journal entry with key '{key}'. "
                    f"Known keys include: {known}"
                )
            wanted = {key, *ancestors(graph, key)}
            selected = [entry for entry in entries if entry.key in wanted]
        selected_keys = {entry.key for entry in selected}
        payload = {
            "nodes": [_entry_view(entry) for entry in selected],
            "edges": [
                {"src": edge.src, "dst": edge.dst}
                for edge in graph.edges
                if edge.src in selected_keys and edge.dst in selected_keys
            ],
            "order_candidates": {
                node: parents
                for node, parents in graph.order_candidates.items()
                if node in selected_keys
            },
        }
        return _ok_text(json.dumps(payload, ensure_ascii=False, indent=2))

    async def invalidate(self, args: dict[str, Any]) -> ToolResult:
        try:
            params = WorkflowInvalidateParams.model_validate(args)
        except ValidationError as exc:
            return _error(f"workflow_invalidate: invalid arguments: {exc}")
        store = self._store()
        if store is None:
            return _error(
                "workflow_invalidate: no artifact store in this session, so "
                "there is no workflow journal to invalidate against"
            )
        key = params.key.strip()
        reason = params.reason.strip()
        if not key or not reason:
            return _error("workflow_invalidate: both key and reason are required")
        if not await journal_key_exists(store, key):
            return _error(
                f"workflow_invalidate: no journaled result with key '{key}'. "
                "Run workflow_lineage first to find the node's key."
            )
        feedback = (params.feedback or "").strip() or None
        await write_invalidation(
            store,
            key=key,
            reason=reason,
            feedback=feedback,
            carry_previous=params.carry_previous,
        )
        return _ok_text(
            f"Invalidated journal entry {key}. On the next run of the same "
            "workflow script this node re-runs"
            + (" with your feedback injected" if feedback else "")
            + ", and every node whose prompt derives from its output re-runs "
            "automatically; unaffected nodes keep their cached results."
        )


__all__ = (
    "JournalTools",
    "WorkflowInvalidateParams",
    "WorkflowLineageParams",
)
