"""Shared helpers for workflow journal tests.

``FakeArtifactStore`` implements the slice of the ``artifact_store`` service
the workflow journal consumes (``write_artifact`` / ``list_artifacts`` /
``read``), with the same observable contract: sequential ids,
``created_by.timestamp`` floats, and newest-first listing. ``StubRun`` is a
``_WorkflowRun`` whose child-session spawn is replaced by a scripted stub.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi import ToolResult
from agentm.extensions.builtin._workflow.journal import _Journal
from agentm.extensions.builtin._workflow.sdk import _BudgetService, _WorkflowRun


class FakeArtifactStore:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._next_id = 1
        self._clock = 1000.0

    async def write_artifact(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        tags: list[str] | None = None,
    ) -> dict[str, str]:
        self._clock += 1.0
        artifact_id = f"art_{self._next_id:04d}"
        self._next_id += 1
        self._items.append(
            {
                "id": artifact_id,
                "kind": kind,
                "title": title,
                "tags": list(tags or []),
                "body": body,
                "created_by": {"timestamp": self._clock},
            }
        )
        return {"artifact_id": artifact_id, "path": f"/fake/{artifact_id}"}

    async def list_artifacts(self, args: dict[str, Any]) -> ToolResult:
        kind = args.get("kind")
        tags = list(args.get("tags") or [])
        limit = max(1, int(args.get("limit", 100)))
        matches = [
            item
            for item in self._items
            if (kind is None or item["kind"] == kind)
            and set(tags).issubset(set(item["tags"]))
        ]
        matches.sort(key=lambda item: item["created_by"]["timestamp"], reverse=True)
        return ToolResult(
            content=[],
            extras={
                "artifacts": [
                    {
                        "id": item["id"],
                        "kind": item["kind"],
                        "title": item["title"],
                        "tags": item["tags"],
                        "created_by": item["created_by"],
                    }
                    for item in matches[:limit]
                ]
            },
        )

    async def read(self, args: dict[str, Any]) -> ToolResult:
        artifact_id = args.get("artifact_id")
        for item in self._items:
            if item["id"] == artifact_id:
                return ToolResult(
                    content=[],
                    extras={"artifact_id": artifact_id, "body": item["body"]},
                )
        return ToolResult(content=[], is_error=True)


@dataclass(slots=True)
class StubRun(_WorkflowRun):
    """``_WorkflowRun`` with the child-session spawn replaced by a stub that
    pops scripted results and records the prompts it was driven with."""

    spawn_results: list[str] = field(default_factory=list)
    spawn_prompts: list[str] = field(default_factory=list)

    async def _spawn_and_drive(  # type: ignore[override]
        self,
        prompt: str,
        scenario: str | None,
        model: str | None,
        isolation: str | None,
        tool_allowlist: list[str] | None,
        **_kwargs: object,
    ) -> str:
        self.spawn_prompts.append(prompt)
        return self.spawn_results.pop(0)


def make_run(store: FakeArtifactStore, results: list[str]) -> StubRun:
    return StubRun(
        api=None,  # type: ignore[arg-type]  # only reached by the real spawn path
        journal=_Journal(store=store),
        budget_svc=_BudgetService(),
        semaphore=asyncio.Semaphore(2),
        spawn_results=list(results),
    )
