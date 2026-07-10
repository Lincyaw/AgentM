"""In-memory artifact store fake for workflow journal tests.

Implements the slice of the ``artifact_store`` service the workflow journal
consumes (``write_artifact`` / ``list_artifacts`` / ``read``), with the same
observable contract: sequential ids, ``created_by.timestamp`` floats, and
newest-first listing.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import ToolResult


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
