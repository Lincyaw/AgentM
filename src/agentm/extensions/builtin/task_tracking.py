"""Builtin ``task`` atom — session-scoped structured task tracking.

The agent creates, decomposes, and tracks progress on tasks during a
session. Tasks live in memory only — they don't persist across sessions
(use ``memory`` for that). Mirrors the task-tracking pattern from Claude
Code (TaskCreate / TaskGet / TaskList / TaskUpdate).

Tools:

* ``task_create(subject, description)`` — create a task, optionally under
  a parent for hierarchical decomposition.
* ``task_update(task_id, …)`` — change status / subject / description /
  dependencies; ``status="deleted"`` removes the task.
* ``task_list()`` — overview of all live tasks with status and blockers.
* ``task_get(task_id)`` — full details including description and deps.

Dependency model: ``blocks`` / ``blocked_by`` edges between tasks.
Completing a task auto-removes it from downstream ``blocked_by`` lists.

§11: single file; ``MANIFEST`` + ``install(api, config)``; no atom-to-atom
imports; ``core.abi`` only; no ``core.runtime.*`` / ``core._internal``.
State is per-session and in-memory.
"""

from __future__ import annotations

import json
from typing import Any, Literal  # noqa: UP035

from pydantic import BaseModel, Field

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest

_Status = Literal["pending", "in_progress", "completed"]


class _Task:
    __slots__ = (
        "id", "subject", "description", "active_form",
        "status", "parent_id", "blocks", "blocked_by", "metadata",
    )

    def __init__(
        self,
        *,
        id: str,
        subject: str,
        description: str,
        active_form: str | None = None,
        status: _Status = "pending",
        parent_id: str | None = None,
    ) -> None:
        self.id = id
        self.subject = subject
        self.description = description
        self.active_form = active_form
        self.status: _Status = status
        self.parent_id = parent_id
        self.blocks: set[str] = set()
        self.blocked_by: set[str] = set()
        self.metadata: dict[str, Any] = {}

    def to_summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "subject": self.subject,
            "status": self.status,
        }
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.blocked_by:
            d["blocked_by"] = sorted(self.blocked_by)
        return d

    def to_detail(self) -> dict[str, Any]:
        d = self.to_summary()
        d["description"] = self.description
        if self.active_form:
            d["active_form"] = self.active_form
        if self.blocks:
            d["blocks"] = sorted(self.blocks)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class _TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, _Task] = {}
        self._next_id: int = 1

    def _alloc_id(self) -> str:
        tid = str(self._next_id)
        self._next_id += 1
        return tid

    def create(
        self,
        subject: str,
        description: str,
        *,
        active_form: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _Task:
        tid = self._alloc_id()
        task = _Task(
            id=tid,
            subject=subject,
            description=description,
            active_form=active_form,
            parent_id=parent_id,
        )
        if metadata:
            task.metadata.update(metadata)
        self._tasks[tid] = task
        return task

    def get(self, task_id: str) -> _Task | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[_Task]:
        return list(self._tasks.values())

    def delete(self, task_id: str) -> _Task | None:
        task = self._tasks.pop(task_id, None)
        if task is None:
            return None
        for other in self._tasks.values():
            other.blocked_by.discard(task_id)
            other.blocks.discard(task_id)
        return task

    def complete(self, task: _Task) -> None:
        task.status = "completed"
        for other in self._tasks.values():
            other.blocked_by.discard(task.id)

    def _link_dependency(self, blocker_id: str, blocked_id: str) -> str | None:
        blocker = self._tasks.get(blocker_id)
        if blocker is None:
            return f"task {blocker_id!r} not found"
        blocked = self._tasks.get(blocked_id)
        if blocked is None:
            return f"task {blocked_id!r} not found"
        blocker.blocks.add(blocked_id)
        blocked.blocked_by.add(blocker_id)
        return None

    def add_blocks(self, task_id: str, target_ids: list[str]) -> str | None:
        for tid in target_ids:
            if (err := self._link_dependency(task_id, tid)):
                return err
        return None

    def add_blocked_by(self, task_id: str, blocker_ids: list[str]) -> str | None:
        for bid in blocker_ids:
            if (err := self._link_dependency(bid, task_id)):
                return err
        return None


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

MANIFEST = ExtensionManifest(
    name="task_tracking",
    description=(
        "Session-scoped task tracking: create, decompose, and track "
        "progress on structured work items within a session."
    ),
    registers=(
        "tool:task_create",
        "tool:task_update",
        "tool:task_list",
        "tool:task_get",
    ),
    requires=(),
)


# ---------------------------------------------------------------------------
# Tool schemas (Pydantic → JSON Schema via pydantic_to_tool_schema)
# ---------------------------------------------------------------------------

class _CreateParams(BaseModel):
    subject: str = Field(description="Brief, actionable title in imperative form (e.g. 'Fix authentication bug in login flow').")
    description: str = Field(description="What needs to be done.")
    active_form: str | None = Field(default=None, description="Present-continuous form shown while in_progress (e.g. 'Fixing authentication bug').")
    parent_id: str | None = Field(default=None, description="ID of a parent task to nest under.")
    metadata: dict[str, Any] | None = Field(default=None, description="Arbitrary metadata to attach.")


class _UpdateParams(BaseModel):
    task_id: str = Field(description="The ID of the task to update.")
    status: Literal["pending", "in_progress", "completed", "deleted"] | None = Field(default=None, description="New status. 'deleted' permanently removes the task.")
    subject: str | None = Field(default=None, description="New subject.")
    description: str | None = Field(default=None, description="New description.")
    active_form: str | None = Field(default=None, description="New active-form text.")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata keys to merge. Set a key to null to delete it.")
    add_blocks: list[str] | None = Field(default=None, description="Task IDs that cannot start until this one completes.")
    add_blocked_by: list[str] | None = Field(default=None, description="Task IDs that must complete before this one can start.")


class _GetParams(BaseModel):
    task_id: str = Field(description="The ID of the task to retrieve.")


class _ListParams(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(payload: Any) -> ToolResult:
    text = json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(msg: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=msg)], is_error=True)


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    mgr = _TaskManager()

    async def _create(args: dict[str, Any]) -> ToolResult:
        parent_id = args.get("parent_id")
        if parent_id is not None and mgr.get(str(parent_id)) is None:
            return _error(f"parent task {parent_id!r} not found")

        task = mgr.create(
            subject=str(args["subject"]),
            description=str(args["description"]),
            active_form=args.get("active_form"),
            parent_id=str(parent_id) if parent_id is not None else None,
            metadata=args.get("metadata"),
        )
        return _ok(task.to_detail())

    async def _update(args: dict[str, Any]) -> ToolResult:
        task_id = str(args["task_id"])

        status = args.get("status")
        if status == "deleted":
            removed = mgr.delete(task_id)
            if removed is None:
                return _error(f"task {task_id!r} not found")
            return _ok({"deleted": task_id})

        task = mgr.get(task_id)
        if task is None:
            return _error(f"task {task_id!r} not found")

        if "subject" in args:
            task.subject = str(args["subject"])
        if "description" in args:
            task.description = str(args["description"])
        if "active_form" in args:
            task.active_form = args["active_form"]

        if "metadata" in args and isinstance(args["metadata"], dict):
            for k, v in args["metadata"].items():
                if v is None:
                    task.metadata.pop(k, None)
                else:
                    task.metadata[k] = v

        if "add_blocks" in args:
            err = mgr.add_blocks(task_id, [str(x) for x in args["add_blocks"]])
            if err:
                return _error(err)

        if "add_blocked_by" in args:
            err = mgr.add_blocked_by(task_id, [str(x) for x in args["add_blocked_by"]])
            if err:
                return _error(err)

        if status is not None:
            if status not in ("pending", "in_progress", "completed"):
                return _error(f"invalid status {status!r}")
            if status == "completed":
                mgr.complete(task)
            else:
                task.status = status  # type: ignore[assignment]

        return _ok(task.to_detail())

    async def _list(args: dict[str, Any]) -> ToolResult:  # noqa: ARG001
        tasks = mgr.list_all()
        if not tasks:
            return _ok({"tasks": [], "summary": "No tasks."})
        return _ok({"tasks": [t.to_summary() for t in tasks]})

    async def _get(args: dict[str, Any]) -> ToolResult:
        task = mgr.get(str(args["task_id"]))
        if task is None:
            return _error(f"task {args['task_id']!r} not found")
        return _ok(task.to_detail())

    api.register_tool(
        FunctionTool(
            name="task_create",
            description=(
                "Create a structured task to track progress on a multi-step "
                "piece of work. Use for complex work that benefits from "
                "decomposition and progress tracking."
            ),
            parameters=pydantic_to_tool_schema(_CreateParams),
            fn=_create,
        )
    )
    api.register_tool(
        FunctionTool(
            name="task_update",
            description=(
                "Update a task's status, subject, description, or "
                "dependencies. Set status to 'in_progress' when starting "
                "work, 'completed' when done, 'deleted' to remove."
            ),
            parameters=pydantic_to_tool_schema(_UpdateParams),
            fn=_update,
        )
    )
    api.register_tool(
        FunctionTool(
            name="task_list",
            description=(
                "List all tasks with their status and blockers. Use to "
                "check overall progress or find available work."
            ),
            parameters=pydantic_to_tool_schema(_ListParams),
            fn=_list,
        )
    )
    api.register_tool(
        FunctionTool(
            name="task_get",
            description=(
                "Get full details of a task including description, "
                "dependencies, and metadata."
            ),
            parameters=pydantic_to_tool_schema(_GetParams),
            fn=_get,
        )
    )
