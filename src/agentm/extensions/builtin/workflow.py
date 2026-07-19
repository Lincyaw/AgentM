"""``workflow`` atom — run an LLM-authored Python orchestration script.

A workflow script gets the curated SDK names ``agent`` / ``parallel`` /
``pipeline`` / ``budget`` / ``args`` / ``log`` / ``phase`` and can fan out
deterministic child sessions.

MIGRATION STATUS (v2-trajectory branch): the heavy runtime for this atom lives
in the private ``agentm.extensions.builtin._workflow`` package (runner, sdk,
journal, journal_tools, lineage — ~2500 LOC on main) and depends on a
``MODEL_RESOLVER_SERVICE`` role and the main-branch spawn/child-session
semantics. Neither the ``_workflow`` package nor ``MODEL_RESOLVER_SERVICE``
exist on this branch yet. Porting them is a distinct, large effort tracked
separately.

Until that package is ported, this module preserves the atom's *shape* —
``MANIFEST``, ``WorkflowConfig``, and the ``install`` signature — and registers
the ``workflow`` / ``workflow_lineage`` / ``workflow_invalidate`` tools as
placeholders that return an explicit "not available on this branch" tool-error
instead of silently importing a package that does not exist. Every
placeholder is flagged with ``# TODO(migration):``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest

# TODO(migration): main resolved this from ``_workflow.sdk``; the workflow
# worker child sessions run under this purpose. Inlined so the install guard
# below keeps working once the runner package is ported.
_WORKER_PURPOSE = "workflow"

AgentMockMode = Literal["off", "mock"]


class WorkflowConfig(BaseModel):
    max_concurrency: int | None = None
    max_agents: int | None = None
    wall_clock_timeout_s: float | None = None
    default_scenario: str | None = None
    budget_tokens: int | None = Field(default=None, gt=0)
    default_agent_timeout_s: float | None = None
    agent_mock: AgentMockMode = Field(
        default="off",
        description=(
            "Set to 'mock' to dry-run workflow agent() calls without spawning "
            "child sessions. Each call logs its parameters and returns a "
            "synthetic result."
        ),
    )


MANIFEST = ExtensionManifest(
    name="workflow",
    description=(
        "Run an LLM-authored async Python orchestration script in a curated "
        "namespace that fans out deterministic child agent sessions via the "
        "agent/parallel/pipeline primitives. Only the final result returns to "
        "the model's context. Companion tools: workflow_lineage inspects the "
        "journaled node graph, workflow_invalidate flags a wrong node so the "
        "next run redoes it and everything derived from it."
    ),
    registers=(
        "tool:workflow",
        "tool:workflow_lineage",
        "tool:workflow_invalidate",
    ),
    config_schema=WorkflowConfig,
    # The workflow-local journal (resume) is backed by the artifact_store
    # service; the atom reaches it via services.get, never an import.
    requires=("artifact_store",),
    priority=AtomInstallPriority.TOOL,
)


# TODO(migration): these mirror the ``_workflow.runner`` / ``_workflow.journal_tools``
# tool parameter schemas so the tool surface stays stable. Replace with the
# ported schemas once the ``_workflow`` package is migrated.


class _WorkflowArgs(BaseModel):
    script: str | None = Field(
        default=None, description="Inline async Python orchestration script."
    )
    script_path: str | None = Field(
        default=None,
        description="Path to a script file; mutually exclusive with 'script'.",
    )
    args: Any | None = Field(
        default=None, description="Top-level args payload passed to the script."
    )
    cwd: str | None = Field(default=None, description="Working directory override.")
    model: str | None = Field(
        default=None, description="Model profile name for child agent() calls."
    )
    agent_mock: str | None = Field(
        default=None, description="Override the configured agent_mock mode."
    )


class _WorkflowLineageParams(BaseModel):
    pass


class _WorkflowInvalidateParams(BaseModel):
    key: str = Field(description="Journal node key to flag as wrong.")
    feedback: str | None = Field(
        default=None, description="Feedback injected into the redone node's prompt."
    )


def _not_available(tool: str) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    f"workflow: the '{tool}' tool is not available on this "
                    "build. The workflow runtime (the private _workflow runner "
                    "package and the model_resolver service) has not been "
                    "migrated to the v2-trajectory branch yet."
                ),
            )
        ],
        is_error=True,
    )


class _WorkflowRuntime:
    def __init__(self, api: AtomAPI, config: WorkflowConfig) -> None:
        self._api = api
        self._config = config

    def install(self) -> None:
        # TODO(migration): construct the real WorkflowRunner + JournalTools once
        # the ``_workflow`` package and MODEL_RESOLVER_SERVICE are ported; wire
        # the runner into these tools in place of the placeholder handlers.
        self._api.register_tool(
            FunctionTool(
                name="workflow_lineage",
                description=(
                    "Inspect the workflow journal as a dependency graph. "
                    "(Not available on this build — pending runner migration.)"
                ),
                parameters=pydantic_to_tool_schema(_WorkflowLineageParams),
                fn=self._lineage,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="workflow_invalidate",
                description=(
                    "Flag one journaled agent() result as wrong so the next run "
                    "redoes it and its downstream nodes. "
                    "(Not available on this build — pending runner migration.)"
                ),
                parameters=pydantic_to_tool_schema(_WorkflowInvalidateParams),
                fn=self._invalidate,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="workflow",
                description=(
                    "Run an async Python orchestration script that fans out "
                    "deterministic child agent sessions. "
                    "(Not available on this build — pending runner migration.)"
                ),
                parameters=pydantic_to_tool_schema(_WorkflowArgs),
                fn=self._run_workflow,
                metadata={"workflow": True},
            )
        )

    async def _run_workflow(self, args: dict[str, Any]) -> ToolResult:
        del args
        return _not_available("workflow")

    async def _lineage(self, args: dict[str, Any]) -> ToolResult:
        del args
        return _not_available("workflow_lineage")

    async def _invalidate(self, args: dict[str, Any]) -> ToolResult:
        del args
        return _not_available("workflow_invalidate")


def install(api: AtomAPI, config: WorkflowConfig) -> None:
    if api.ctx.purpose == _WORKER_PURPOSE:
        return
    _WorkflowRuntime(api, config).install()


__all__ = (
    "MANIFEST",
    "WorkflowConfig",
    "install",
)
