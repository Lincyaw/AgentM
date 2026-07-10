"""``workflow`` atom — run an LLM-authored Python orchestration script.

The atom entry point stays in this file so scenario manifests continue to load
``agentm.extensions.builtin.workflow``. The heavier runtime lives under the
private ``agentm.extensions.builtin._workflow`` package, following the same
entry-file/private-implementation pattern as the ``operations`` atom.

See ``.claude/designs/dynamic-workflow.md`` for the design. A workflow script
gets the curated SDK names ``agent`` / ``parallel`` / ``pipeline`` / ``budget`` /
``args`` / ``log`` / ``phase`` and can fan out deterministic child sessions.
"""

from __future__ import annotations

import json

from loguru import logger
from pydantic import BaseModel, Field
from typing import Any

from agentm.core.abi import (
    ARTIFACT_STORE_SERVICE,
    ExtensionAPI,
    FunctionTool,
    MODEL_RESOLVER_SERVICE,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest
from agentm.extensions.builtin._workflow.lineage import (
    ancestors as _lineage_ancestors,
    derive_lineage as _derive_lineage,
)
from agentm.extensions.builtin._workflow.runner import (
    WorkflowRunner,
    WorkflowValidationError,
    _WORKFLOW_RUNNER_SERVICE,
    _WorkflowArgs,
    _coerce_result,
    _detect_script_mode as _detect_script_mode,
    _error,
    _validate_script as _validate_script,
)
from agentm.extensions.builtin._workflow.sdk import (
    AgentMockMode,
    AgentResult,
    BudgetSnapshot,
    JournalEntry,
    RunSummary,
    WorkflowContext,
    WorkflowPhaseEvent,
    WorkflowResult,
    _AGENT_COUNT_BACKSTOP,
    _WORKER_PURPOSE,
    _coerce_agent_mock,
    _default_concurrency,
    load_journal_entries,
    write_invalidation,
)


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
        "journaled node graph (which result flowed into which prompt), "
        "workflow_invalidate flags a wrong node so the next run redoes it and "
        "everything derived from it."
    ),
    registers=(
        "tool:workflow",
        "tool:workflow_lineage",
        "tool:workflow_invalidate",
        "event:workflow_phase",
    ),
    config_schema=WorkflowConfig,
    # The workflow-local journal (resume) is backed by the artifact_store
    # service; the atom reaches it via api.get_service, never an import.
    requires=("artifact_store",),
    conflicts=(),
)

# Name of the service that backs the workflow-local resume journal.


class _WorkflowLineageParams(BaseModel):
    key: str | None = Field(
        default=None,
        description=(
            "Journal key of one node: restrict the output to that node plus "
            "its upstream ancestors. Omit to get the full graph."
        ),
    )


class _WorkflowInvalidateParams(BaseModel):
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


class _JournalTools:
    """workflow_lineage / workflow_invalidate — the recovery-loop surface
    over the journal (reliability-substrate.md §4.3)."""

    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api

    def _store(self) -> Any | None:
        return self._api.get_service(ARTIFACT_STORE_SERVICE)

    async def lineage(self, args: dict[str, Any]) -> ToolResult:
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
        graph = _derive_lineage(entries)
        key_arg = args.get("key")
        selected = entries
        if isinstance(key_arg, str) and key_arg.strip():
            key = key_arg.strip()
            by_key = {entry.key: entry for entry in entries}
            if key not in by_key:
                known = ", ".join(sorted(by_key)[:10])
                return _error(
                    f"workflow_lineage: no journal entry with key '{key}'. "
                    f"Known keys include: {known}"
                )
            wanted = {key, *_lineage_ancestors(graph, key)}
            selected = [entry for entry in entries if entry.key in wanted]
        selected_keys = {entry.key for entry in selected}
        payload = {
            "nodes": [_entry_view(entry) for entry in selected],
            "edges": [
                {"src": edge.src, "dst": edge.dst, "kind": edge.kind}
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
        store = self._store()
        if store is None:
            return _error(
                "workflow_invalidate: no artifact store in this session, so "
                "there is no workflow journal to invalidate against"
            )
        key = str(args.get("key", "")).strip()
        reason = str(args.get("reason", "")).strip()
        if not key or not reason:
            return _error("workflow_invalidate: both key and reason are required")
        entries = await load_journal_entries(store)
        if key not in {entry.key for entry in entries}:
            return _error(
                f"workflow_invalidate: no journaled result with key '{key}'. "
                "Run workflow_lineage first to find the node's key."
            )
        feedback_arg = args.get("feedback")
        feedback = (
            str(feedback_arg).strip()
            if isinstance(feedback_arg, str) and str(feedback_arg).strip()
            else None
        )
        await write_invalidation(
            store,
            key=key,
            reason=reason,
            feedback=feedback,
            carry_previous=bool(args.get("carry_previous", False)),
        )
        return _ok_text(
            f"Invalidated journal entry {key}. On the next run of the same "
            "workflow script this node re-runs"
            + (" with your feedback injected" if feedback else "")
            + ", and every node whose prompt derives from its output re-runs "
            "automatically; unaffected nodes keep their cached results."
        )


class _WorkflowRuntime:
    def __init__(self, api: ExtensionAPI, config: WorkflowConfig) -> None:
        self._api = api
        self._wall_clock_timeout = self._positive_float_or_none(
            config.wall_clock_timeout_s
        )
        self._agent_mock = config.agent_mock
        self._model_resolver = api.get_service(MODEL_RESOLVER_SERVICE)
        self._runner = WorkflowRunner(
            api,
            concurrency=self._positive_int_or_default(
                config.max_concurrency,
                _default_concurrency(),
            ),
            max_agents=self._positive_int_or_default(
                config.max_agents,
                _AGENT_COUNT_BACKSTOP,
            ),
            wall_clock_timeout=self._wall_clock_timeout,
            default_scenario=config.default_scenario,
            budget_tokens=config.budget_tokens,
            default_agent_timeout=self._positive_float_or_none(
                config.default_agent_timeout_s
            ),
            agent_mock=self._agent_mock,
            model_resolver=self._model_resolver,
        )

    def install(self) -> None:
        self._api.set_service(_WORKFLOW_RUNNER_SERVICE, self._runner)
        journal_tools = _JournalTools(self._api)
        self._api.register_tool(
            FunctionTool(
                name="workflow_lineage",
                description=(
                    "Inspect the workflow journal as a dependency graph: one "
                    "node per journaled agent() call (key, prompt/result "
                    "snippets, invalidated flag), with an edge where one "
                    "node's result appears verbatim in another node's prompt. "
                    "Nodes whose dataflow was transformed before interpolation "
                    "fall back to conservative order_candidates. Use this to "
                    "trace a wrong result back to its upstream nodes before "
                    "deciding which one to invalidate."
                ),
                parameters=_WorkflowLineageParams,
                fn=journal_tools.lineage,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="workflow_invalidate",
                description=(
                    "Flag one journaled agent() result as wrong. The next run "
                    "of the same workflow script redoes that node (with your "
                    "feedback injected into its prompt) and automatically "
                    "redoes every downstream node derived from its output, "
                    "while unaffected nodes keep their cached results. The "
                    "flag is an append-only record; nothing is deleted."
                ),
                parameters=_WorkflowInvalidateParams,
                fn=journal_tools.invalidate,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="workflow",
                description=(
                    "Run an async Python orchestration script (inline or from a "
                    "file via script_path) that fans out deterministic child agent "
                    "sessions. ONLY the script's final return value enters your "
                    "context — all intermediate child outputs are dropped, so use "
                    "this for high-fan-out verify/judge/sweep harnesses where "
                    "per-child detail is noise. Names: agent, parallel, pipeline, "
                    "budget, args, json, log, phase. The script runs in a curated "
                    "namespace (data builtins only; no import / open / time / "
                    "random). agent() results are journaled by a hash of the "
                    "prompt plus the agent's options (schema/scenario/model/"
                    "...; retry/timeout and the top-level args payload are "
                    "excluded), so re-running a workflow resumes from cache. "
                    "agent(prompt, schema={...}) returns a parsed dict conforming "
                    "to the JSON Schema."
                ),
                parameters=_WorkflowArgs,
                fn=self.run_workflow,
                metadata={"workflow": True},
            )
        )

    @staticmethod
    def _positive_int_or_default(value: int | None, default: int) -> int:
        if value is not None and value > 0:
            return value
        return default

    @staticmethod
    def _positive_float_or_none(value: float | None) -> float | None:
        if value is not None and value > 0:
            return value
        return None

    @staticmethod
    def _cwd_arg(args: dict[str, Any]) -> str | None:
        cwd_arg = args.get("cwd")
        if isinstance(cwd_arg, str) and cwd_arg.strip():
            return str(cwd_arg)
        return None

    def _provider_arg(
        self, model_arg: Any
    ) -> tuple[tuple[str, dict[str, Any]] | None, ToolResult | None]:
        if not isinstance(model_arg, str) or not model_arg.strip():
            return None, None
        if self._model_resolver is None:
            return None, _error(
                "workflow: model= requires the model_resolver service, "
                "but it is not available in this session"
            )
        provider = self._model_resolver(model_arg.strip())
        if provider is None:
            return None, _error(f"workflow: unknown model profile '{model_arg}'")
        return provider, None

    async def run_workflow(self, args: dict[str, Any]) -> ToolResult:
        script = args.get("script")
        script_path_raw = args.get("script_path")

        if script_path_raw and script:
            return _error("workflow: 'script' and 'script_path' are mutually exclusive")

        cwd = self._cwd_arg(args)
        provider, error = self._provider_arg(args.get("model"))
        if error is not None:
            return error

        try:
            agent_mock_arg = _coerce_agent_mock(
                args.get("agent_mock"), self._agent_mock
            )
        except ValueError as exc:
            return _error(str(exc))

        try:
            if isinstance(script_path_raw, str) and script_path_raw.strip():
                result = await self._runner.run_file(
                    script_path_raw,
                    args.get("args"),
                    cwd=cwd,
                    provider=provider,
                    agent_mock=agent_mock_arg,
                )
            elif isinstance(script, str) and script.strip():
                result = await self._runner.run_script(
                    script,
                    args.get("args"),
                    cwd=cwd,
                    provider=provider,
                    agent_mock=agent_mock_arg,
                )
            else:
                return _error("workflow: either 'script' or 'script_path' is required")
        except WorkflowValidationError as exc:
            return _error(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return _error(f"workflow: {exc}")
        except TimeoutError:
            return _error(
                "workflow: script exceeded wall-clock budget "
                f"({self._wall_clock_timeout}s)"
            )
        except Exception as exc:
            logger.warning("workflow script error: {}: {}", type(exc).__name__, exc)
            return _error(f"workflow script error: {type(exc).__name__}: {exc}")

        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=_coerce_result(result),
                )
            ],
            extras={
                "progress": self._runner.last_progress,
                "summary": self._runner.last_run_summary,
            },
        )


def install(api: ExtensionAPI, config: WorkflowConfig) -> None:
    if api.purpose == _WORKER_PURPOSE:
        return
    _WorkflowRuntime(api, config).install()


__all__ = (
    "AgentResult",
    "BudgetSnapshot",
    "MANIFEST",
    "RunSummary",
    "WorkflowConfig",
    "WorkflowContext",
    "WorkflowPhaseEvent",
    "WorkflowResult",
    "WorkflowRunner",
    "install",
)
