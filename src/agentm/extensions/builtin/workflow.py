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

from loguru import logger
from pydantic import BaseModel, Field
from typing import Any

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    MODEL_RESOLVER_SERVICE,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest
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
    RunSummary,
    WorkflowContext,
    WorkflowPhaseEvent,
    WorkflowResult,
    _AGENT_COUNT_BACKSTOP,
    _WORKER_PURPOSE,
    _coerce_agent_mock,
    _default_concurrency,
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
        "the model's context."
    ),
    registers=(
        "tool:workflow",
        "event:workflow_phase",
    ),
    config_schema=WorkflowConfig,
    # The workflow-local journal (resume) is backed by the artifact_store
    # service; the atom reaches it via api.get_service, never an import.
    requires=("artifact_store",),
    conflicts=(),
)

# Name of the service that backs the workflow-local resume journal.


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
