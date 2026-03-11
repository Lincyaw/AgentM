"""Sub-Agent creation and AgentPool management."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field

from agentm.config.schema import AgentConfig, ModelConfig, ScenarioConfig
from agentm.core.compression import build_compression_hook
from agentm.core.prompt import load_prompt_template
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.tools.think import think


# ---------------------------------------------------------------------------
# Role-specific structured output schemas
# ---------------------------------------------------------------------------


class _BaseAnswer(BaseModel):
    """Shared fields across all sub-agent answer types."""

    findings: str = Field(
        description=(
            "Structured findings following the <output> format in your system "
            "prompt. Exact service names in backticks. No reasoning or caveats."
        ),
    )


class ScoutAnswer(_BaseAnswer):
    """Scout agent output: structural map of the incident + investigation leads."""

    leads: list[str] = Field(
        description=(
            "3-6 divergent investigation directions. Each lead is one sentence: "
            "'[service/component] may [cause] because [evidence]'. "
            "Cover different fault domains (network, resource, dependency, config, code)."
        ),
    )


class DeepAnalyzeAnswer(_BaseAnswer):
    """Deep-analyze agent output: causal mechanism + refined hypotheses."""

    leads: list[str] = Field(
        description=(
            "1-3 refined hypotheses about specific causal mechanisms. Narrow "
            "and evidence-heavy — only include leads that scout-level analysis "
            "could NOT have produced."
        ),
    )


class VerifyAnswer(_BaseAnswer):
    """Verify agent output: adversarial test verdict with tagged (+)/(-) evidence."""

    verdict: str = Field(
        description=(
            "SUPPORTED, CONTRADICTED, or INCONCLUSIVE — followed by exactly "
            "one sentence citing the strongest piece of evidence. SUPPORTED "
            "means the hypothesis survived active disproof attempts."
        ),
    )


# Keep backward-compatible alias used by task_manager docstrings
SubAgentAnswer = ScoutAnswer | DeepAnalyzeAnswer | VerifyAnswer

ANSWER_SCHEMA: dict[str, type[BaseModel]] = {
    "scout": ScoutAnswer,
    "deep_analyze": DeepAnalyzeAnswer,
    "verify": VerifyAnswer,
}


def create_sub_agent(
    agent_id: str,
    config: AgentConfig,
    tool_registry: ToolRegistry,
    task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
    model_config: ModelConfig | None = None,
    trajectory: TrajectoryCollector | None = None,
    task_id: str | None = None,
    checkpointer: Any = None,
) -> Any:
    """Create a Sub-Agent subgraph via create_react_agent. Returns a CompiledGraph.

    A new compiled subgraph is created per dispatch so that the *agent_id*
    and *task_id* are baked into the hooks at compile time — no mutable
    overrides needed.
    """
    llm_kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }
    if model_config is not None:
        llm_kwargs["api_key"] = model_config.api_key
        if model_config.base_url:
            llm_kwargs["base_url"] = model_config.base_url
    model = ChatOpenAI(**llm_kwargs)

    tools = [
        tool_registry.get(name).create_with_config(**config.tool_settings.get(name, {}))
        for name in config.tools
    ]
    # Think tool is always available — it's a free scratchpad, not counted in budget.
    tools.append(think)

    tools_description = "\n".join(f"- `{t.name}`: {t.description}" for t in tools)
    template_context = {"agent_id": agent_id, "tools_description": tools_description}

    if config.prompt is None:
        if config.task_type_prompts and task_type in config.task_type_prompts:
            prompt = load_prompt_template(
                config.task_type_prompts[task_type], **template_context
            )
        else:
            prompt = ""
    else:
        prompt = load_prompt_template(config.prompt, **template_context)
        if config.task_type_prompts and task_type in config.task_type_prompts:
            overlay = load_prompt_template(
                config.task_type_prompts[task_type], **template_context
            )
            prompt = prompt + "\n\n" + overlay

    max_steps = config.execution.max_steps
    budget_hook = _build_budget_hook(max_steps)

    if config.compression is not None:
        compression_hook = build_compression_hook(config.compression)
        pre_model_hook = _chain_hooks(budget_hook, compression_hook)
    else:
        pre_model_hook = budget_hook

    # Dedup layer: wrap tools + hook (between compression and llm_input)
    if config.execution.dedup is not None and config.execution.dedup.enabled:
        from agentm.agents.dedup import (
            DedupTracker,
            build_dedup_hook,
            wrap_tool_with_dedup,
        )

        dedup_tracker = DedupTracker(
            max_cache_size=config.execution.dedup.max_cache_size,
        )
        tools = [wrap_tool_with_dedup(t, dedup_tracker) for t in tools]
        dedup_hook = build_dedup_hook(dedup_tracker)
        pre_model_hook = _chain_hooks(pre_model_hook, dedup_hook)

    if trajectory is not None:
        from agentm.agents.hooks import build_llm_input_hook

        llm_input_hook = build_llm_input_hook(
            trajectory,
            ["orchestrator", agent_id],
            task_id=task_id,
        )
        pre_model_hook = _chain_hooks(pre_model_hook, llm_input_hook)

    agent_kwargs: dict[str, Any] = dict(
        model=model,
        tools=tools,
        prompt=prompt,
        name=agent_id,
        pre_model_hook=pre_model_hook,
        response_format=ANSWER_SCHEMA[task_type],
    )
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    graph = create_react_agent(**agent_kwargs)

    # Retry structured output parsing on validation failures.
    # Pydantic ValidationError is a ValueError subclass, which
    # default_retry_on excludes — so we use a custom retry_on.
    sr_node = graph.nodes.get("generate_structured_response")
    if sr_node is not None:
        sr_node.retry_policy = [
            RetryPolicy(
                max_attempts=3,
                retry_on=lambda exc: isinstance(exc, (ValueError, TypeError)),
            )
        ]

    return graph


def _build_budget_hook(max_steps: int) -> Any:
    """Build a pre_model_hook that injects remaining-step awareness.

    Only injects a message when the budget is running low (last third).
    No message is injected when steps are plentiful — avoids wasting tokens.
    """

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        # Count AI messages that invoke real tools (not think).
        # Think tool calls are free and should not consume budget.
        step = 0
        for m in messages:
            if getattr(m, "type", "") != "ai":
                continue
            tool_calls = getattr(m, "tool_calls", None)
            if not tool_calls or any(tc.get("name") != "think" for tc in tool_calls):
                step += 1
        remaining = max(0, max_steps - step)

        if remaining <= 3:
            urgency = (
                f"WARNING: You have {remaining} steps remaining out of {max_steps}. "
                f"You MUST summarize your findings NOW and produce your final report. "
                f"Do NOT call any more tools — write your conclusion immediately."
            )
        elif remaining <= max_steps // 3:
            urgency = (
                f"BUDGET: {remaining}/{max_steps} steps remaining. "
                f"Start wrapping up — prioritize the most important remaining queries, "
                f"then produce your summary."
            )
        else:
            # Budget is plentiful — no injection needed
            return {"messages": messages}

        budget_msg = HumanMessage(content=urgency)
        return {"messages": [*messages, budget_msg]}

    return hook


def _chain_hooks(*hooks: Any) -> Any:
    """Chain multiple pre_model_hooks — each receives the output of the previous."""

    def chained(state: dict[str, Any]) -> dict[str, Any]:
        result = state
        for hook in hooks:
            result = hook(result)
        return result

    return chained


class AgentPool:
    """Factory for worker agents — creates a fresh compiled subgraph per dispatch.

    Sub-Agents are NOT added as graph nodes. They are compiled independently
    and managed by the TaskManager, which launches them as asyncio.Tasks.

    Each call to ``create_worker`` compiles a new subgraph with the caller's
    *agent_id* and optional *task_id* baked into the hooks, eliminating the
    need for mutable path overrides and preventing state leakage between
    dispatches.
    """

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        tool_registry: ToolRegistry,
        model_config: ModelConfig | None = None,
        trajectory: TrajectoryCollector | None = None,
        checkpointer: Any = None,
    ) -> None:
        self._worker_config = scenario_config.agents["worker"]
        self._tool_registry = tool_registry
        self._model_config = model_config
        self._trajectory = trajectory
        self._checkpointer = checkpointer

    @property
    def worker_max_steps(self) -> int:
        """Max tool-call steps configured for the worker agent."""
        return self._worker_config.execution.max_steps

    def create_worker(
        self,
        agent_id: str,
        task_type: str,
        task_id: str | None = None,
    ) -> Any:
        """Create a fresh compiled worker agent subgraph.

        Selects the implementation based on ``execution.subgraph_mode``:
        - ``"react"`` (default): create_react_agent (react/sub_agent.py)
        - ``"node"``: manually-controlled 4-node graph (node/worker.py)

        Each call compiles a new subgraph so that *agent_id* and *task_id*
        are baked into the hooks — no shared state between dispatches.
        """
        mode = self._worker_config.execution.subgraph_mode
        if mode == "node":
            from agentm.agents.node.worker import build_worker_subgraph

            return build_worker_subgraph(
                agent_id=agent_id,
                config=self._worker_config,
                tool_registry=self._tool_registry,
                task_type=task_type,  # type: ignore[arg-type]
                model_config=self._model_config,
                trajectory=self._trajectory,
                task_id=task_id,
                checkpointer=self._checkpointer,
            )
        return create_sub_agent(
            agent_id,
            self._worker_config,
            self._tool_registry,
            task_type,
            self._model_config,
            self._trajectory,
            task_id=task_id,
            checkpointer=self._checkpointer,
        )
