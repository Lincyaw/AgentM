"""WorkerLoopFactory -- creates SimpleAgentLoop instances for worker agents.

Replaces the legacy AgentPool + build_worker_subgraph combination.  All
wiring (tools, system prompt, middleware, output schema) is assembled here
and injected into SimpleAgentLoop.

No LangGraph dependency.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from agentm.config.schema import AgentConfig, ModelConfig, ScenarioConfig, create_chat_model
from agentm.core.prompt import load_prompt_template
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.harness.loops.simple import SimpleAgentLoop
from agentm.harness.middleware import (
    BudgetMiddleware,
    CompressionMiddleware,
    DedupMiddleware,
    LoopDetectionMiddleware,
    TrajectoryMiddleware,
)
from agentm.harness.tool import Tool

logger = logging.getLogger(__name__)


class WorkerLoopFactory:
    """Factory that produces configured SimpleAgentLoop instances for workers.

    Each call to ``create_worker`` builds a fresh loop with the correct
    tools, system prompt, middleware stack, and output schema for the
    given task type.
    """

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        tool_registry: ToolRegistry,
        model_config: ModelConfig | None = None,
        *,
        extra_tools: list[Tool] | None = None,
        extra_middleware: list[Any] | None = None,
        trajectory: TrajectoryCollector | None = None,
        answer_schemas: dict[str, type[BaseModel]] | None = None,
    ) -> None:
        self._worker_config: AgentConfig = scenario_config.agents["worker"]
        self._tool_registry = tool_registry
        self._model_config = model_config
        self._extra_tools: list[Tool] = extra_tools or []
        self._extra_middleware = extra_middleware or []
        self._trajectory = trajectory
        self._answer_schemas: dict[str, type[BaseModel]] | None = answer_schemas

    @property
    def worker_max_steps(self) -> int:
        return self._worker_config.execution.max_steps

    @property
    def worker_timeout(self) -> int:
        return self._worker_config.execution.timeout

    def create_worker(self, agent_id: str, task_type: str) -> SimpleAgentLoop:
        """Build a fully configured SimpleAgentLoop for a worker dispatch."""
        config = self._worker_config

        # -- Tools --
        tools = self._build_tools(config)
        tools_description = "\n".join(f"- `{t.name}`: {t.description}" for t in tools)

        # -- System prompt --
        system_prompt = self._build_system_prompt(
            config, agent_id, task_type, tools_description
        )

        # -- Middleware --
        middleware = self._build_middleware(config, agent_id)

        # -- Output schema --
        # answer_schemas injected from ScenarioWiring via builder
        output_schema = (
            self._answer_schemas.get(task_type)
            if self._answer_schemas is not None
            else None
        )

        # -- Model --
        model = create_chat_model(
            model=config.model,
            temperature=config.temperature,
            model_config=self._model_config,
        )
        model_with_tools = model.bind_tools([t.to_openai_schema() for t in tools])

        # -- Retry config --
        retry_cfg = config.execution.retry

        logger.info(
            "WorkerLoopFactory: created worker %s (task_type=%s, tools=%d, middleware=%d)",
            agent_id,
            task_type,
            len(tools),
            len(middleware),
        )

        return SimpleAgentLoop(
            model=model_with_tools,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
            output_schema=output_schema,
            retry_max_attempts=retry_cfg.max_attempts,
            retry_initial_interval=retry_cfg.initial_interval,
            retry_backoff_factor=retry_cfg.backoff_factor,
        )

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_tools(self, config: AgentConfig) -> list[Tool]:
        """Assemble the tool list: registry tools + extra + think."""
        tools: list[Tool] = [
            self._tool_registry.get(name).create_tool(
                **config.tool_settings.get(name, {})
            )
            for name in config.tools
        ]
        if self._extra_tools:
            tools.extend(self._extra_tools)
            logger.info(
                "WorkerLoopFactory: extra tools: %s",
                [t.name for t in self._extra_tools],
            )
        if self._worker_config.include_think_tool:
            from agentm.tools.think import think
            tools.append(think)
        return tools

    def _build_system_prompt(
        self,
        config: AgentConfig,
        agent_id: str,
        task_type: str,
        tools_description: str,
    ) -> str:
        """Build the system prompt: base template + task_type overlay via Jinja2."""
        template_context = {
            "agent_id": agent_id,
            "tools_description": tools_description,
        }

        if config.prompt is None:
            base_prompt = ""
        else:
            base_prompt = load_prompt_template(
                config.prompt, base_dir=None, **template_context
            )

        if config.task_type_prompts and task_type in config.task_type_prompts:
            overlay = load_prompt_template(
                config.task_type_prompts[task_type],
                base_dir=None,
                **template_context,
            )
            system_prompt = (
                (base_prompt + "\n\n" + overlay).strip() if base_prompt else overlay
            )
        else:
            system_prompt = base_prompt

        return system_prompt

    def _build_middleware(
        self, config: AgentConfig, agent_id: str
    ) -> list[Any]:
        """Assemble the middleware stack in execution order."""
        middleware: list[Any] = []

        # Extra middleware (injected by caller) goes first
        if self._extra_middleware:
            middleware.extend(self._extra_middleware)

        # Budget
        budget_mw = BudgetMiddleware(
            config.execution.max_steps,
            tool_call_budget=config.execution.tool_call_budget,
        )
        middleware.append(budget_mw)

        # Loop detection
        ld = config.execution.loop_detection
        middleware.append(LoopDetectionMiddleware(
            threshold=ld.threshold,
            window_size=ld.window_size,
            think_stall_limit=ld.think_stall_limit,
        ))

        # Compression
        if config.compression is not None:
            middleware.append(
                CompressionMiddleware(
                    config.compression, model_config=self._model_config
                )
            )

        # Trajectory
        if self._trajectory is not None:
            middleware.append(
                TrajectoryMiddleware(
                    self._trajectory,
                    agent_path=["orchestrator", agent_id],
                )
            )

        # Dedup
        if config.execution.dedup is not None and config.execution.dedup.enabled:
            middleware.append(
                DedupMiddleware(
                    max_cache_size=config.execution.dedup.max_cache_size
                )
            )

        return middleware
