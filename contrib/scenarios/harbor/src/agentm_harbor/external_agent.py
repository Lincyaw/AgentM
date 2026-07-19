"""Harbor external agent — AgentM runs locally, tool calls in sandbox.

Usage::

    harbor trial start -p <task> \\
        --agent-import-path agentm_harbor:ExternalAgentMAgent \\
        --ae AGENTM_MODEL=doubao \\
        --ae AGENTM_API_KEY=... \\
        --ae AGENTM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

Trajectory is managed by AgentM's own observability layer (OTLP / JSONL),
not Harbor's ATIF format.  Inspect with ``agentm trace``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from loguru import logger

from agentm_harbor.harbor_ops import set_harbor_environment

SCENARIO = "arl:harbor"


class ExternalAgentMAgent(BaseAgent):
    """AgentM as a Harbor external agent.

    The agent process runs on the host; bash and file tool calls route
    through Harbor's ``BaseEnvironment`` into the sandbox container.
    """

    def __init__(self, logs_dir: Path, *args: Any, **kwargs: Any) -> None:
        self._reasoning_effort: str | None = kwargs.pop("reasoning_effort", None)
        super().__init__(logs_dir, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "agentm-external"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from agentm import AgentSession, AgentSessionConfig

        set_harbor_environment(environment)

        env_patch = dict(self.extra_env)
        api_key = env_patch.get("AGENTM_API_KEY", os.environ.get("AGENTM_API_KEY", ""))
        if api_key:
            env_patch.setdefault("OPENAI_API_KEY", api_key)
        base_url = env_patch.get("AGENTM_BASE_URL", os.environ.get("AGENTM_BASE_URL", ""))
        if base_url:
            env_patch.setdefault("OPENAI_BASE_URL", base_url)
        if self._reasoning_effort:
            env_patch.setdefault("AGENTM_REASONING_EFFORT", self._reasoning_effort)

        saved: dict[str, str | None] = {}
        for key, val in env_patch.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = val

        model = self.model_name or env_patch.get("AGENTM_MODEL") or os.environ.get("AGENTM_MODEL")
        config = AgentSessionConfig(
            cwd=os.getcwd(),
            model=model,
            scenario=SCENARIO,
        )

        session = await AgentSession.create(config)
        logger.info("agentm-external: session {} started", session.session_id)

        try:
            messages = await session.prompt(instruction)
        finally:
            await session.shutdown()
            for key, prev in saved.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

        tool_calls = sum(
            len(m.content)
            for m in messages
            if m.role == "tool_result"
        )
        logger.info(
            "agentm-external: session {} done, {} tool calls",
            session.session_id, tool_calls,
        )

        self._populate_token_counts(context, session.session_id)

    def _populate_token_counts(self, context: AgentContext, session_id: str) -> None:
        try:
            from agentm.core.observability import clickhouse as ch

            url = ch.get_url()
            if url is None:
                return
            usage = ch.usage(url, session_id) or {}
            context.n_input_tokens = usage.get("input_tokens", 0)
            context.n_output_tokens = usage.get("output_tokens", 0)
            context.n_cache_tokens = (
                usage.get("cache_read", 0) + usage.get("cache_write", 0)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("token count lookup failed: {}", exc)
