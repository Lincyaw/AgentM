"""Harbor external agent backed by an embedded AgentM SDK session."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from agentm import (
    AgentSession,
    AgentSessionConfig,
    ScenarioLoader,
    ScenarioSpec,
    load_scenario_manifest,
)
from agentm.config import DefaultSessionSpecResolver
from agentm.storage.trajectory import resolve_trajectory_store_or_create
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from loguru import logger

from agentm_harbor.harbor_ops import HarborOpsConfig, harbor_bindings
from agentm_harbor.human_interrupt import HumanInterruptServer

SCENARIO = "arl:harbor"


def _find_scenario_yaml(configured: str | None = None) -> Path:
    if configured:
        return Path(configured).expanduser()
    package_dir = Path(__file__).parent
    for candidate in (
        package_dir / "scenario.yaml",
        package_dir.parents[1] / "scenario.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("scenario.yaml not found in agentm_harbor package")


def _load_scenario(scenario: str) -> ScenarioSpec:
    return load_scenario_manifest(
        _find_scenario_yaml(),
        requested_name=scenario,
    )


def _scenario_loader(path: Path) -> ScenarioLoader:
    def load(scenario: str) -> ScenarioSpec:
        return load_scenario_manifest(
            path,
            requested_name=scenario,
        )

    return load


def _effective_env(
    process_env: Mapping[str, str],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    values = dict(process_env)
    values.update(overrides)
    return values


def _user_config_path(env: Mapping[str, str]) -> Path:
    home = env.get("AGENTM_HOME")
    if home:
        return Path(home).expanduser() / "config.toml"
    return Path.home() / ".agentm" / "config.toml"


def _raise_run_errors(errors: list[BaseException]) -> None:
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    raise BaseExceptionGroup(
        "AgentM Harbor run failed during multiple lifecycle phases",
        errors,
    )


class ExternalAgentMAgent(BaseAgent):
    """Run AgentM locally while Harbor owns the sandbox lifecycle."""

    @staticmethod
    def name() -> str:
        return "agentm-external"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        effective_env = _effective_env(os.environ, self.extra_env)
        if "AGENTM_API_KEY" not in effective_env:
            openai_api_key = effective_env.get("OPENAI_API_KEY")
            if openai_api_key:
                effective_env["AGENTM_API_KEY"] = openai_api_key
        if "AGENTM_BASE_URL" not in effective_env:
            openai_base_url = effective_env.get("OPENAI_BASE_URL")
            if openai_base_url:
                effective_env["AGENTM_BASE_URL"] = openai_base_url
        if self.model_name:
            effective_env["AGENTM_PROVIDER"] = self.model_name
        scenario_path = _find_scenario_yaml(effective_env.get("AGENTM_SCENARIO_YAML"))
        spec_resolver = DefaultSessionSpecResolver(
            user_config=_user_config_path(effective_env),
            env=effective_env,
        )
        trajectory = resolve_trajectory_store_or_create(
            str(self.logs_dir),
            env=effective_env,
        )
        operations, writer = harbor_bindings(
            environment,
            HarborOpsConfig(work_dir="/"),
        )
        try:
            session = await AgentSession.create(
                AgentSessionConfig(
                    cwd="/",
                    scenario=SCENARIO,
                    scenario_loader=_scenario_loader(scenario_path),
                    spec_resolver=spec_resolver,
                    environment_operations=operations,
                    resource_writer=writer,
                    trajectory_store=trajectory.store,
                )
            )
        except BaseException as creation_error:
            try:
                trajectory.close()
            except Exception as close_error:
                raise BaseExceptionGroup(
                    "AgentM session creation and trajectory cleanup failed",
                    (creation_error, close_error),
                ) from creation_error
            raise

        errors: list[BaseException] = []
        turns = session.get_turns()
        try:
            interrupt = HumanInterruptServer(session)
            await interrupt.start()
            try:
                session.register_cleanup(interrupt.stop)
            except BaseException:
                await interrupt.stop()
                raise
            logger.info(
                "agentm-external: session {} started",
                session.session_id,
            )
            await session.run(instruction)
            await session.idle()
        except BaseException as run_error:
            errors.append(run_error)
            logger.error(
                "agentm-external: session {} failed: {}",
                session.session_id,
                run_error,
            )
        finally:
            turns = session.get_turns()
            context.n_input_tokens = sum(turn.meta.total_input_tokens for turn in turns)
            context.n_output_tokens = sum(turn.meta.total_output_tokens for turn in turns)
            context.n_cache_tokens = sum(
                turn.meta.cache_read_tokens + turn.meta.cache_write_tokens for turn in turns
            )
            try:
                await session.shutdown()
            except BaseException as shutdown_error:
                errors.append(shutdown_error)
            try:
                trajectory.close()
            except Exception as close_error:
                errors.append(close_error)

        if not errors:
            logger.info(
                "agentm-external: session {} completed with {} turn(s)",
                session.session_id,
                len(turns),
            )
        _raise_run_errors(errors)


__all__ = ("ExternalAgentMAgent",)
