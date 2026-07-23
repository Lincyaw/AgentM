# code-health: ignore-file[AM021,AM025] -- Harbor exposes optional ARL metadata dynamically
"""Harbor external agent backed by an embedded AgentM SDK session."""

from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import agentm_toolbox
from agentm import (
    AgentSession,
    AgentSessionConfig,
    ScenarioLoader,
    ScenarioSpec,
    load_scenario_manifest,
)
from agentm.config import DefaultSessionSpecResolver
from agentm.control import SessionControlServer
from agentm.core.abi.roles import (
    RESOURCE_WRITER,
    bind_environment_operations,
    bind_resource_store,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.storage.resources import LocalResourceStore
from agentm.storage.trajectory import resolve_trajectory_store_or_create
from agentm_toolbox import (
    REMOTE_DEPENDENCIES,
    REMOTE_TOOLBOX_COMMAND,
    REMOTE_TOOLBOX_ROOT,
    ToolboxDependency,
)
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from loguru import logger

from agentm_harbor.harbor_ops import HarborOpsConfig, harbor_bindings

SCENARIO = "arl:harbor"
_TOOLBOX_SETUP_TIMEOUT = 300


def _toolbox_setup_command(dependency: ToolboxDependency) -> str:
    executable = shlex.quote(dependency.executable)
    requirement = shlex.quote(dependency.requirement)
    return (
        f"if command -v {executable} >/dev/null 2>&1; then exit 0; fi; "
        "agentm_pip_break_flag=''; "
        "if python3 -m pip install --help 2>&1 "
        "| grep -q -- '--break-system-packages'; then "
        "agentm_pip_break_flag='--break-system-packages'; fi; "
        "python3 -m pip install --quiet --disable-pip-version-check "
        f"$agentm_pip_break_flag {requirement}; "
        f"if ! command -v {executable} >/dev/null 2>&1; then "
        "agentm_scripts_dir=$(python3 -c "
        "'import sysconfig; print(sysconfig.get_path(\"scripts\"))'); "
        f'ln -sf "$agentm_scripts_dir"/{executable} '
        f"/usr/local/bin/{executable}; fi; "
        f"command -v {executable} >/dev/null 2>&1"
    )


async def _provision_remote_toolbox(environment: BaseEnvironment) -> None:
    for dependency in REMOTE_DEPENDENCIES:
        result: ExecResult = await environment.exec(
            _toolbox_setup_command(dependency),
            cwd="/",
            timeout_sec=_TOOLBOX_SETUP_TIMEOUT,
        )
        if result.return_code == 0:
            continue
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(
            f"could not provision toolbox dependency {dependency.requirement}: {detail[:500]}"
        )

    package_source = Path(agentm_toolbox.__file__).resolve()
    package_source = package_source.parent
    package_target = f"{REMOTE_TOOLBOX_ROOT}/agentm_toolbox"
    prepare = await environment.exec(
        f"mkdir -p -- {shlex.quote(REMOTE_TOOLBOX_ROOT)}",
        cwd="/",
        timeout_sec=_TOOLBOX_SETUP_TIMEOUT,
    )
    if prepare.return_code != 0:
        detail = (prepare.stderr or prepare.stdout or "unknown error").strip()
        raise RuntimeError(f"could not prepare remote toolbox: {detail[:500]}")
    await environment.upload_dir(package_source, package_target)
    verify = await environment.exec(
        f"{REMOTE_TOOLBOX_COMMAND} repository-index --help >/dev/null",
        cwd="/",
        timeout_sec=_TOOLBOX_SETUP_TIMEOUT,
    )
    if verify.return_code != 0:
        detail = (verify.stderr or verify.stdout or "unknown error").strip()
        raise RuntimeError(f"could not activate remote toolbox: {detail[:500]}")


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


def _resume_session_id(context: AgentContext) -> str:
    metadata = context.metadata or {}
    session_id = metadata.get("agentm_session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("AgentM Harbor resume requires context.metadata['agentm_session_id']")
    return session_id


def _fork_request(env: Mapping[str, str]) -> tuple[str, int] | None:
    source_session_id = env.get("AGENTM_FORK_FROM_SESSION", "").strip()
    raw_turn = env.get("AGENTM_FORK_TURN", "").strip()
    if not source_session_id:
        if raw_turn:
            raise ValueError("AGENTM_FORK_TURN requires AGENTM_FORK_FROM_SESSION")
        return None
    if not raw_turn:
        raise ValueError("AGENTM_FORK_FROM_SESSION requires AGENTM_FORK_TURN")
    try:
        turn = int(raw_turn)
    except ValueError as exc:
        raise ValueError("AGENTM_FORK_TURN must be a non-negative integer") from exc
    if turn < 0:
        raise ValueError("AGENTM_FORK_TURN must be a non-negative integer")
    return source_session_id, turn


def _sync_execution_metadata(
    context: AgentContext,
    environment: BaseEnvironment,
    *,
    agentm_session_id: str | None = None,
    agentm_parent_session_id: str | None = None,
    agentm_fork_turn: int | None = None,
) -> None:
    metadata = dict(context.metadata or {})
    if agentm_session_id is not None:
        metadata["agentm_session_id"] = agentm_session_id
    if agentm_parent_session_id is not None:
        metadata["agentm_parent_session_id"] = agentm_parent_session_id
    if agentm_fork_turn is not None:
        metadata["agentm_fork_turn"] = agentm_fork_turn

    arl = getattr(environment, "arl", None)
    arl_session_id = getattr(arl, "session_id", None)
    if not isinstance(arl_session_id, str) or not arl_session_id:
        context.metadata = metadata or None
        return

    previous_arl_session_id = metadata.get("arl_session_id")
    metadata["arl_session_id"] = arl_session_id

    parent_session_id = getattr(arl, "parent_session_id", None)
    if isinstance(parent_session_id, str) and parent_session_id:
        metadata["arl_parent_session_id"] = parent_session_id
        fork_step = getattr(arl, "fork_step", None)
        if isinstance(fork_step, int) and not isinstance(fork_step, bool):
            metadata["arl_fork_step"] = fork_step
    else:
        metadata.pop("arl_parent_session_id", None)
        metadata.pop("arl_fork_step", None)

    steps = getattr(arl, "steps", None)
    if isinstance(steps, list) and steps:
        step_index = getattr(steps[-1], "step_index", None)
        if isinstance(step_index, int) and not isinstance(step_index, bool):
            metadata["arl_step"] = step_index
    elif previous_arl_session_id != arl_session_id:
        # A fresh or forked ARL session has no checkpoint of its own yet.
        metadata.pop("arl_step", None)

    context.metadata = metadata


class ExternalAgentMAgent(BaseAgent):
    """Run AgentM locally while Harbor owns the sandbox lifecycle."""

    SUPPORTS_RESUME = True

    @staticmethod
    def name() -> str:
        return "agentm-external"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        await _provision_remote_toolbox(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self._run_session(
            instruction,
            environment,
            context,
            resume_session_id=None,
        )

    async def resume(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self._run_session(
            instruction,
            environment,
            context,
            resume_session_id=_resume_session_id(context),
        )

    async def _run_session(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
        *,
        resume_session_id: str | None,
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
        fork_request = _fork_request(effective_env)
        fork_prompt = effective_env.get("AGENTM_FORK_PROMPT", "").strip()
        if resume_session_id is not None and fork_request is not None:
            raise ValueError("AgentM Harbor cannot resume and fork in the same run")
        if fork_request is None and fork_prompt:
            raise ValueError("AGENTM_FORK_PROMPT requires AGENTM_FORK_FROM_SESSION")
        scenario_path = _find_scenario_yaml(effective_env.get("AGENTM_SCENARIO_YAML"))
        spec_resolver = DefaultSessionSpecResolver(
            user_config=_user_config_path(effective_env),
            env=effective_env,
        )
        trajectory = resolve_trajectory_store_or_create(
            str(self.logs_dir),
            env=effective_env,
        )
        resource_store = LocalResourceStore(
            workspace_root=self.logs_dir,
            root=self.logs_dir / "resources",
            discover_manifest=False,
        )
        operations, writer = harbor_bindings(
            environment,
            HarborOpsConfig(work_dir="/"),
        )
        host_services = ServiceRegistry()
        bind_resource_store(host_services, resource_store)
        host_services.bind(RESOURCE_WRITER, writer)
        bind_environment_operations(host_services, operations)
        session_config = AgentSessionConfig(
            cwd="/",
            scenario=SCENARIO,
            scenario_loader=_scenario_loader(scenario_path),
            spec_resolver=spec_resolver,
            trajectory_store=trajectory.store,
        )
        _sync_execution_metadata(context, environment)
        fork_source: AgentSession | None = None
        try:
            if resume_session_id is not None:
                session = await AgentSession.resume(
                    resume_session_id,
                    trajectory.store,
                    session_config,
                    host_services=host_services,
                )
            elif fork_request is not None:
                source_session_id, fork_turn = fork_request
                source_meta, source_turns = await asyncio.to_thread(
                    trajectory.store.load_prefix,
                    source_session_id,
                    fork_turn,
                )
                root_session_id = source_meta.config.get("root_session_id")
                if not isinstance(root_session_id, str) or not root_session_id:
                    raise ValueError("fork source metadata has no valid root_session_id")
                fork_source = await AgentSession.create(
                    replace(
                        session_config,
                        purpose="harbor-fork-source",
                        session_id=source_session_id,
                        root_session_id=root_session_id,
                        parent_session_id=None,
                        initial_turns=source_turns,
                    ),
                    host_services=host_services,
                )
                session = await AgentSession.fork(
                    fork_source,
                    at=fork_turn,
                    purpose="harbor-fork",
                )
            else:
                session = await AgentSession.create(session_config, host_services=host_services)
        except BaseException as creation_error:
            cleanup_errors: list[BaseException] = []
            if fork_source is not None:
                try:
                    await fork_source.shutdown()
                except BaseException as source_shutdown_error:
                    cleanup_errors.append(source_shutdown_error)
            try:
                trajectory.close()
            except Exception as close_error:
                cleanup_errors.append(close_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "AgentM session creation and cleanup failed",
                    (creation_error, *cleanup_errors),
                ) from creation_error
            raise

        fork_parent_session_id = session.ctx.parent_session_id
        selected_fork_turn = fork_request[1] if fork_request is not None else None
        _sync_execution_metadata(
            context,
            environment,
            agentm_session_id=session.session_id,
            agentm_parent_session_id=fork_parent_session_id,
            agentm_fork_turn=selected_fork_turn,
        )
        errors: list[BaseException] = []
        turns = session.get_turns()
        try:
            interrupt = SessionControlServer(session)
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
            if fork_parent_session_id is not None and selected_fork_turn is not None:
                logger.info(
                    "agentm-external: session {} forked from {} at turn {}",
                    session.session_id,
                    fork_parent_session_id,
                    selected_fork_turn,
                )
            await session.run(fork_prompt or instruction)
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
            _sync_execution_metadata(
                context,
                environment,
                agentm_session_id=session.session_id,
                agentm_parent_session_id=fork_parent_session_id,
                agentm_fork_turn=selected_fork_turn,
            )
            try:
                await session.shutdown()
            except BaseException as shutdown_error:
                errors.append(shutdown_error)
            if fork_source is not None:
                try:
                    await fork_source.shutdown()
                except BaseException as source_shutdown_error:
                    errors.append(source_shutdown_error)
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
