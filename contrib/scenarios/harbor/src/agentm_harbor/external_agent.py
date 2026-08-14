# code-health: ignore-file[AM021,AM025] -- Harbor exposes optional ARL metadata dynamically
"""Harbor external agent backed by an embedded AgentM SDK session."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
import uuid
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
from agentm.core.abi.events import TurnCommittedEvent
from agentm.core.abi.roles import (
    RESOURCE_WRITER,
    bind_environment_operations,
    bind_resource_store,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.store import SessionMeta, TrajectoryDiagnostic, TrajectoryStore
from agentm.core.abi.trajectory import Turn
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

#: The composition a dispatched child gets. Named here rather than left to the
#: caller: the model has to guess it otherwise, and a wrong guess comes back as
#: an infrastructure error it cannot act on.
CRITIC_SCENARIO = "arl:harbor-critic"
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
    """Resolve a scenario name to the manifest that declares it.

    One name per file, so a second composition needs a second file and a way to
    reach it. Without this the loader answered every request with the run's own
    manifest, which meant a child session could only ever be another copy of the
    agent under test -- and a copy of the agent is the one reviewer whose
    blind spots are guaranteed to match.
    """

    manifests = {CRITIC_SCENARIO: path.parent / "critic.yaml"}

    def load(scenario: str) -> ScenarioSpec:
        return load_scenario_manifest(
            manifests.get(scenario, path),
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


#: Diagnostic phase under which each committed turn records the sandbox
#: checkpoint that existed when it ended. Two independent numbers describe one
#: run -- an AgentM turn index and an ARL step -- and forking needs both. Their
#: correspondence is knowable only while the run is happening, so it is written
#: down then; a fork that had to guess it produced an agent whose history
#: described a filesystem it was not given.
_STEP_PHASE = "arl-checkpoint"


def _arl_step(environment: BaseEnvironment) -> tuple[str, int] | None:
    """The sandbox session and its newest checkpoint, when there is one."""
    arl = getattr(environment, "arl", None)
    session_id = getattr(arl, "session_id", None)
    if not isinstance(session_id, str) or not session_id:
        return None
    steps = getattr(arl, "steps", None)
    if not isinstance(steps, list) or not steps:
        return None
    step_index = getattr(steps[-1], "step_index", None)
    if not isinstance(step_index, int) or isinstance(step_index, bool):
        return None
    return session_id, step_index


def _record_environment_steps(
    session: AgentSession,
    environment: BaseEnvironment,
    store: TrajectoryStore,
) -> None:
    """Write down which sandbox checkpoint each turn ended at.

    Costs one row per turn and buys the only thing that makes a two-sided fork
    checkable: without it, ``AGENTM_FORK_TURN`` and ``fork_step`` are two
    numbers from different counting systems that a person lines up by hand,
    and lining them up wrong fails silently.
    """

    async def _on_committed(event: TurnCommittedEvent) -> None:
        turn = event.turn
        if turn is None:
            return
        current = _arl_step(environment)
        if current is None:
            return
        arl_session_id, step_index = current
        try:
            await asyncio.to_thread(
                store.append_diagnostic,
                TrajectoryDiagnostic(
                    id=uuid.uuid4().hex,
                    session_id=session.session_id,
                    timestamp=time.time(),
                    level="info",
                    source="agentm-harbor",
                    phase=_STEP_PHASE,
                    message=json.dumps({"arl_session_id": arl_session_id, "arl_step": step_index}),
                    turn_index=turn.index,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a lost mapping is not a lost run
            logger.warning(
                "agentm-external: could not record sandbox step for turn {}: {}",
                turn.index,
                exc,
            )

    session.on(TurnCommittedEvent.CHANNEL, _on_committed)


async def _load_fork_prefix(
    store: TrajectoryStore,
    source_session_id: str,
    fork_turn: int,
) -> tuple[SessionMeta, list[Turn]]:
    """The source's turns up to and including ``fork_turn``.

    Both failures here are things an operator types by hand, so both say what
    was asked for and what exists instead. Unwrapped, a mistyped turn arrives
    as ``KeyError: 41`` from three layers down.
    """
    try:
        return await asyncio.to_thread(store.load_prefix, source_session_id, fork_turn)
    except KeyError as exc:
        try:
            _, turns = await asyncio.to_thread(store.load, source_session_id)
        except KeyError:
            raise ValueError(
                f"no session {source_session_id} in this trajectory store -- "
                "AGENTM_TRAJECTORY_DSN and AGENTM_TRAJECTORY_SCHEMA must name "
                "the store the source run wrote to"
            ) from exc
        if not turns:
            raise ValueError(
                f"session {source_session_id} has no committed turns to fork from"
            ) from exc
        raise ValueError(
            f"session {source_session_id} has no turn {fork_turn}; "
            f"its committed turns run {turns[0].index}..{turns[-1].index}"
        ) from exc


def _check_environment_alignment(
    store: TrajectoryStore,
    environment: BaseEnvironment,
    *,
    source_session_id: str,
    fork_turn: int,
) -> None:
    """Refuse a fork whose conversation and filesystem are from different moments.

    The two halves are requested separately -- ``AGENTM_FORK_TURN`` here, and
    ``fork_from``/``fork_step`` on the environment -- and nothing has connected
    them until now. A mismatch does not fail: the run proceeds with an agent
    that remembers writing files the sandbox never received, which reads as a
    confused agent rather than as a bad launch.
    """
    arl = getattr(environment, "arl", None)
    environment_source = getattr(arl, "parent_session_id", None)
    environment_step = getattr(arl, "fork_step", None)
    if not isinstance(environment_source, str) or not environment_source:
        logger.warning(
            "agentm-external: forking the trajectory of {} at turn {} into a "
            "sandbox that was not forked -- the agent's history will describe "
            "files this environment does not have",
            source_session_id,
            fork_turn,
        )
        return
    if not isinstance(environment_step, int) or isinstance(environment_step, bool):
        return

    recorded = _recorded_step(store, source_session_id, fork_turn)
    if recorded is None:
        logger.warning(
            "agentm-external: no recorded sandbox checkpoint for {} turn {}, so "
            "its alignment with fork_step={} cannot be checked; the source run "
            "predates step recording",
            source_session_id,
            fork_turn,
            environment_step,
        )
        return
    recorded_session_id, recorded_step = recorded
    if recorded_session_id != environment_source:
        raise ValueError(
            f"fork mismatch: turn {fork_turn} of {source_session_id} was run in "
            f"sandbox {recorded_session_id}, but the environment was forked from "
            f"{environment_source}"
        )
    if recorded_step != environment_step:
        raise ValueError(
            f"fork mismatch: turn {fork_turn} of {source_session_id} ended at "
            f"sandbox step {recorded_step}, but the environment was forked at "
            f"step {environment_step} -- pass --ek fork_step={recorded_step}"
        )


def _recorded_step(
    store: TrajectoryStore,
    session_id: str,
    turn_index: int,
) -> tuple[str, int] | None:
    """The sandbox checkpoint a turn ended at, as recorded while it ran."""
    try:
        diagnostics = store.list_diagnostics(session_id)
    except Exception as exc:  # noqa: BLE001 - an unreadable log is not a mismatch
        logger.warning("agentm-external: cannot read {} diagnostics: {}", session_id, exc)
        return None
    for diagnostic in reversed(diagnostics):
        if diagnostic.phase != _STEP_PHASE or diagnostic.turn_index != turn_index:
            continue
        try:
            payload = json.loads(diagnostic.message)
        except ValueError as exc:
            logger.warning("agentm-external: unreadable step record: {}", exc)
            return None
        arl_session_id = payload.get("arl_session_id")
        arl_step = payload.get("arl_step")
        if isinstance(arl_session_id, str) and isinstance(arl_step, int):
            return arl_session_id, arl_step
    return None


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
                source_meta, source_turns = await _load_fork_prefix(
                    trajectory.store,
                    source_session_id,
                    fork_turn,
                )
                root_session_id = source_meta.config.get("root_session_id")
                if not isinstance(root_session_id, str) or not root_session_id:
                    raise ValueError(
                        f"session {source_session_id} cannot be forked: its stored "
                        "metadata has no root_session_id, which happens when it was "
                        "written by an SDK older than session metadata version 1"
                    )
                _check_environment_alignment(
                    trajectory.store,
                    environment,
                    source_session_id=source_session_id,
                    fork_turn=fork_turn,
                )
                # Built in one step, from a prefix that already ends at
                # ``fork_turn`` -- ``load_prefix`` is inclusive of it.
                #
                # A resumed attempt is a continuation, not a subordinate, so it
                # carries no parent. Nothing distinguishes a run with a parent
                # from a dispatched subagent, and every atom that asks "am I a
                # subagent" reads that pointer -- ``sub_agent`` asks it to stop
                # children spawning children, so a resumed run marked as a child
                # silently has no ``dispatch_agent``.
                #
                # Where the run came from is recorded in the trial metadata
                # below, which is where the readers of that fact already look.
                session = await AgentSession.create(
                    replace(
                        session_config,
                        purpose="harbor-fork",
                        root_session_id=root_session_id,
                        parent_session_id=None,
                        fork_source_session_id=source_session_id,
                        fork_point=fork_turn,
                        initial_turns=source_turns,
                    ),
                    host_services=host_services,
                )
            else:
                session = await AgentSession.create(session_config, host_services=host_services)
        except BaseException as creation_error:
            try:
                trajectory.close()
            except Exception as close_error:
                raise BaseExceptionGroup(
                    "AgentM session creation and cleanup failed",
                    (creation_error, close_error),
                ) from creation_error
            raise

        # The origin is reported from the request, not from the session's own
        # parent pointer, which a resumed run does not set.
        fork_parent_session_id = fork_request[0] if fork_request is not None else None
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
        _record_environment_steps(session, environment, trajectory.store)
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
