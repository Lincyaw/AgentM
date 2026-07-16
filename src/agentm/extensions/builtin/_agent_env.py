"""ARL agent-env sandbox: shared utilities and entry point.

One sandbox per AgentM session. All config comes from :class:`AgentEnvConfig`
(scenario manifest). Connection parameters (``gateway_url``, ``api_key``) fall
through to the ARL SDK's own config chain when unset.
"""

from __future__ import annotations

import asyncio
import posixpath
from collections.abc import Callable
from inspect import isawaitable, iscoroutinefunction
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    ExtensionAPI,
    SessionShutdownEvent,
)

if TYPE_CHECKING:
    from arl import SandboxSession as ArlSandboxSession

_AGENT_ENV_SESSION_SERVICE = "agent_env.session_id"
_AGENT_ENV_EXPERIMENT_SERVICE = "agent_env.experiment_id"
_AGENT_ENV_WORK_DIR_SERVICE = "agent_env.work_dir"
_OPERATION_TIMEOUT_GRACE_SECONDS = 30.0


class AgentEnvConfig(BaseModel):
    image: str | None = None
    experiment_id: str | None = None
    attach_session: str | None = None
    gateway_url: str | None = None
    api_key: str | None = None
    profile: str | None = None
    config_env: dict[str, Any] | None = None
    work_dir: str | None = None
    timeout: float | None = None
    idle_timeout_seconds: int | None = None
    max_lifetime_seconds: int | None = None
    create_timeout: float | None = None
    cpu_request: str | None = None
    cpu_limit: str | None = None
    memory_request: str | None = None
    memory_limit: str | None = None
    delete_on_shutdown: bool | None = None
    private_containers: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _normalize_work_dir(work_dir: str) -> str:
    normalized = posixpath.normpath(work_dir)
    if not normalized.startswith("/"):
        raise ValueError(f"agent_env work_dir must be absolute, got {work_dir!r}")
    return normalized.rstrip("/") or "/"


def _sandbox_abs(work_dir: str, path: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(work_dir, path))


def _is_in_work_dir(work_dir: str, path: str) -> bool:
    abs_path = _sandbox_abs(work_dir, path)
    if work_dir == "/":
        return abs_path.startswith("/")
    return abs_path == work_dir or abs_path.startswith(work_dir + "/")


def _workspace_relative_path(work_dir: str, path: str) -> str | None:
    abs_path = _sandbox_abs(work_dir, path)
    if work_dir == "/":
        rel_path = abs_path.lstrip("/")
    elif abs_path.startswith(work_dir + "/"):
        rel_path = abs_path[len(work_dir) + 1 :]
    else:
        return None
    return rel_path or None


# ---------------------------------------------------------------------------
# ARL SDK helpers
# ---------------------------------------------------------------------------

def _step_timeout_budget(steps: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for step in steps:
        raw = step.get("timeoutSeconds", step.get("timeout"))
        if not isinstance(raw, bool) and isinstance(raw, (int, float)) and raw > 0:
            values.append(float(raw))
    if not values:
        return None
    return max(values) + _OPERATION_TIMEOUT_GRACE_SECONDS


async def _call_maybe_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sync or async SDK method without blocking the event loop."""
    if iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await asyncio.to_thread(fn, *args, **kwargs)
    if isawaitable(result):
        return await result
    return result


async def _close_client(client: Any) -> None:
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is not None:
        await _call_maybe_async(close)


def _gateway_client_class(arl_module: Any) -> Any:
    return getattr(arl_module, "GatewayClient", None) or getattr(
        arl_module, "AsyncGatewayClient"
    )


def _arl_class(arl_module: Any, *names: str) -> Any:
    for name in names:
        cls = getattr(arl_module, name, None)
        if cls is not None:
            return cls
    raise AttributeError(f"arl module has none of: {', '.join(names)}")


async def _async_execute(
    session: "ArlSandboxSession",
    steps: list[dict[str, Any]],
    *,
    on_output: Callable[[str, str], None] | None = None,
) -> Any:
    """Execute steps in the sandbox.

    ARL execute operations are idempotent, so the SDK recovers a dropped
    connection internally (``recover=True``): it polls the operation by id
    until it resolves. ``recover_timeout`` bounds that wait to the step's own
    timeout budget so a genuinely dead session still fails fast.
    """
    return await _call_maybe_async(
        session.execute,
        steps,
        on_output=on_output,
        recover_timeout=_step_timeout_budget(steps),
    )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

async def _replay_fork_environment(
    api: ExtensionAPI,
    session: "ArlSandboxSession",
) -> None:
    """Replay the source session's full sandbox history into this fork's sandbox.

    Source ARL session_id is read from lineage (preferred) or resolved via
    experiment_id lookup (requires exactly one match). Replays the entire
    history -- lineage turn indices are AgentM conversation turns, not ARL
    execute steps, so partial replay by turn count would be incorrect.
    """
    lineage = api.lineage
    if not isinstance(lineage, dict) or lineage.get("kind") != "fork":
        return

    source_arl_session = lineage.get("arl_session_id")
    if not isinstance(source_arl_session, str) or not source_arl_session:
        experiment_id = lineage.get("arl_experiment_id") or lineage.get(
            "source_session_id"
        )
        if not experiment_id:
            return

        import arl  # type: ignore[import-not-found]

        client = _gateway_client_class(arl)()
        try:
            arl_sessions = await _call_maybe_async(
                client.list_experiment_sessions, experiment_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent_env: experiment lookup failed for {}: {}", experiment_id, exc
            )
            return
        finally:
            await _close_client(client)

        if len(arl_sessions) != 1:
            logger.warning(
                "agent_env: expected 1 ARL session for experiment {}, got {} -- skipping replay",
                experiment_id,
                len(arl_sessions),
            )
            return
        source_arl_session = arl_sessions[0].id

    logger.info(
        "agent_env: replaying {} into {}",
        source_arl_session,
        session.session_id,
    )
    try:
        result = await _call_maybe_async(
            session.replay_from, source_session_id=source_arl_session
        )
        if result.errors:
            logger.warning(
                "agent_env: replay completed with {} errors out of {} steps",
                result.errors,
                result.steps_replayed,
            )
        else:
            logger.info("agent_env: replay complete -- {} steps", result.steps_replayed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_env: ARL replay failed: {}", exc)


async def _replay_resume_environment(
    api: ExtensionAPI,
    session: "ArlSandboxSession",
    experiment_id: str | None,
) -> None:
    """Restore the mutable sandbox on ``--resume`` of an ARL-backed session.

    A resumed session gets a fresh, empty sandbox; without restoration the
    agent continues its conversation against a blank filesystem. Each run of a
    given AgentM session creates its ARL sandbox under ``experiment_id ==
    session_id`` (the default), so a prior ARL session under this session's
    experiment id is exactly this session's own earlier sandbox. Replay its
    recorded steps into the fresh sandbox to reconstruct state.

    Ambiguity guard: eval experiment ids are per-task, so a resumed eval task
    finds exactly one prior (its own earlier run) -- safe to replay. A single
    prior is always unambiguous. Only when several priors exist do we require a
    session-scoped experiment id (== the AgentM ``session_id``), so a genuinely
    shared experiment (many sibling tasks) is skipped rather than replaying an
    arbitrary sibling's sandbox. Forks are handled by
    :func:`_replay_fork_environment` via lineage and never reach here with a
    prior (a fork's experiment id is its own new session id).
    """
    if not experiment_id:
        return

    import arl  # type: ignore[import-not-found]

    client = _gateway_client_class(arl)()
    try:
        sessions = await _call_maybe_async(
            client.list_experiment_sessions, experiment_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "agent_env: resume env lookup failed for {}: {}", experiment_id, exc
        )
        return
    finally:
        await _close_client(client)

    # A prior ARL session is a valid replay source even after its sandbox was
    # torn down -- replay_from re-executes recorded steps from the gateway's
    # store, not a live container. So do NOT filter on deleted_at here; only
    # exclude the freshly created current sandbox.
    priors = [
        s for s in sessions if getattr(s, "id", None) not in (None, session.session_id)
    ]
    if not priors:
        return  # fresh start -- no earlier sandbox to restore

    source = max(priors, key=lambda s: str(getattr(s, "created_at", "") or ""))
    if len(priors) > 1:
        logger.info(
            "agent_env: resume -- {} candidate priors under experiment {}, "
            "picking latest {}",
            len(priors),
            experiment_id,
            source.id,
        )
    logger.info(
        "agent_env: resume -- replaying prior sandbox {} into {}",
        source.id,
        session.session_id,
    )
    try:
        result = await _call_maybe_async(
            session.replay_from, source_session_id=source.id
        )
        if result.errors:
            logger.warning(
                "agent_env: resume replay completed with {} errors out of {} steps",
                result.errors,
                result.steps_replayed,
            )
        else:
            logger.info(
                "agent_env: resume replay complete -- {} steps", result.steps_replayed
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "agent_env: resume replay failed (prior sandbox may be gone): {}", exc
        )


def _build_resources(config: AgentEnvConfig) -> Any:
    reqs = {
        k: v
        for k, v in {"cpu": config.cpu_request, "memory": config.memory_request}.items()
        if v
    }
    lims = {
        k: v
        for k, v in {"cpu": config.cpu_limit, "memory": config.memory_limit}.items()
        if v
    }
    if not reqs and not lims:
        return None
    from arl.types import ResourceRequirements  # type: ignore[import-not-found]

    return ResourceRequirements(requests=reqs, limits=lims)


async def install_agent_env(api: ExtensionAPI, config: AgentEnvConfig) -> None:
    # Deferred imports to avoid circular dependency with bash/writer subpackages.
    from agentm.extensions.builtin.bash.agent_env import AgentEnvBashOperations
    from agentm.extensions.builtin.writer.agent_env import AgentEnvResourceWriter

    try:
        import arl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "operations backend 'agent_env' requires the 'arl-env' package. "
            "Install with: uv sync --extra agent-env"
        ) from exc

    work_dir = config.work_dir or "/workspace"
    delete_on_shutdown = (
        config.delete_on_shutdown if config.delete_on_shutdown is not None else True
    )

    owned = True
    session: ArlSandboxSession
    if config.attach_session:
        sandbox_cls = _arl_class(arl, "SandboxSession", "AsyncSandboxSession")
        session = await _call_maybe_async(
            sandbox_cls.attach,
            config.attach_session,
            timeout=config.create_timeout or 600.0,
        )
        owned = False
        logger.info("agent_env: attached to existing sandbox {}", config.attach_session)
    elif config.image:
        # Filter private_containers: only include entries with both name and image.
        pcs = [
            pc for pc in (config.private_containers or [])
            if pc.get("name") and pc.get("image")
        ] or None
        if pcs:
            logger.info("agent_env: private_containers={}", pcs)
        managed_cls = _arl_class(arl, "ManagedSession", "AsyncManagedSession")
        session = managed_cls(
            image=config.image,
            experiment_id=config.experiment_id or api.session_id,
            workspace_dir=work_dir,
            timeout=config.create_timeout or 600.0,
            resources=_build_resources(config),
            profile=config.profile or "default",
            config_env=config.config_env,
            idle_timeout_seconds=config.idle_timeout_seconds,
            private_containers=pcs,
        )
    else:
        raise RuntimeError(
            "operations backend 'agent_env': 'image' or 'attach_session' required. "
            "Set via manifest config or ${AGENTM_AGENT_ENV_IMAGE} in the config value."
        )

    if owned:
        await _call_maybe_async(session.create_sandbox)

    session_id = session.session_id or ""
    if session_id:
        api.set_service(_AGENT_ENV_SESSION_SERVICE, session_id)
    if owned and hasattr(session, "experiment_id"):
        api.set_service(_AGENT_ENV_EXPERIMENT_SERVICE, session.experiment_id)
    api.set_service(_AGENT_ENV_WORK_DIR_SERVICE, work_dir)

    bash_ops = AgentEnvBashOperations(
        session,
        default_work_dir=work_dir,
        default_timeout=config.timeout,
    )
    writer = AgentEnvResourceWriter(session, work_dir=work_dir)
    api.register_operations(bash=bash_ops)
    api.register_resource_writer(writer)

    if owned:
        await _replay_fork_environment(api, session)
        await _replay_resume_environment(
            api, session, config.experiment_id or api.session_id
        )

    def _on_shutdown(_event: SessionShutdownEvent) -> None:
        if not owned:
            return
        if delete_on_shutdown:
            from arl import GatewayClient  # type: ignore[import-not-found]

            client = GatewayClient()
            try:
                client.delete_session(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "agent_env: sandbox deletion failed on shutdown: {}", exc
                )
            finally:
                client.close()
        else:
            logger.info(
                "agent_env: keeping sandbox {} for external cleanup", session_id
            )

    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
