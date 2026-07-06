"""ARL agent-env sandbox: shared utilities and entry point.

One sandbox per AgentM session. All config comes from :class:`AgentEnvConfig`
(scenario manifest). Connection parameters (``gateway_url``, ``api_key``) fall
through to the ARL SDK's own config chain when unset.
"""

from __future__ import annotations

import asyncio
import posixpath
import time
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
_OPERATION_POLL_INTERVAL_SECONDS = 2.0
_OPERATION_TIMEOUT_GRACE_SECONDS = 30.0
_OPERATION_STATUS_DONE = "done"
_OPERATION_STATUS_ERROR = "error"


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


async def _get_execute_operation(
    session: "ArlSandboxSession",
    operation_id: str,
) -> Any:
    getter = getattr(session, "get_execute_operation", None)
    if getter is not None:
        return await _call_maybe_async(getter, operation_id)

    session_id = getattr(session, "session_id", None)
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(
            f"cannot recover ARL execute operation {operation_id}: missing session_id"
        )

    import arl  # type: ignore[import-not-found]

    client = _gateway_client_class(arl)()
    try:
        return await _call_maybe_async(
            client.get_execute_operation, session_id, operation_id
        )
    finally:
        await _close_client(client)


async def _recover_execute_operation(
    session: "ArlSandboxSession",
    steps: list[dict[str, Any]],
    operation_id: str,
    started_at: float,
) -> Any:
    budget = _step_timeout_budget(steps)
    deadline = started_at + budget if budget is not None else None

    while True:
        operation = await _get_execute_operation(
            session,
            operation_id,
        )
        if operation.result is not None:
            return operation.result

        status = (operation.status or "").lower()
        if status == _OPERATION_STATUS_ERROR:
            raise RuntimeError(
                operation.error or f"ARL execute operation {operation_id} failed"
            )
        if status == _OPERATION_STATUS_DONE:
            raise RuntimeError(
                f"ARL execute operation {operation_id} finished without result"
            )

        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"ARL execute operation {operation_id} still pending after "
                    f"{budget:.0f}s"
                )
            sleep_for = min(_OPERATION_POLL_INTERVAL_SECONDS, remaining)
        else:
            sleep_for = _OPERATION_POLL_INTERVAL_SECONDS
        await asyncio.sleep(sleep_for)


async def _async_execute(
    session: "ArlSandboxSession",
    steps: list[dict[str, Any]],
    *,
    on_output: Callable[[str, str], None] | None = None,
) -> Any:
    from arl import GatewayOperationTimeout  # type: ignore[import-not-found]

    started_at = time.monotonic()
    try:
        return await _call_maybe_async(session.execute, steps, on_output=on_output)
    except GatewayOperationTimeout as exc:
        return await _recover_execute_operation(
            session,
            steps,
            exc.operation_id,
            started_at,
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

    bash_ops = AgentEnvBashOperations(
        session,
        default_work_dir=work_dir,
        default_timeout=config.timeout,
    )
    writer = AgentEnvResourceWriter(
        session, work_dir=work_dir,
        session_id=session_id,
    )
    api.register_operations(bash=bash_ops)
    api.register_resource_writer(writer)

    if owned:
        await _replay_fork_environment(api, session)

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
