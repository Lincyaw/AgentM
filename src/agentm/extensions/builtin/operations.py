"""Unified ``operations`` atom — registers the Operations bundle (file I/O + shell).

Backend selected by ``config["backend"]``:

- ``"local"`` (default) — local-FS / asyncio-subprocess, suitable for CLI
  and most scenarios.
- ``"agent_env"`` — ARL sandbox-backed, for Kubernetes-isolated execution.

Implementations live in ``bash/`` (BashOperations), ``writer/`` (ResourceWriter),
and ``_agent_env.py`` (shared ARL helpers + agent-env install entry point).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest


class OperationsConfig(BaseModel):
    backend: Literal["local", "agent_env"] = "local"
    # agent_env-specific properties (ignored when backend=local)
    image: str | None = None
    experiment_id: str | None = None
    attach_session: str | None = None
    gateway_url: str | None = None
    api_key: str | None = None
    profile: str | None = None
    config_env: dict[str, object] | None = None
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


MANIFEST = ExtensionManifest(
    name="operations",
    description=(
        "Registers the Operations bundle (file I/O + shell). "
        "Backend selected by config: 'local' (default) or 'agent_env'."
    ),
    registers=(),
    config_schema=OperationsConfig,
    requires=(),
    api_version=1,
    tier=1,
)


class _OperationsRuntime:
    """Install-time backend dispatcher for the unified operations atom."""

    def __init__(self, api: ExtensionAPI, config: OperationsConfig) -> None:
        self._api = api
        self._config = config

    async def install(self) -> None:
        if self._config.backend == "local":
            self._install_local()
            return
        if self._config.backend == "agent_env":
            await self._install_agent_env()
            return
        raise ValueError(f"Unknown operations backend: {self._config.backend!r}")

    def _install_local(self) -> None:
        # Sub-installers expect a plain dict; forward the full model dump.
        from agentm.extensions.builtin.bash.local import install_local

        install_local(self._api, self._config.model_dump())

    async def _install_agent_env(self) -> None:
        from agentm.extensions.builtin._agent_env import (
            AgentEnvConfig,
            install_agent_env,
        )

        await install_agent_env(
            self._api,
            AgentEnvConfig.model_validate(self._config.model_dump(exclude={"backend"})),
        )


async def install(api: ExtensionAPI, config: OperationsConfig) -> None:
    await _OperationsRuntime(api, config).install()


__all__ = (
    "MANIFEST",
    "OperationsConfig",
    "install",
)
