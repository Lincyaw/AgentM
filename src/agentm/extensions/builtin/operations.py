"""Unified ``operations`` atom — registers the Operations bundle (file I/O + shell).

Backend selected by ``config["backend"]``:

- ``"local"`` (default) — local-FS / asyncio-subprocess, suitable for CLI
  and most scenarios.
- ``"agent_env"`` — ARL sandbox-backed, for Kubernetes-isolated execution.

Replaces the former ``operations_local`` and ``operations_agent_env`` atoms
with a single entry point and a dispatcher. The implementations live in
``_operations/local.py`` and ``_operations/agent_env.py`` (the ``_`` prefix
prevents auto-discovery as separate atoms).
"""

from __future__ import annotations

from typing import Literal

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
    namespace: str | None = None
    work_dir: str | None = None
    timeout: float | None = None
    idle_timeout_seconds: int | None = None
    max_lifetime_seconds: int | None = None
    create_timeout: float | None = None
    cpu_request: str | None = None
    cpu_limit: str | None = None
    memory_request: str | None = None
    memory_limit: str | None = None
    max_replicas: int | None = None
    min_replicas: int | None = None
    scale_up_step: int | None = None
    delete_on_shutdown: bool | None = None

MANIFEST = ExtensionManifest(
    name="operations",
    description=(
        "Registers the Operations bundle (file I/O + shell). "
        "Backend selected by config: 'local' (default) or 'agent_env'."
    ),
    registers=(),
    config_schema=OperationsConfig,
    requires=(),
)

async def install(api: ExtensionAPI, config: OperationsConfig) -> None:
    # Sub-installers expect a plain dict; forward the full model dump.
    config_dict = config.model_dump()
    backend = config.backend
    if backend == "local":
        from agentm.extensions.builtin._operations.local import install_local

        install_local(api, config_dict)
    elif backend == "agent_env":
        from agentm.extensions.builtin._operations.agent_env import (
            AgentEnvConfig,
            install_agent_env,
        )

        install_agent_env(
            api, AgentEnvConfig.model_validate(config.model_dump(exclude={"backend"}))
        )
    else:
        raise ValueError(f"Unknown operations backend: {backend!r}")
