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

from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="operations",
    description=(
        "Registers the Operations bundle (file I/O + shell). "
        "Backend selected by config: 'local' (default) or 'agent_env'."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {
            "backend": {
                "type": "string",
                "enum": ["local", "agent_env"],
                "default": "local",
            },
            # agent_env-specific properties (ignored when backend=local)
            "image": {"type": "string"},
            "experiment_id": {"type": "string"},
            "pool_ref": {"type": "string"},
            "gateway_url": {"type": "string"},
            "namespace": {"type": "string"},
            "work_dir": {"type": "string"},
            "timeout": {"type": ["number", "null"]},
            "idle_timeout_seconds": {"type": ["integer", "null"]},
        },
        "additionalProperties": False,
    },
    requires=(),
)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    backend = config.get("backend", "local")
    if backend == "local":
        from agentm.extensions.builtin._operations.local import install_local

        install_local(api, config)
    elif backend == "agent_env":
        from agentm.extensions.builtin._operations.agent_env import install_agent_env

        install_agent_env(api, config)
    else:
        raise ValueError(f"Unknown operations backend: {backend!r}")
