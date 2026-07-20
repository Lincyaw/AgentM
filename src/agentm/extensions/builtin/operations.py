"""Local operations atom: configure and register the local environment."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    AtomInstallPriority,
    BASH_OPERATIONS_SERVICE,
    ENVIRONMENT_OPERATIONS_SERVICE,
    AtomAPI,
    BashOperations,
    EnvironmentOperations,
)
from agentm.environments.local import (
    LocalBashOperations,
    LocalEnvironmentOperations,
)
from agentm.extensions import ExtensionManifest


class OperationsConfig(BaseModel):
    """Configuration placeholder for the local operations backend."""

    model_config = ConfigDict(extra="forbid")


MANIFEST = ExtensionManifest(
    name="operations",
    description="Registers local shell operations for SDK sessions.",
    registers=(
        "service:operations:bash",
        "service:operations:environment",
    ),
    config_schema=OperationsConfig,
    requires=(),
    priority=AtomInstallPriority.SERVICE,
)


def install(session: AtomAPI, config: OperationsConfig) -> None:
    del config
    existing_environment = session.services.get(ENVIRONMENT_OPERATIONS_SERVICE)
    existing_bash = session.services.get(BASH_OPERATIONS_SERVICE)
    if existing_environment is not None or existing_bash is not None:
        if not isinstance(existing_environment, EnvironmentOperations):
            raise TypeError("operations atom found an incomplete environment binding")
        if not isinstance(existing_bash, BashOperations):
            raise TypeError("operations atom found an incomplete bash binding")
        if existing_environment.bash is not existing_bash:
            raise ValueError(
                "environment and bash services must describe the same backend"
            )
        return
    bash = LocalBashOperations()
    cwd = session.ctx.cwd or os.getcwd()
    environment = LocalEnvironmentOperations(cwd=cwd, bash=bash)
    session.register_operations(environment=environment, bash=bash)


__all__ = (
    "MANIFEST",
    "OperationsConfig",
    "install",
)
