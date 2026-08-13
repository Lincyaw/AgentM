# code-health: ignore-file[AM025] -- validates typed service bindings at the atom boundary
"""Explicit local environment and logical-resource backend."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    BASH_OPERATIONS_ROLE,
    BASH_OPERATIONS_SERVICE,
    ENVIRONMENT_OPERATIONS,
    ENVIRONMENT_OPERATIONS_SERVICE,
    RESOURCE_READER,
    RESOURCE_READER_SERVICE,
    RESOURCE_STORE,
    RESOURCE_STORE_SERVICE,
    RESOURCE_WRITER,
    RESOURCE_WRITER_SERVICE,
    AtomAPI,
    BashOperations,
    EnvironmentOperations,
    ResourceReader,
    ResourceStore,
    ResourceWriter,
    bind_resource_store,
)
from agentm.environments.local import LocalBashOperations, LocalEnvironmentOperations
from agentm.extensions import ExtensionManifest
from agentm.storage.resources import LocalResourceStore


class LocalBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_root: str | None = None
    resource_manifest_path: str | None = None
    discover_resource_manifest: bool = True


MANIFEST = ExtensionManifest(
    name="local_backend",
    description="Register one explicit local environment and resource backend.",
    registers=(
        BASH_OPERATIONS_ROLE.capability,
        ENVIRONMENT_OPERATIONS.capability,
        RESOURCE_READER.capability,
        RESOURCE_STORE.capability,
        RESOURCE_WRITER.capability,
    ),
    config_schema=LocalBackendConfig,
    requires=(),
)


def install(api: AtomAPI, config: LocalBackendConfig) -> None:
    environment = api.services.get(ENVIRONMENT_OPERATIONS_SERVICE)
    bash = api.services.get(BASH_OPERATIONS_SERVICE)
    if environment is None and bash is None:
        local_bash = LocalBashOperations()
        local_environment: EnvironmentOperations = LocalEnvironmentOperations(
            cwd=api.ctx.cwd or os.getcwd(),
            bash=local_bash,
        )
        api.register_operations(environment=local_environment, bash=local_bash)
        environment_kind = local_environment.ref.kind
    elif not isinstance(environment, EnvironmentOperations) or not isinstance(
        bash, BashOperations
    ):
        raise TypeError("local_backend found an incomplete operations binding")
    elif environment.bash is not bash:
        raise ValueError(
            "local_backend requires environment and bash services from one backend"
        )
    else:
        environment_kind = environment.ref.kind

    reader = api.services.get(RESOURCE_READER_SERVICE)
    store = api.services.get(RESOURCE_STORE_SERVICE)
    writer = api.services.get(RESOURCE_WRITER_SERVICE)
    if reader is not None and not isinstance(reader, ResourceReader):
        raise TypeError("local_backend found an invalid ResourceReader binding")
    if store is not None and not isinstance(store, ResourceStore):
        raise TypeError("local_backend found an invalid ResourceStore binding")
    if writer is not None and not isinstance(writer, ResourceWriter):
        raise TypeError("local_backend found an invalid ResourceWriter binding")
    existing_resources = (reader, store, writer)
    if all(service is not None for service in existing_resources):
        return
    if any(service is not None for service in existing_resources):
        raise TypeError("local_backend found an incomplete resource binding")

    # About to bind the local, host-rooted resource store. bash runs inside the
    # environment; file tools read/write through this store. If the environment
    # is a sandbox/remote whose filesystem is not the host, a file written here
    # is invisible to bash and vice versa -- an agent that writes a script then
    # runs it silently sees nothing. Refuse the split-brain instead of letting
    # it surface as a confusing runtime miss; a non-local environment must
    # register its own matching resource backend.
    if environment_kind not in {"local", "host"}:
        raise ValueError(
            f"local_backend: the environment is {environment_kind!r} but no "
            "resource backend is registered, so file tools would fall back to a "
            "host-local store the environment cannot see. Register a matching "
            "resource_reader/resource_store/resource_writer from the same "
            "backend as the environment."
        )

    resources = LocalResourceStore(
        workspace_root=Path(api.ctx.cwd or "."),
        root=config.resource_root,
        manifest_path=config.resource_manifest_path,
        discover_manifest=config.discover_resource_manifest,
    )
    api.services.bind(RESOURCE_READER, resources)
    bind_resource_store(api.services, resources)
    api.services.bind(RESOURCE_WRITER, resources)


__all__ = ("MANIFEST", "LocalBackendConfig", "install")
