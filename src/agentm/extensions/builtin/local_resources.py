"""Local workspace and logical-resource storage atom."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    RESOURCE_READER_SERVICE,
    RESOURCE_STORE_SERVICE,
    RESOURCE_WRITER_SERVICE,
    AtomInstallPriority,
    ResourceReader,
    ResourceStore,
    ResourceWriter,
)
from agentm.extensions import ExtensionManifest
from agentm.storage.resources import LocalResourceStore


class LocalResourcesConfig(BaseModel):
    root: str | None = None
    manifest_path: str | None = None
    discover_manifest: bool = True


MANIFEST = ExtensionManifest(
    name="local_resources",
    description="Register transactional local workspace and logical-resource storage.",
    registers=(
        "service:resource_reader",
        "service:resource_store",
        "service:resource_writer",
    ),
    config_schema=LocalResourcesConfig,
    requires=(),
    priority=AtomInstallPriority.SERVICE,
)


def install(session: Any, config: LocalResourcesConfig) -> None:
    reader = session.services.get(RESOURCE_READER_SERVICE)
    store = session.services.get(RESOURCE_STORE_SERVICE)
    writer = session.services.get(RESOURCE_WRITER_SERVICE)
    if reader is not None and not isinstance(reader, ResourceReader):
        raise TypeError("local_resources found an invalid ResourceReader binding")
    if store is not None and not isinstance(store, ResourceStore):
        raise TypeError("local_resources found an invalid ResourceStore binding")
    if writer is not None and not isinstance(writer, ResourceWriter):
        raise TypeError("local_resources found an invalid ResourceWriter binding")
    if reader is not None and store is not None and writer is not None:
        return

    resources = LocalResourceStore(
        workspace_root=Path(session.ctx.cwd or "."),
        root=config.root,
        manifest_path=config.manifest_path,
        discover_manifest=config.discover_manifest,
    )
    if reader is None:
        session.register_resource_reader(resources)
    if store is None:
        session.register_resource_store(resources)
    if writer is None:
        session.register_resource_writer(resources)


__all__ = ("LocalResourcesConfig", "MANIFEST", "install")
