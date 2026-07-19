"""Embedded SDK presenter with host-side default composition."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import AgentSessionConfig
from agentm.core.runtime.session import Session
from agentm.core.runtime.session_core import SessionRuntimeConfig
from agentm.core.runtime.session_factory import create_from_config
from agentm.storage.trajectory.resolve import (
    ResolvedTrajectoryStorage,
    resolve_trajectory_storage_or_create,
)

_STORAGE_OWNER_SERVICE = "agentm.sdk.trajectory_storage_owner"


class AgentSession(Session):
    """SDK session with one host-resolved trajectory storage backend."""

    def __init__(self, runtime: SessionRuntimeConfig) -> None:
        super().__init__(runtime)
        storage_owner = self.services.get(
            _STORAGE_OWNER_SERVICE,
            ResolvedTrajectoryStorage,
        )
        if storage_owner is None:
            return
        selected = storage_owner.storage
        if (
            selected.turn_store is not self.store
            or selected.node_store is not self.get_trajectory_node_store()
        ):
            return
        storage_owner.acquire()

        async def release_storage() -> None:
            await asyncio.to_thread(storage_owner.release)

        self.register_cleanup(release_storage)

    @classmethod
    async def create(cls, config: AgentSessionConfig) -> "AgentSession":
        if config.trajectory_storage is not None:
            session = await create_from_config(config, session_type=cls)
            return cast(AgentSession, session)

        resolved = resolve_trajectory_storage_or_create(config.cwd or None)
        host_services = ServiceRegistry()
        host_services.register(
            _STORAGE_OWNER_SERVICE,
            resolved,
            ResolvedTrajectoryStorage,
            scope="resource",
        )
        try:
            session = await create_from_config(
                replace(
                    config,
                    trajectory_storage=resolved.storage,
                ),
                session_type=cls,
                host_services=host_services,
            )
        except BaseException as creation_error:
            try:
                resolved.close()
            except Exception as close_error:
                raise BaseExceptionGroup(
                    "session creation and trajectory cleanup failed",
                    (creation_error, close_error),
                ) from creation_error
            raise

        return cast(AgentSession, session)


__all__ = ["AgentSession"]
