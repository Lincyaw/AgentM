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
    ResolvedTrajectoryStore,
    resolve_trajectory_store_or_create,
)

_STORE_OWNER_SERVICE = "agentm.sdk.trajectory_store_owner"


class AgentSession(Session):
    """SDK session with one host-resolved trajectory storage backend."""

    def __init__(self, runtime: SessionRuntimeConfig) -> None:
        super().__init__(runtime)
        store_owner = self.services.get(
            _STORE_OWNER_SERVICE,
            ResolvedTrajectoryStore,
        )
        if store_owner is None:
            return
        if store_owner.store is not self.store:
            return
        store_owner.acquire()

        async def release_store() -> None:
            await asyncio.to_thread(store_owner.release)

        self.register_cleanup(release_store)

    @classmethod
    async def create(cls, config: AgentSessionConfig) -> "AgentSession":
        if config.trajectory_store is not None:
            session = await create_from_config(config, session_type=cls)
            return cast(AgentSession, session)

        resolved = resolve_trajectory_store_or_create(config.cwd or None)
        host_services = ServiceRegistry()
        host_services.register(
            _STORE_OWNER_SERVICE,
            resolved,
            ResolvedTrajectoryStore,
            scope="resource",
        )
        try:
            session = await create_from_config(
                replace(
                    config,
                    trajectory_store=resolved.store,
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
