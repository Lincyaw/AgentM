"""Embedded SDK presenter with host-side default composition."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

from agentm.core.abi.operations import BashOperations
from agentm.core.abi.roles import HOST_BASH_OPERATIONS_SERVICE
from agentm.core.abi.session_api import AgentSessionConfig
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.store import TrajectoryStore
from agentm.core.runtime.session import Session
from agentm.core.runtime.session_core import SessionRuntimeConfig
from agentm.core.runtime.session_factory import create_from_config
from agentm.storage.trajectory.resolve import (
    ResolvedTrajectoryStore,
    resolve_trajectory_store_or_create,
)

_STORE_OWNER_SERVICE = "agentm.sdk.trajectory_store_owner"


def _host_services_with_host_bash() -> ServiceRegistry:
    """Host defaults: shell execution pinned to the SDK host machine.

    Registered at scope="host" so the whole session tree shares one instance
    regardless of each session's own environment backend.
    """
    from agentm.environments.local import LocalBashOperations

    services = ServiceRegistry()
    services.register(
        HOST_BASH_OPERATIONS_SERVICE,
        LocalBashOperations(),
        BashOperations,
        scope="host",
    )
    return services


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
        host_services = _host_services_with_host_bash()
        if config.trajectory_store is not None:
            return cast(
                AgentSession,
                await create_from_config(
                    config,
                    session_type=cls,
                    host_services=host_services,
                ),
            )

        resolved = resolve_trajectory_store_or_create(config.cwd or None)
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

    @classmethod
    async def resume(
        cls,
        session_id: str,
        store: TrajectoryStore,
        config: AgentSessionConfig,
        *,
        host_services: ServiceRegistry | None = None,
    ) -> "AgentSession":
        merged = _host_services_with_host_bash()
        if host_services is not None:
            merged.inherit_from(host_services)
        return cast(
            AgentSession,
            await super().resume(
                session_id,
                store,
                config,
                host_services=merged,
            ),
        )


__all__ = ["AgentSession"]
