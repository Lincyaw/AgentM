"""Embedded SDK presenter with host-side default composition."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

from agentm.core.abi.compaction import CompactionPublisher, SessionCompactor
from agentm.core.abi.roles import (
    COMPACTION_PUBLISHER_SERVICE,
    SESSION_COMPACTOR_SERVICE,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import AgentSessionConfig
from agentm.core.abi.store import TrajectoryStore
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
            return _configure_default_compaction(session, config)

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

        return _configure_default_compaction(session, config)

    @classmethod
    async def resume(
        cls,
        session_id: str,
        store: TrajectoryStore,
        config: AgentSessionConfig,
    ) -> "AgentSession":
        session = await super().resume(session_id, store, config)
        return _configure_default_compaction(session, config)


def _configure_default_compaction(
    session: Session,
    config: AgentSessionConfig,
) -> AgentSession:
    result = cast(AgentSession, session)
    _register_default_compaction_services(result, config)
    return result


def _register_default_compaction_services(
    session: AgentSession,
    config: AgentSessionConfig,
) -> None:
    from agentm.presenter.compaction import (
        AgentSessionCompactor,
        TrajectoryCompactionPublisher,
    )

    store = session.store
    if store is None:
        return
    resource_store = session.get_resource_store()
    if not session.services.has(SESSION_COMPACTOR_SERVICE):
        session.services.register(
            SESSION_COMPACTOR_SERVICE,
            AgentSessionCompactor(
                replace(
                    config,
                    trajectory_store=store,
                    resource_store=resource_store,
                )
            ),
            SessionCompactor,
            scope="host",
        )
    if resource_store is not None and not session.services.has(
        COMPACTION_PUBLISHER_SERVICE
    ):
        session.services.register(
            COMPACTION_PUBLISHER_SERVICE,
            TrajectoryCompactionPublisher(
                store=store,
                resource_store=resource_store,
            ),
            CompactionPublisher,
            scope="host",
        )


__all__ = ["AgentSession"]
