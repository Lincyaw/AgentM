"""Cwd-bound session services, split from AgentSession construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi import AgentMessage, LoopConfig
from agentm.harness.resource_loader import DefaultResourceLoader, ResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.session_manager import SessionManager


@dataclass(frozen=True, slots=True)
class AgentSessionRuntimeDiagnostic:
    level: str
    message: str


@dataclass(slots=True)
class AgentSessionServices:
    cwd: str
    resource_loader: ResourceLoader
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CreateAgentSessionServicesOptions:
    cwd: str
    resource_loader: ResourceLoader | None = None


@dataclass(frozen=True, slots=True)
class CreateAgentSessionFromServicesOptions:
    services: AgentSessionServices
    extensions: list[tuple[str, dict[str, Any]]]
    provider: tuple[str, dict[str, Any]]
    initial_messages: list[AgentMessage] = field(default_factory=list)
    session_manager: SessionManager | None = None
    loop_config: LoopConfig | None = None
    parent_bus: Any = None
    parent_session_id: str | None = None
    purpose: str = "root"


async def create_agent_session_services(
    options: CreateAgentSessionServicesOptions,
) -> AgentSessionServices:
    loader = options.resource_loader
    if loader is None:
        loader = DefaultResourceLoader(cwd=Path(options.cwd))
    loader.reload()
    return AgentSessionServices(cwd=options.cwd, resource_loader=loader)


async def create_agent_session_from_services(
    options: CreateAgentSessionFromServicesOptions,
) -> AgentSession:
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=options.services.cwd,
            extensions=options.extensions,
            provider=options.provider,
            initial_messages=list(options.initial_messages),
            session_manager=options.session_manager,
            resource_loader=options.services.resource_loader,
            loop_config=options.loop_config,
            parent_bus=options.parent_bus,
            parent_session_id=options.parent_session_id,
            purpose=options.purpose,
        )
    )
