"""Session replacement runtime for switch/new/fork flows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentm.core.abi import AgentLoop, EventBus, TextContent, Tool
from agentm.core.abi.session import ENTRY_TYPE_MESSAGE
from agentm.harness.session_cwd import assert_session_cwd_exists
from agentm.harness.atom_reloader import AtomReloader
from agentm.harness.extension import (
    CommandSpec,
    ProviderConfig,
    Renderer,
    _ExtensionAPIImpl,
)
from agentm.harness.resource_loader import ResourceLoader
from agentm.harness.session_manager import SessionManager

if TYPE_CHECKING:
    from agentm.harness.session import AgentSession
    from agentm.harness.session_services import AgentSessionServices


@dataclass(slots=True)
class SessionRuntime:
    bus: EventBus
    session_manager: SessionManager
    resource_loader: ResourceLoader
    loop: AgentLoop
    active_provider_box: dict[str, ProviderConfig | None]
    tools: list[Tool]
    commands: dict[str, CommandSpec]
    providers: dict[str, ProviderConfig]
    renderers: dict[str, Renderer]
    apis: dict[str, _ExtensionAPIImpl]
    reloader: AtomReloader
    pending_user_messages: list[str | list[Any]]


@dataclass(frozen=True, slots=True)
class CreateAgentSessionRuntimeResult:
    session: AgentSession
    services: AgentSessionServices
    diagnostics: list[Any]


CreateAgentSessionRuntimeFactory = Callable[
    [str, "AgentSessionServices", SessionManager, str],
    Awaitable[CreateAgentSessionRuntimeResult],
]


class AgentSessionRuntime:
    def __init__(
        self,
        session: AgentSession,
        services: "AgentSessionServices",
        create_runtime: CreateAgentSessionRuntimeFactory,
        diagnostics: list[Any] | None = None,
    ) -> None:
        self._session = session
        self._services = services
        self._create_runtime = create_runtime
        self._diagnostics = diagnostics or []

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def services(self) -> "AgentSessionServices":
        return self._services

    @property
    def cwd(self) -> str:
        return self._services.cwd

    @property
    def diagnostics(self) -> list[Any]:
        return list(self._diagnostics)

    async def _replace(
        self,
        session_manager: SessionManager,
        *,
        cwd: str,
        reason: str,
    ) -> None:
        previous = self._session
        await previous.shutdown()
        result = await self._create_runtime(
            cwd, self._services, session_manager, reason
        )
        self._session = result.session
        self._services = result.services
        self._diagnostics = list(result.diagnostics)

    async def switch_session(
        self, session_path: str, *, cwd_override: str | None = None
    ) -> None:
        session_manager = SessionManager.open(session_path, cwd_override=cwd_override)
        assert_session_cwd_exists(session_manager, self.cwd)
        await self._replace(
            session_manager,
            cwd=session_manager.get_cwd(),
            reason="resume",
        )

    async def new_session(self, *, parent_session: str | None = None) -> None:
        session_manager = SessionManager.create(self.cwd)
        if parent_session is not None:
            session_manager.new_session(parent_session=parent_session)
        await self._replace(session_manager, cwd=self.cwd, reason="new")

    async def fork(
        self,
        entry_id: str,
        *,
        position: str = "before",
    ) -> dict[str, Any]:
        selected_entry = self._session.session_manager.get_entry(entry_id)
        if selected_entry is None:
            raise ValueError("Invalid entry ID for forking")

        selected_text: str | None = None
        target_leaf_id: str | None
        if position == "at":
            target_leaf_id = selected_entry.id
        else:
            if selected_entry.type != ENTRY_TYPE_MESSAGE:
                raise ValueError("Invalid entry ID for forking")
            message = selected_entry.payload
            if getattr(message, "role", None) != "user":
                raise ValueError("Invalid entry ID for forking")
            selected_text = " ".join(
                block.text
                for block in getattr(message, "content", [])
                if isinstance(block, TextContent)
            )
            target_leaf_id = selected_entry.parent_id

        if self._session.session_manager.is_persisted():
            if target_leaf_id is None:
                session_manager = SessionManager.create(self.cwd)
                session_manager.new_session(
                    parent_session=self._session.session_manager.get_session_file()
                )
            else:
                current_file = self._session.session_manager.get_session_file()
                if current_file is None:
                    raise RuntimeError("Persisted session is missing a session file")
                source_manager = SessionManager.open(Path(current_file))
                branched_path = source_manager.create_branched_session(target_leaf_id)
                if branched_path is None:
                    raise RuntimeError("Failed to create forked session")
                session_manager = SessionManager.open(branched_path)
        else:
            session_manager = self._session.session_manager.fork_at(
                target_leaf_id or entry_id
            )
            if target_leaf_id is None:
                session_manager.reset_leaf()

        await self._replace(
            session_manager, cwd=session_manager.get_cwd(), reason="fork"
        )
        return {"selected_text": selected_text}
