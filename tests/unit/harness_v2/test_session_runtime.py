from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.kernel import AssistantMessage, AssistantStreamEvent, MessageEnd, Model, TextContent
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.session_cwd import MissingSessionCwdError, assert_session_cwd_exists
from agentm.harness.session_manager import SessionManager
from agentm.harness.session_runtime import AgentSessionRuntime
from agentm.harness.session_services import (
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
    create_agent_session_from_services,
    create_agent_session_services,
)
from agentm.harness.extension import ProviderConfig


class _ScriptedProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        return self._iter(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=self.reply)],
                timestamp=1.0,
                stop_reason="end_turn",
            )
        )

    async def _iter(self, msg: AssistantMessage) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=msg)


def _install_provider_module(name: str, reply: str) -> str:
    module = types.ModuleType(name)
    provider = _ScriptedProvider(reply)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-runtime",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-runtime",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-runtime",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


@pytest.mark.asyncio
async def test_create_services_and_session_from_services(tmp_path: Path) -> None:
    provider_module = _install_provider_module(
        "tests.unit.harness_v2._fixtures.runtime_provider_services",
        "hi from services",
    )
    services = await create_agent_session_services(
        CreateAgentSessionServicesOptions(
            cwd=str(tmp_path),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    session = await create_agent_session_from_services(
        CreateAgentSessionFromServicesOptions(
            services=services,
            extensions=[],
            provider=(provider_module, {}),
        )
    )
    final = await session.prompt("hello")
    assert isinstance(final[-1], AssistantMessage)
    assert final[-1].content[0].text == "hi from services"
    await session.shutdown()


@pytest.mark.asyncio
async def test_runtime_fork_replaces_session_with_selected_branch(tmp_path: Path) -> None:
    provider_module = _install_provider_module(
        "tests.unit.harness_v2._fixtures.runtime_provider_runtime",
        "forked reply",
    )
    services = await create_agent_session_services(
        CreateAgentSessionServicesOptions(
            cwd=str(tmp_path),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[],
            provider=(provider_module, {}),
            resource_loader=services.resource_loader,
        )
    )
    await session.prompt("first")
    user_entry = next(
        entry for entry in session.session_manager.get_active_branch() if entry.type == "message"
    )

    async def create_runtime(
        cwd: str,
        current_services: Any,
        session_manager: SessionManager,
        _reason: str,
    ) -> Any:
        new_services = await create_agent_session_services(
            CreateAgentSessionServicesOptions(
                cwd=cwd,
                resource_loader=current_services.resource_loader,
            )
        )
        new_session = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=new_services,
                extensions=[],
                provider=(provider_module, {}),
                session_manager=session_manager,
            )
        )
        return types.SimpleNamespace(
            session=new_session,
            services=new_services,
            diagnostics=[],
        )

    runtime = AgentSessionRuntime(session, services, create_runtime)
    result = await runtime.fork(user_entry.id, position="before")
    assert result["selected_text"] == "first"
    assert runtime.session.session_manager.get_messages() == []
    await runtime.session.shutdown()


def test_missing_session_cwd_raises(tmp_path: Path) -> None:
    session_path = tmp_path / "missing-cwd.jsonl"
    manager = SessionManager.create(str(tmp_path), tmp_path / "sessions")
    manager._cwd = str(tmp_path / "does-not-exist")  # type: ignore[attr-defined]
    manager._session_file = session_path  # type: ignore[attr-defined]
    manager.new_session()

    with pytest.raises(MissingSessionCwdError):
        assert_session_cwd_exists(manager, str(tmp_path))
