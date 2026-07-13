from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agentm.extensions.builtin._agent_env import (
    AgentEnvConfig,
    install_agent_env,
)
from agentm.extensions.builtin.operations import OperationsConfig


class _FakeOperationsAPI:
    session_id = "agentm-session"
    lineage = None

    def __init__(self) -> None:
        self.services: dict[str, str] = {}
        self.operations: dict[str, Any] = {}
        self.resource_writer: Any = None
        self.handlers: list[tuple[str, Any]] = []

    def set_service(self, name: str, value: str) -> None:
        self.services[name] = value

    def register_operations(self, *, bash: Any) -> None:
        self.operations = {"bash": bash}

    def register_resource_writer(self, writer: Any) -> None:
        self.resource_writer = writer

    def on(self, channel: str, handler: Any) -> None:
        self.handlers.append((channel, handler))


def test_operations_config_forwards_agent_env_profile() -> None:
    config = OperationsConfig.model_validate(
        {
            "backend": "agent_env",
            "image": "train-ticket-agent-env:local",
            "profile": "gpu",
        }
    )

    agent_config = AgentEnvConfig.model_validate(config.model_dump(exclude={"backend"}))

    assert agent_config.profile == "gpu"


@pytest.mark.asyncio
async def test_install_attach_passes_timeout_and_registers_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncSandboxSession:
        def __init__(self) -> None:
            self._session_id: str | None = None

        @classmethod
        async def attach(cls, session_id: str, **kwargs: Any) -> Any:
            captured["session_id"] = session_id
            captured.update(kwargs)
            instance = cls()
            instance._session_id = session_id
            return instance

        @property
        def session_id(self) -> str | None:
            return self._session_id

    monkeypatch.setitem(
        sys.modules,
        "arl",
        SimpleNamespace(
            AsyncSandboxSession=FakeAsyncSandboxSession,
            GatewayClient=SimpleNamespace,
        ),
    )

    api = _FakeOperationsAPI()
    await install_agent_env(
        cast(Any, api),
        AgentEnvConfig(
            attach_session="sandbox-session",
            gateway_url="http://gateway.invalid",
            create_timeout=123.0,
        ),
    )

    assert captured == {
        "session_id": "sandbox-session",
        "timeout": 123.0,
    }
    assert api.services["agent_env.session_id"] == "sandbox-session"
    assert set(api.operations) == {"bash"}
    assert api.resource_writer is not None
