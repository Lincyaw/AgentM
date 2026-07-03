from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, cast

from agentm.extensions.builtin._operations import agent_env
from agentm.extensions.builtin._operations.agent_env import (
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

    def register_operations(self, *, file: Any, bash: Any) -> None:
        self.operations = {"file": file, "bash": bash}

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


def test_install_attach_passes_timeout_and_does_not_use_global_arl_api_key(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeSandboxSession:
        @classmethod
        def attach(cls, session_id: str, **kwargs: Any) -> Any:
            captured["arl_api_key_during_attach"] = os.environ.get("ARL_API_KEY")
            captured["session_id"] = session_id
            captured.update(kwargs)
            return SimpleNamespace(session_id=session_id)

    monkeypatch.setitem(
        sys.modules,
        "arl",
        SimpleNamespace(SandboxSession=FakeSandboxSession),
    )
    monkeypatch.setenv("ARL_API_KEY", "global-key-that-must-not-leak")
    monkeypatch.delenv("AGENTM_AGENT_ENV_API_KEY", raising=False)

    api = _FakeOperationsAPI()
    install_agent_env(
        cast(Any, api),
        AgentEnvConfig(
            attach_session="sandbox-session",
            gateway_url="http://gateway.invalid",
            create_timeout=123.0,
        ),
    )

    assert captured == {
        "arl_api_key_during_attach": None,
        "session_id": "sandbox-session",
        "gateway_url": "http://gateway.invalid",
        "api_key": None,
        "timeout": 123.0,
    }
    assert os.environ["ARL_API_KEY"] == "global-key-that-must-not-leak"
    assert api.services["agent_env.session_id"] == "sandbox-session"
    assert set(api.operations) == {"file", "bash"}
    assert api.resource_writer is not None


def test_execute_session_recovers_pending_operation_result(monkeypatch) -> None:
    expected = object()

    class PendingOperationError(Exception):
        operation_id = "op-1"

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.responses = [
                SimpleNamespace(status="running", result=None, error=""),
                SimpleNamespace(status="done", result=expected, error=""),
            ]

        def get_execute_operation(self, session_id: str, operation_id: str) -> Any:
            self.calls.append((session_id, operation_id))
            return self.responses.pop(0)

    class FakeSession:
        session_id = "session-1"

        def __init__(self) -> None:
            self._client = FakeClient()

        def execute(self, steps: list[dict[str, Any]]) -> Any:
            raise PendingOperationError

    session = FakeSession()
    monkeypatch.setattr(agent_env.time, "sleep", lambda _seconds: None)

    result = agent_env._execute_session_sync(
        session,
        [{"cmd": "sleep 10", "timeout": 1}],
    )

    assert result is expected
    assert session._client.calls == [
        ("session-1", "op-1"),
        ("session-1", "op-1"),
    ]
