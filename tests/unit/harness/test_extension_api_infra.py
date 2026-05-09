from __future__ import annotations

from typing import Any

import pytest

from agentm.core.abi import EventBus
from agentm.harness.extension import _ExtensionAPIImpl
from agentm.harness.session_config import AgentSessionConfig


class _SessionView:
    def get_messages(self) -> list[Any]:
        return []

    def get_branch(self) -> list[Any]:
        return []

    def get_leaf_id(self) -> str | None:
        return None

    def get_entry(self, entry_id: str) -> Any | None:
        del entry_id
        return None

    def get_loop_config(self) -> Any:
        return None

    def append_entry(self, type: str, payload: Any, parent_id: str | None = None) -> str:
        del type, payload, parent_id
        return "entry"


def _api(tmp_path: Any, *, child_factory: Any | None = None) -> _ExtensionAPIImpl:
    return _ExtensionAPIImpl(
        bus=EventBus(),
        cwd=str(tmp_path),
        session_id="session",
        session=_SessionView(),
        tools=[],
        commands={},
        providers={},
        renderers={},
        pending_user_messages=[],
        model_getter=lambda: None,
        provider_getter=lambda: None,
        child_session_factory=child_factory,
    )


@pytest.mark.asyncio
async def test_add_observer_fires_for_every_emitted_channel(tmp_path: Any) -> None:
    api = _api(tmp_path)
    seen: list[tuple[str, Any]] = []

    api.add_observer(lambda channel, event: seen.append((channel, event)))

    await api.events.emit("alpha", {"value": 1})
    api.events.emit_sync("beta", {"value": 2})

    assert seen == [("alpha", {"value": 1}), ("beta", {"value": 2})]


@pytest.mark.asyncio
async def test_spawn_child_session_accepts_dataclass_dict_and_kwargs(tmp_path: Any) -> None:
    received: list[Any] = []

    async def _factory(config: Any) -> Any:
        received.append(config)
        return config

    api = _api(tmp_path, child_factory=_factory)
    dataclass_config = AgentSessionConfig(cwd=str(tmp_path), provider=("fake", {}))

    assert await api.spawn_child_session(dataclass_config) is dataclass_config
    assert await api.spawn_child_session({"cwd": str(tmp_path), "provider": ("fake", {})}) == {
        "cwd": str(tmp_path),
        "provider": ("fake", {}),
    }
    assert await api.spawn_child_session(cwd=str(tmp_path), provider=("fake", {})) == {
        "cwd": str(tmp_path),
        "provider": ("fake", {}),
    }
    assert received[0] is dataclass_config


@pytest.mark.asyncio
async def test_spawn_child_session_rejects_mixed_config_and_kwargs(tmp_path: Any) -> None:
    api = _api(tmp_path, child_factory=lambda config: config)

    with pytest.raises(TypeError, match="either a config object or keyword args"):
        await api.spawn_child_session({}, cwd=str(tmp_path))


def test_service_registry_round_trips_and_rejects_duplicate(tmp_path: Any) -> None:
    api = _api(tmp_path)
    service = object()

    assert api.get_service("artifact_store") is None
    api.set_service("artifact_store", service)

    assert api.get_service("artifact_store") is service
    with pytest.raises(KeyError, match="already registered"):
        api.set_service("artifact_store", object())
