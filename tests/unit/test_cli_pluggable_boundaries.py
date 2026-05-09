from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.ai import ProviderDescriptor, ProviderRegistry
from agentm.cli import _build_session_config
from agentm.core.abi import EventBus
from agentm.harness.session_manager import SessionManager


class _FakeSessionStore:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.recent_cwds: list[Path] = []
        self.created_cwds: list[Path] = []
        self.resume_state = SessionManager.in_memory("/fake")
        self.recent_state = SessionManager.in_memory("/fake")
        self.created_state = SessionManager.in_memory("/fake")

    def open(self, id: str) -> SessionManager:
        self.opened.append(id)
        return self.resume_state

    def most_recent(self, cwd: Path) -> SessionManager | None:
        self.recent_cwds.append(cwd)
        return self.recent_state

    def create(self, cwd: Path) -> SessionManager:
        self.created_cwds.append(cwd)
        return self.created_state


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            id="fake-provider",
            api="fake-api",
            env_var_precedence=("FAKE_API_KEY",),
            aliases=("fake",),
            extension_module="tests.fake_provider",
            default_model="fake-default",
        ),
        lambda: _Provider(),
    )
    return registry


class _Provider:
    api = "fake-api"

    async def stream(
        self, model: Any, context: Any, options: Any | None = None
    ) -> list[Any]:
        return [model, context, options]

    async def stream_simple(
        self, model: Any, context: Any, options: Any | None = None
    ) -> list[Any]:
        return [model, context, options]


def _build_with(
    store: _FakeSessionStore, **overrides: Any
) -> tuple[Any, SessionManager]:
    kwargs = {
        "scenario": None,
        "extra_extensions": [],
        "no_extensions": False,
        "no_skills": False,
        "no_prompt_templates": False,
        "tool_allowlist": None,
        "provider": "fake",
        "model": "fake-model",
        "cwd": "/fake",
        "bus": EventBus(),
        "session_store": store,
        "provider_registry": _registry(),
    }
    kwargs.update(overrides)
    config, state = _build_session_config(**kwargs)
    return config, state  # type: ignore[return-value]


def test_cli_resume_uses_injected_session_store_without_disk_lookup() -> None:
    store = _FakeSessionStore()

    config, state = _build_with(store, resume="abc123")

    assert store.opened == ["abc123"]
    assert store.recent_cwds == []
    assert store.created_cwds == []
    assert state is store.resume_state
    assert config.session_manager is store.resume_state
    assert config.provider == ("tests.fake_provider", {"model": "fake-model"})


def test_cli_continue_uses_injected_session_store_without_disk_lookup() -> None:
    store = _FakeSessionStore()

    config, state = _build_with(store, continue_recent=True)

    assert store.opened == []
    assert store.recent_cwds == [Path("/fake")]
    assert store.created_cwds == []
    assert state is store.recent_state
    assert config.session_manager is store.recent_state
