"""Session config inheritance contracts for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.cli import CliRunConfig, _build_cli_lineage, _build_session_config
from agentm.core.abi import EventBus
from agentm.core.abi.session import SessionContext, SessionHeader
from agentm.core.lib.user_config import ModelProfile


class _State:
    session_file: Path | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        self._header = SessionHeader(
            type="session",
            version=1,
            id="source",
            timestamp=0.0,
            cwd="/workspace",
            config=config,
        )

    def append_message(self, message: Any) -> Any:
        raise NotImplementedError

    def build_session_context(self, leaf_id: str | None = None) -> SessionContext:
        del leaf_id
        return SessionContext(messages=[])

    def get_session_id(self) -> str:
        return "source"

    def get_session_file(self) -> str | None:
        return None

    def get_header(self) -> SessionHeader:
        return self._header

    def is_persisted(self) -> bool:
        return False


class _Store:
    def __init__(self, state: _State) -> None:
        self.state = state

    def open(self, id: str) -> _State:
        assert id == "source"
        return self.state

    def most_recent(self, cwd: Path) -> _State | None:
        del cwd
        return self.state

    def create(self, cwd: Path) -> _State:
        del cwd
        return self.state

    def fork(self, source_id: str, **_: Any) -> _State:
        assert source_id == "source"
        return self.state


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def build(self, provider: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        built = dict(config)
        self.calls.append((provider, built))
        return (f"built.{provider}", built)


def test_raw_model_override_preserves_inherited_provider(tmp_path: Path) -> None:
    state = _State(
        {
            "provider": [
                "agentm.extensions.builtin.llm_anthropic",
                {
                    "name": "anthropic",
                    "model": "old-model",
                    "base_url": "https://api.example.test",
                },
            ],
        }
    )
    registry = _Registry()

    session_config, _ = _build_session_config(
        CliRunConfig(
            provider="openai",
            model="new-model",
            cwd=str(tmp_path),
            resume="source",
            model_explicit=True,
            no_extensions=True,
        ),
        bus=EventBus(),
        session_store=_Store(state),
        provider_registry=registry,  # type: ignore[arg-type]
    )

    assert session_config.provider == (
        "agentm.extensions.builtin.llm_anthropic",
        {
            "name": "anthropic",
            "model": "new-model",
            "base_url": "https://api.example.test",
        },
    )
    assert registry.calls == []


def test_profile_model_override_replaces_inherited_provider(tmp_path: Path) -> None:
    state = _State(
        {
            "provider": [
                "agentm.extensions.builtin.llm_anthropic",
                {"name": "anthropic", "model": "old-model"},
            ],
        }
    )
    registry = _Registry()
    profile = ModelProfile(provider="openai", name="fast", model="gpt-4")

    session_config, _ = _build_session_config(
        CliRunConfig(
            provider="openai",
            model="gpt-4",
            cwd=str(tmp_path),
            resume="source",
            model_explicit=True,
            profile=profile,
            no_extensions=True,
        ),
        bus=EventBus(),
        session_store=_Store(state),
        provider_registry=registry,  # type: ignore[arg-type]
    )

    assert session_config.provider == ("built.openai", {"model": "gpt-4", "name": "fast"})
    assert registry.calls == [("openai", {"model": "gpt-4", "name": "fast"})]


def test_cli_lineage_defaults_to_root_for_new_prompt(tmp_path: Path) -> None:
    lineage = _build_cli_lineage(
        CliRunConfig(
            provider="openai",
            model="gpt-4",
            cwd=str(tmp_path),
            prompt="hello",
            no_extensions=True,
        ),
        stored=None,
    )

    assert lineage == {
        "kind": "root",
        "entrypoint": "agentm.cli",
        "prompt": True,
    }


def test_cli_lineage_records_fork_selector(tmp_path: Path) -> None:
    lineage = _build_cli_lineage(
        CliRunConfig(
            provider="openai",
            model="gpt-4",
            cwd=str(tmp_path),
            fork="source-session",
            fork_turn_index=3,
            no_extensions=True,
        ),
        stored={"lineage": {"kind": "root", "entrypoint": "agentm.cli"}},
    )

    assert lineage == {
        "kind": "fork",
        "entrypoint": "agentm.cli",
        "source_session_id": "source-session",
        "fork_point": {"turn_index": 3},
        "source_lineage": {"kind": "root", "entrypoint": "agentm.cli"},
    }


def test_cli_fork_sets_parent_session_id_for_trace_index(tmp_path: Path) -> None:
    state = _State(
        {
            "provider": [
                "agentm.extensions.builtin.llm_anthropic",
                {"name": "anthropic", "model": "old-model"},
            ],
        }
    )

    session_config, _ = _build_session_config(
        CliRunConfig(
            provider="openai",
            model="gpt-4",
            cwd=str(tmp_path),
            fork="source",
            fork_turn_index=3,
            no_extensions=True,
        ),
        bus=EventBus(),
        session_store=_Store(state),
    )

    assert session_config.parent_session_id == "source"
    assert session_config.lineage == {
        "kind": "fork",
        "entrypoint": "agentm.cli",
        "source_session_id": "source",
        "fork_point": {"turn_index": 3},
    }
