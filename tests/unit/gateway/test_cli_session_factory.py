"""Gateway session factory metadata contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentm.gateway.cli import _build_session_factory


class _ProviderRegistry:
    def build(self, provider: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return (f"built.{provider}", dict(config))


class _State:
    def get_header(self) -> Any:
        return SimpleNamespace(
            config={
                "lineage": {"kind": "root", "entrypoint": "agentm.gateway"},
                "experiment": {"kind": "reminder_injection", "case_id": "1188"},
            }
        )


@pytest.mark.asyncio
async def test_gateway_fork_sets_parent_session_id_for_trace_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "agentm.ai.DEFAULT_PROVIDER_REGISTRY",
        _ProviderRegistry(),
    )
    monkeypatch.setattr(
        "agentm.core.runtime.session_bootstrap.make_default_session_store",
        lambda cwd: object(),
    )

    def fake_resolve_session_state(**kwargs: Any) -> _State:
        assert kwargs["fork"] == "source-session"
        assert kwargs["fork_up_to"] == 65
        return _State()

    monkeypatch.setattr(
        "agentm.core.runtime.session_bootstrap.resolve_session_state",
        fake_resolve_session_state,
    )

    async def fake_create(config: Any) -> Any:
        captured["config"] = config
        return SimpleNamespace(session_id="fork-session")

    monkeypatch.setattr(
        "agentm.core.runtime.session.AgentSession.create",
        staticmethod(fake_create),
    )

    factory = _build_session_factory(
        provider="openai",
        model="gpt-test",
        profile=None,
    )
    await factory(
        str(tmp_path),
        "terminal:t1",
        "local",
        None,
        {"fork_source": "source-session", "fork_up_to": 65},
    )

    config = captured["config"]
    assert config.parent_session_id == "source-session"
    assert config.lineage == {
        "kind": "fork",
        "entrypoint": "agentm.gateway",
        "source_session_id": "source-session",
        "fork_point": {"up_to": 65},
        "source_lineage": {"kind": "root", "entrypoint": "agentm.gateway"},
    }
    assert config.experiment == {"kind": "reminder_injection", "case_id": "1188"}
