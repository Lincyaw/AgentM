"""Tests for the ``install(api, config)`` provider entrypoint.

Uses a tiny stub ``api`` with a recording ``register_provider``. The harness
layer is imported lazily inside ``install``; we monkey-patch it to provide a
minimal ``ProviderConfig`` so these tests don't depend on its real shape.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from agentm.core.kernel.stream import Model
from agentm.llm.anthropic import AnthropicStreamFn, install


@dataclass
class _StubProviderConfig:
    stream_fn: Any
    model: Model
    name: str


@pytest.fixture(autouse=True)
def _fake_harness_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``agentm.harness.extension`` exposing ``ProviderConfig``.

    The real harness module is being implemented in a parallel phase; this
    fixture makes the lazy import inside ``install()`` resolve to a stub
    matching the documented shape.
    """

    pkg_name = "agentm.harness.extension"
    module = types.ModuleType(pkg_name)
    module.ProviderConfig = _StubProviderConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, pkg_name, module)


class _StubAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def register_provider(self, name: str, config: Any) -> None:
        self.calls.append((name, config))


def test_install_registers_provider_with_known_model() -> None:
    api = _StubAPI()

    install(api, {"model": "claude-opus-4-7"})

    assert len(api.calls) == 1
    name, cfg = api.calls[0]
    assert name == "anthropic"
    assert isinstance(cfg, _StubProviderConfig)
    assert cfg.name == "anthropic"
    assert cfg.model.id == "claude-opus-4-7"
    assert cfg.model.provider == "anthropic"
    assert cfg.model.context_window == 200_000
    assert cfg.model.max_output_tokens == 32_768
    assert isinstance(cfg.stream_fn, AnthropicStreamFn)


def test_install_unknown_model_uses_defaults_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _StubAPI()

    with caplog.at_level("WARNING", logger="agentm.llm.anthropic"):
        install(api, {"model": "claude-future-9000"})

    assert len(api.calls) == 1
    _, cfg = api.calls[0]
    assert cfg.model.context_window == 200_000
    assert cfg.model.max_output_tokens == 8_192
    assert any("unknown model id" in rec.message for rec in caplog.records)


def test_install_missing_model_raises_value_error() -> None:
    api = _StubAPI()
    with pytest.raises(ValueError, match="config\\['model'\\]"):
        install(api, {})
    assert api.calls == []


def test_install_passes_api_key_and_base_url() -> None:
    api = _StubAPI()
    install(
        api,
        {
            "model": "claude-sonnet-4-6",
            "api_key": "sk-test",
            "base_url": "https://example.invalid",
        },
    )
    _, cfg = api.calls[0]
    assert cfg.stream_fn.api_key == "sk-test"
    assert cfg.stream_fn.base_url == "https://example.invalid"
