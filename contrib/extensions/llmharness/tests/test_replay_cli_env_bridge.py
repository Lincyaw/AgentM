"""Fail-stop: ``llmharness-replay`` env→provider config bridge.

``agentm`` CLI lifts ``AGENTM_PROVIDER`` / ``AGENTM_MODEL`` +
provider-specific env vars (``OPENAI_BASE_URL`` etc.) into the provider
config dict via ``ProviderRegistry.build``. ``llmharness-replay`` must
do the same so users running against a self-signed / Warpgate-fronted
endpoint don't have to hand-stuff each knob into a JSON spec.
"""

from __future__ import annotations

import pytest

from llmharness.replay.cli import _parse_provider, resolve_default_provider_spec


def test_resolve_default_provider_bridges_openai_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTM_PROVIDER", "openai")
    monkeypatch.setenv("AGENTM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.delenv("OPENAI_VERIFY_SSL", raising=False)
    monkeypatch.delenv("WARPGATE_TICKET", raising=False)

    spec = resolve_default_provider_spec()
    assert spec is not None
    module, cfg = spec
    assert module == "agentm.extensions.builtin.llm_openai"
    assert cfg["model"] == "gpt-4o-mini"
    assert cfg["base_url"] == "https://example.com/v1"


def test_parse_provider_none_uses_env_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTM_PROVIDER", "openai")
    monkeypatch.setenv("AGENTM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.invalid/v1")

    spec = _parse_provider(None)
    assert spec is not None
    _, cfg = spec
    assert cfg["base_url"] == "https://proxy.invalid/v1"


def test_parse_provider_bare_id_env_bridges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.invalid/v1")

    spec = _parse_provider("openai")
    assert spec is not None
    module, cfg = spec
    assert module == "agentm.extensions.builtin.llm_openai"
    assert cfg["base_url"] == "https://proxy.invalid/v1"


def test_parse_provider_explicit_json_skips_env_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``module:{json}`` wins outright; env knobs do not leak in."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://should-not-appear/")
    spec = _parse_provider(
        'agentm.extensions.builtin.llm_openai:{"model":"sft-4b"}'
    )
    assert spec is not None
    module, cfg = spec
    assert module == "agentm.extensions.builtin.llm_openai"
    assert cfg == {"model": "sft-4b"}


def test_parse_provider_unknown_bare_module_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Third-party module paths not in the registry preserve legacy behaviour."""
    monkeypatch.delenv("AGENTM_MODEL", raising=False)
    spec = _parse_provider("my_org.llm_custom")
    assert spec == ("my_org.llm_custom", {})
