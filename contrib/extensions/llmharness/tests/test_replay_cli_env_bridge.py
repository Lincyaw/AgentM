"""Fail-stop: ``llmharness-replay`` env→provider config bridge.

``agentm`` CLI lifts ``AGENTM_PROVIDER`` / ``AGENTM_MODEL`` +
provider-specific env vars (``OPENAI_BASE_URL`` etc.) into the provider
config dict via ``ProviderRegistry.build``. ``llmharness-replay`` must
do the same so users running against a self-signed / Warpgate-fronted
endpoint don't have to hand-stuff each knob into a JSON spec.
"""

from __future__ import annotations

import pytest

from llmharness.replay.cli import _parse_provider


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


