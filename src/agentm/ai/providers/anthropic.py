from __future__ import annotations

from typing import Any

from agentm.ai.types import ProviderDefinition, ResolvedAuth
from agentm.core.kernel.stream import Model, StreamFn
from agentm.llm.anthropic import AnthropicStreamFn

_KNOWN_MODELS: dict[str, tuple[int, int]] = {
    "claude-opus-4-7": (200_000, 32_768),
    "claude-sonnet-4-6": (200_000, 64_000),
    "claude-haiku-4-5-20251001": (200_000, 8_192),
}


def _build_model(model_id: str) -> Model:
    context_window, max_output_tokens = _KNOWN_MODELS.get(model_id, (200_000, 8_192))
    return Model(
        id=model_id,
        provider="anthropic",
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


def _stream_factory(
    model: Model,
    config: dict[str, Any],
    auth: ResolvedAuth | None,
) -> StreamFn:
    api_key = config.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        api_key = auth.api_key if auth is not None else None
    if not isinstance(api_key, str) or not api_key:
        raise RuntimeError(
            "No API key found for anthropic. Authenticate via OAuth or set ANTHROPIC_API_KEY."
        )
    base_url = config.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise TypeError("provider config 'base_url' must be a string when set")
    return AnthropicStreamFn(api_key=api_key, base_url=base_url)


PROVIDER = ProviderDefinition(
    id="anthropic",
    display_name="Anthropic",
    api="anthropic-messages",
    default_model="claude-sonnet-4-6",
    model_factory=_build_model,
    stream_factory=_stream_factory,
    env_vars=("ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"),
    oauth_provider_id="anthropic",
)
