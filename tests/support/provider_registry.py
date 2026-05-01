from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4

from agentm.ai.api_registry import register_api_provider, unregister_api_providers
from agentm.ai.types import ProviderDefinition, ResolvedAuth
from agentm.core.kernel.stream import Model, StreamFn


@dataclass(frozen=True, slots=True)
class RegisteredTestProvider:
    provider_id: str
    source_id: str


ModelBuilder = tuple[str, int, int]


def _build_model(provider_id: str, model_id: str, shape: ModelBuilder) -> Model:
    _, context_window, max_output_tokens = shape
    return Model(
        id=model_id,
        provider=provider_id,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


@contextmanager
def temporary_provider(
    stream_fn: StreamFn,
    *,
    provider_id: str | None = None,
    default_model: str | None = None,
    model_provider: str | None = None,
    api: str = "test-api",
    context_window: int = 10_000,
    max_output_tokens: int = 1_000,
) -> Iterator[str]:
    provider_name = provider_id or f"test-provider-{uuid4().hex}"
    source_id = f"tests.support.provider_registry:{provider_name}:{uuid4().hex}"
    model_id = default_model or provider_name
    provider_label = model_provider or provider_name
    shape: ModelBuilder = (model_id, context_window, max_output_tokens)

    def model_factory(selected_model_id: str) -> Model:
        return _build_model(provider_label, selected_model_id, shape)

    def stream_factory(
        model: Model,
        config: dict[str, object],
        auth: ResolvedAuth | None,
    ) -> StreamFn:
        del model, config, auth
        return stream_fn

    register_api_provider(
        ProviderDefinition(
            id=provider_name,
            display_name=provider_name,
            api=api,
            default_model=model_id,
            model_factory=model_factory,
            stream_factory=stream_factory,
            requires_auth=False,
        ),
        source_id=source_id,
    )
    try:
        yield provider_name
    finally:
        unregister_api_providers(source_id)
