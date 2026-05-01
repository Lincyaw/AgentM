from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.ai.types import ProviderDefinition, ResolvedAuth
from agentm.core.kernel.messages import AssistantMessage, TextContent
from agentm.core.kernel.stream import MessageEnd, Model, StreamFn


def build_generic_model(
    provider: str,
    model_id: str,
    *,
    context_window: int = 200_000,
    max_output_tokens: int = 8_192,
) -> Model:
    return Model(
        id=model_id,
        provider=provider,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


class UnsupportedProviderStream:
    def __init__(self, provider_id: str) -> None:
        self._provider_id = provider_id

    async def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[MessageEnd]:
        del messages, tools, system, signal, thinking
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Provider '{self._provider_id}' is registered in the AgentM "
                            "provider registry but its Python transport is not implemented yet."
                        ),
                    )
                ],
                timestamp=0.0,
                stop_reason="error",
                usage=None,
            )
        )


def unsupported_stream_factory(
    provider_id: str,
) -> StreamFn:
    return UnsupportedProviderStream(provider_id)


def make_stub_provider(
    *,
    provider_id: str,
    display_name: str,
    api: str,
    default_model: str,
    env_vars: tuple[str, ...] = (),
    oauth_provider_id: str | None = None,
    context_window: int = 200_000,
    max_output_tokens: int = 8_192,
) -> ProviderDefinition:
    def model_factory(model_id: str) -> Model:
        return build_generic_model(
            provider_id,
            model_id,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )

    def stream_factory(model: Model, config: dict[str, Any], auth: ResolvedAuth | None) -> StreamFn:
        del model, config, auth
        return unsupported_stream_factory(provider_id)

    return ProviderDefinition(
        id=provider_id,
        display_name=display_name,
        api=api,
        default_model=default_model,
        model_factory=model_factory,
        stream_factory=stream_factory,
        env_vars=env_vars,
        oauth_provider_id=oauth_provider_id,
    )
