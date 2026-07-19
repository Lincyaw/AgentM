"""Observable provider cache adapter for SDK behavior tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from agentm.core.abi.manifest import AtomInstallPriority
from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.provider import (
    ProviderPromptCacheAdapter,
    ProviderPromptCacheRequest,
    ProviderPromptCacheResult,
)
from agentm.core.abi.roles import PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE
from agentm.core.abi.session_api import AtomAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="test_prompt_cache_adapter",
    description="Expose durable cache generations to a stub provider.",
    registers=("service:provider_prompt_cache_adapter",),
    requires=("context_policy:prompt_cache",),
    priority=AtomInstallPriority.PROVIDER,
)


class ObservablePromptCacheAdapter:
    """Annotate provider-bound messages with the durable cache generation."""

    def apply_prompt_cache(
        self,
        request: ProviderPromptCacheRequest,
    ) -> ProviderPromptCacheResult:
        generation = _generation(request.state.metadata) + 1
        marker = f"{request.state.cache_key}:{generation}"
        messages = tuple(
            _tag_message(message, marker=marker)
            if message.meta.tags.get("cache_key") == request.state.cache_key
            else message
            for message in request.messages
        )
        state = replace(
            request.state,
            provider=request.state.provider or request.model.provider,
            metadata={
                **dict(request.state.metadata),
                "provider_cache_generation": generation,
            },
        )
        return ProviderPromptCacheResult(messages=messages, state=state)


def install(api: AtomAPI, config: dict[str, object]) -> None:
    del config
    api.services.register(
        PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE,
        ObservablePromptCacheAdapter(),
        ProviderPromptCacheAdapter,
        scope="tree",
    )


def _generation(metadata: Mapping[str, object]) -> int:
    value = metadata.get("provider_cache_generation", 0)
    return value if isinstance(value, int) else 0


def _tag_message(message: AgentMessage, *, marker: str) -> AgentMessage:
    return replace(
        message,
        meta=replace(
            message.meta,
            tags={**dict(message.meta.tags), "provider_cache_marker": marker},
        ),
    )


__all__ = ["MANIFEST", "ObservablePromptCacheAdapter", "install"]
