"""Provider selection port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .messages import AgentMessage
from .stream import Model, StreamFn
from .trajectory import PromptCacheState


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Session-local LLM provider registration.

    Provider atoms install through the normal ``ExtensionManifest`` path, then
    register one of these records with the session. Host selection policy lives
    separately in ``ProviderResolver``.
    """

    stream_fn: StreamFn
    model: Model
    name: str
    prompt_cache_adapter: ProviderPromptCacheAdapter | None = None


@dataclass(frozen=True, slots=True)
class ProviderSessionIdentity:
    """Provider/model identity bound to a session history.

    Once a session has committed a turn, this identity is frozen. Changing it
    requires a fork/new session or an explicit future config-change control
    node; silent mid-history provider drift is not allowed.
    """

    name: str
    model_id: str | None = None
    active_set_digest: str | None = None
    frozen_after_turn_index: int | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


ProviderRegistry = Mapping[str, ProviderConfig]


@dataclass(frozen=True, slots=True)
class ProviderPromptCacheRequest:
    """Provider adapter input for prompt-cache materialization."""

    messages: Sequence[AgentMessage]
    model: Model
    state: PromptCacheState
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ProviderPromptCacheResult:
    """Provider adapter output after applying cache hints."""

    messages: Sequence[AgentMessage]
    state: PromptCacheState
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@runtime_checkable
class ProviderPromptCacheAdapter(Protocol):
    """Provider-specific bridge for prompt-cache state.

    Context policy owns when cache boundaries exist. Provider adapters own how
    that boundary is represented for one model API. Core never embeds
    provider-specific cache-control syntax.
    """

    def apply_prompt_cache(
        self,
        request: ProviderPromptCacheRequest,
    ) -> ProviderPromptCacheResult: ...


@runtime_checkable
class ProviderResolver(Protocol):
    """Tree-scoped host policy for selecting the active provider."""

    def resolve_provider(self, providers: ProviderRegistry) -> str | None: ...


__all__ = [
    "ProviderConfig",
    "ProviderPromptCacheAdapter",
    "ProviderPromptCacheRequest",
    "ProviderPromptCacheResult",
    "ProviderRegistry",
    "ProviderResolver",
    "ProviderSessionIdentity",
]
