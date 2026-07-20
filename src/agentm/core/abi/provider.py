# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Provider selection port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .messages import AgentMessage
from .stream import Model, StreamFn
from .trajectory import PromptCacheState


ProviderMetaScalar = str | int | float | bool | None
ProviderMeta = Mapping[str, ProviderMetaScalar]


def _nonempty_string(value: object, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _freeze_metadata(value: ProviderMeta, label: str) -> ProviderMeta:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    copied: dict[str, ProviderMetaScalar] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise TypeError(f"{label}[{key!r}] must be a scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label}[{key!r}] must be finite")
        copied[key] = item
    return MappingProxyType(copied)


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

    def __post_init__(self) -> None:
        if not isinstance(self.stream_fn, StreamFn):
            raise TypeError("provider stream_fn must implement StreamFn")
        if not isinstance(self.model, Model):
            raise TypeError("provider model must be Model")
        _nonempty_string(self.name, "provider name")
        if self.prompt_cache_adapter is not None and not isinstance(
            self.prompt_cache_adapter,
            ProviderPromptCacheAdapter,
        ):
            raise TypeError(
                "provider prompt_cache_adapter must implement "
                "ProviderPromptCacheAdapter"
            )


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

    def __post_init__(self) -> None:
        _nonempty_string(self.name, "provider identity name")
        _nonempty_string(
            self.model_id,
            "provider identity model_id",
            optional=True,
        )
        _nonempty_string(
            self.active_set_digest,
            "provider identity active_set_digest",
            optional=True,
        )
        if self.frozen_after_turn_index is not None and (
            not isinstance(self.frozen_after_turn_index, int)
            or isinstance(self.frozen_after_turn_index, bool)
            or self.frozen_after_turn_index < 0
        ):
            raise ValueError(
                "provider identity frozen_after_turn_index must be a "
                "non-negative integer"
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "provider identity metadata"),
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

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not all(isinstance(message, AgentMessage) for message in messages):
            raise TypeError("provider cache request messages are invalid")
        if not isinstance(self.model, Model):
            raise TypeError("provider cache request model must be Model")
        if not isinstance(self.state, PromptCacheState):
            raise TypeError("provider cache request state must be PromptCacheState")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "provider cache request metadata"),
        )


@dataclass(frozen=True, slots=True)
class ProviderPromptCacheResult:
    """Provider adapter output after applying cache hints."""

    messages: Sequence[AgentMessage]
    state: PromptCacheState
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not all(isinstance(message, AgentMessage) for message in messages):
            raise TypeError("provider cache result messages are invalid")
        if not isinstance(self.state, PromptCacheState):
            raise TypeError("provider cache result state must be PromptCacheState")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "provider cache result metadata"),
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
    """Tree-scoped host policy for selecting the active provider.

    A resolver must return one registered name whenever ``providers`` is
    non-empty. Returning ``None`` is only valid for an empty registry.
    """

    def resolve_provider(self, providers: ProviderRegistry) -> str | None: ...


__all__ = [
    "ProviderConfig",
    "ProviderPromptCacheAdapter",
    "ProviderPromptCacheRequest",
    "ProviderPromptCacheResult",
    "ProviderMeta",
    "ProviderMetaScalar",
    "ProviderRegistry",
    "ProviderResolver",
    "ProviderSessionIdentity",
]
