from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from agentm.core.kernel.stream import Model, StreamFn


class OAuthCredentials(TypedDict):
    refresh: str
    access: str
    expires: int


@dataclass(frozen=True, slots=True)
class OAuthPrompt:
    message: str
    placeholder: str | None = None
    allow_empty: bool = False


@dataclass(frozen=True, slots=True)
class OAuthAuthInfo:
    url: str
    instructions: str | None = None


class OAuthLoginCallbacks(Protocol):
    def on_auth(self, info: OAuthAuthInfo) -> None: ...

    async def on_prompt(self, prompt: OAuthPrompt) -> str: ...

    def on_progress(self, message: str) -> None: ...

    async def on_manual_code_input(self) -> str: ...


class OAuthProviderInterface(Protocol):
    id: str
    name: str
    uses_callback_server: bool

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials: ...

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials: ...

    def get_api_key(self, credentials: OAuthCredentials) -> str: ...


@dataclass(frozen=True, slots=True)
class ResolvedAuth:
    api_key: str
    source: str
    label: str | None = None


StreamFactory = Callable[[Model, dict[str, Any], ResolvedAuth | None], StreamFn]
ModelFactory = Callable[[str], Model]


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    id: str
    display_name: str
    api: str
    default_model: str
    model_factory: ModelFactory
    stream_factory: StreamFactory
    env_vars: tuple[str, ...] = ()
    oauth_provider_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_model(self, model_id: str | None = None) -> Model:
        return self.model_factory(model_id or self.default_model)
