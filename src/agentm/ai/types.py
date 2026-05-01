"""Type definitions for the provider registry. Port of pi-mono `types.ts`.

Only the names referenced by `api_registry.py`, `env_api_keys.py`, and the
OAuth interfaces are included. The full pi-mono `types.ts` defines stream
options, message events, and ~25 provider literals; we intentionally
narrow that surface to what the registry needs today and let consumers
extend `KNOWN_PROVIDERS` / `KNOWN_APIS` when they add new providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Literal, Protocol

KnownApi = Literal[
    "openai-completions",
    "openai-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
]
Api = str  # KnownApi | str — accept any string at runtime, narrow via Literal in callers

KnownProvider = Literal[
    "anthropic",
    "openai",
    "amazon-bedrock",
    "google",
    "google-vertex",
    "azure-openai-responses",
    "openai-codex",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "mistral",
    "huggingface",
    "fireworks",
]
Provider = str

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]


@dataclass(slots=True)
class Model:
    """Minimal model descriptor. Mirrors pi-mono `Model<Api>` shape.

    ``id`` is the provider-side model identifier (e.g. ``claude-3-7-sonnet``).
    ``provider`` and ``api`` route the call through the registry.
    ``options`` carries provider-specific config (base_url, headers, ...) —
    untyped on purpose so the registry stays decoupled from individual
    provider option schemas.
    """

    id: str
    provider: Provider
    api: Api
    options: dict[str, Any] = field(default_factory=dict)


# Stream functions are async-iterable producers in pi-mono. We model them
# as callables returning an async iterator of opaque events; the concrete
# event shape lives with each provider's port (out of scope here).

StreamFunction = Callable[[Model, Any], Awaitable[Iterable[Any]]]


class ApiProvider(Protocol):
    """Protocol every concrete provider implementation must satisfy.

    Mirrors pi-mono `ApiProvider<TApi, TOptions>` interface. The registry
    stores providers by their `api` string; `stream` and `stream_simple`
    receive a `Model` whose `api` field must match `provider.api`.
    """

    api: Api

    def stream(self, model: Model, context: Any, options: Any | None = None) -> Awaitable[Iterable[Any]]:
        ...

    def stream_simple(
        self, model: Model, context: Any, options: Any | None = None
    ) -> Awaitable[Iterable[Any]]:
        ...
