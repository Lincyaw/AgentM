"""OAuth type definitions — port of pi-mono `utils/oauth/types.ts`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol


@dataclass(slots=True)
class OAuthCredentials:
    """Persisted credentials for an OAuth provider session.

    ``expires_at_ms`` is an absolute unix timestamp in **milliseconds**
    (mirrors pi-mono's JS-native ``Date.now()`` units; the explicit ``_ms``
    suffix prevents the second/millisecond ambiguity that caused issue #123
    E14). Additional provider-specific fields (e.g. ``account_id``) can be
    stashed in ``extra``.
    """

    refresh: str
    access: str
    expires_at_ms: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OAuthPrompt:
    """User prompt the OAuth flow may request via ``OAuthLoginCallbacks.on_prompt``."""

    message: str
    placeholder: str | None = None
    allow_empty: bool = False


@dataclass(slots=True, frozen=True)
class OAuthAuthInfo:
    """Auth URL the user must visit to complete the login flow."""

    url: str
    instructions: str | None = None


OnAuth = Callable[[OAuthAuthInfo], None]
OnPrompt = Callable[[OAuthPrompt], Awaitable[str]]
OnProgress = Callable[[str], None]
OnManualCodeInput = Callable[[], Awaitable[str]]


@dataclass(slots=True)
class OAuthLoginCallbacks:
    """Callbacks the host (CLI / web UI) supplies to drive the login flow."""

    on_auth: OnAuth
    on_prompt: OnPrompt
    on_progress: OnProgress | None = None
    on_manual_code_input: OnManualCodeInput | None = None


class OAuthProviderInterface(Protocol):
    """Contract every OAuth provider implementation must satisfy."""

    id: str
    name: str
    uses_callback_server: bool

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        """Run the login flow, returning credentials to persist."""

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        """Refresh expired credentials, returning the updated set."""

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        """Convert credentials into the API key string the provider accepts."""
