"""OAuth interface package — interface-only port of pi-mono OAuth utils.

Concrete provider login flows (Anthropic, GitHub Copilot, Google Gemini
CLI) are not ported. Adding one means implementing
:class:`OAuthProviderInterface` for the provider and registering it via
the consumer's own bootstrap.
"""

from agentm.ai.oauth.pkce import generate_pkce
from agentm.ai.oauth.types import (
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthProviderInterface,
)

__all__ = [
    "OAuthAuthInfo",
    "OAuthCredentials",
    "OAuthLoginCallbacks",
    "OAuthPrompt",
    "OAuthProviderInterface",
    "generate_pkce",
]
