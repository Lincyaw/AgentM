"""Provider interface package.

Interface-only: concrete provider implementations live in
``agentm.extensions.builtin.llm_<provider>``. This package exposes the provider descriptor / registry
types that the CLI and contrib channels use to enumerate available
providers and resolve API keys from the environment.
"""

from agentm.ai.types import (
    DEFAULT_PROVIDER_DESCRIPTORS,
    DEFAULT_PROVIDER_REGISTRY,
    KNOWN_APIS,
    KNOWN_PROVIDERS,
    ApiProvider,
    Model,
    ProviderDescriptor,
    ProviderRegistry,
    StreamFunction,
)

# Convenience re-exports — delegate to the singleton registry.
find_env_keys = DEFAULT_PROVIDER_REGISTRY.find_env_keys
get_env_api_key = DEFAULT_PROVIDER_REGISTRY.get_env_api_key

__all__ = [
    "ApiProvider",
    "DEFAULT_PROVIDER_DESCRIPTORS",
    "DEFAULT_PROVIDER_REGISTRY",
    "KNOWN_APIS",
    "KNOWN_PROVIDERS",
    "Model",
    "ProviderDescriptor",
    "ProviderRegistry",
    "StreamFunction",
    "find_env_keys",
    "get_env_api_key",
]
