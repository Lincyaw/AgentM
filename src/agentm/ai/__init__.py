"""Provider interface package — port of pi-mono `packages/ai/`.

This is an *interface-only* port. Concrete provider implementations
(Anthropic, OpenAI, Bedrock, ...) are deliberately not ported; they
should be added by consumers when needed. The current direct-LLM
path through ``agentm.llm`` continues to work alongside this registry.
"""

from agentm.ai.api_registry import (
    ApiProvider,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)
from agentm.ai.env_api_keys import find_env_keys, get_env_api_key, resolve
from agentm.ai.types import (
    DEFAULT_PROVIDER_DESCRIPTORS,
    DEFAULT_PROVIDER_REGISTRY,
    KNOWN_APIS,
    KNOWN_PROVIDERS,
    Model,
    ProviderDescriptor,
    ProviderRegistry,
    StreamFunction,
)

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
    "clear_api_providers",
    "find_env_keys",
    "get_api_provider",
    "get_api_providers",
    "get_env_api_key",
    "register_api_provider",
    "resolve",
    "unregister_api_providers",
]
