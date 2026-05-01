"""Environment-based API key discovery — port of pi-mono `env-api-keys.ts`.

Cherry-picked: only providers AgentM currently has consumers for, plus
the order convention pi-mono uses (e.g., ``ANTHROPIC_OAUTH_TOKEN`` takes
precedence over ``ANTHROPIC_API_KEY``). The Bedrock/Vertex ambient-
credential probing in pi-mono is out of scope here — those providers
must add their own discovery when ported.
"""

from __future__ import annotations

import os

# Order matters: first match wins inside `get_env_api_key`. Lists are
# tuples to keep them immutable at module level.
_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "azure-openai-responses": ("AZURE_OPENAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "google-vertex": ("GOOGLE_CLOUD_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "github-copilot": ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
}


def find_env_keys(provider: str) -> list[str] | None:
    """Return env var names that are *currently set* for ``provider``.

    Returns ``None`` when the provider is unknown or none of its env
    vars are populated. Mirrors pi-mono ``findEnvKeys``.
    """

    candidates = _PROVIDER_ENV_VARS.get(provider)
    if not candidates:
        return None
    found = [name for name in candidates if os.environ.get(name)]
    return found or None


def get_env_api_key(provider: str) -> str | None:
    """Return the API key for ``provider`` from the highest-priority env var.

    Returns ``None`` if the provider is unknown or no env var is set.
    """

    found = find_env_keys(provider)
    if not found:
        return None
    return os.environ.get(found[0])
