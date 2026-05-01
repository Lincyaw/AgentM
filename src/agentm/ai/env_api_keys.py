from __future__ import annotations

import os
from pathlib import Path

_PROC_ENV_CACHE: dict[str, str] | None = None
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
    "vercel-ai-gateway": ("AI_GATEWAY_API_KEY",),
    "zai": ("ZAI_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "minimax-cn": ("MINIMAX_CN_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "opencode": ("OPENCODE_API_KEY",),
    "opencode-go": ("OPENCODE_API_KEY",),
    "kimi-coding": ("KIMI_API_KEY",),
    "cloudflare-workers-ai": ("CLOUDFLARE_API_KEY",),
    "github-copilot": ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
}


def _get_proc_env(key: str) -> str | None:
    global _PROC_ENV_CACHE
    if _PROC_ENV_CACHE is None:
        _PROC_ENV_CACHE = {}
        environ_path = Path("/proc/self/environ")
        try:
            raw = environ_path.read_bytes()
        except OSError:
            raw = b""
        for item in raw.split(b"\0"):
            if not item or b"=" not in item:
                continue
            name, value = item.split(b"=", 1)
            _PROC_ENV_CACHE[name.decode("utf-8", errors="ignore")] = value.decode(
                "utf-8", errors="ignore"
            )
    return _PROC_ENV_CACHE.get(key)


def find_env_keys(provider: str) -> list[str] | None:
    env_vars = _PROVIDER_ENV_VARS.get(provider)
    if env_vars is None:
        return None
    found = [key for key in env_vars if os.environ.get(key) or _get_proc_env(key)]
    return found or None


def _has_vertex_adc_credentials() -> bool:
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or _get_proc_env(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if explicit:
        return Path(explicit).exists()
    return Path.home().joinpath(
        ".config", "gcloud", "application_default_credentials.json"
    ).exists()


def get_env_api_key(provider: str) -> str | None:
    env_keys = find_env_keys(provider)
    if env_keys:
        key = env_keys[0]
        return os.environ.get(key) or _get_proc_env(key)

    if provider == "google-vertex":
        has_project = bool(
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or _get_proc_env("GOOGLE_CLOUD_PROJECT")
            or _get_proc_env("GCLOUD_PROJECT")
        )
        has_location = bool(
            os.environ.get("GOOGLE_CLOUD_LOCATION")
            or _get_proc_env("GOOGLE_CLOUD_LOCATION")
        )
        if _has_vertex_adc_credentials() and has_project and has_location:
            return "<authenticated>"

    if provider == "amazon-bedrock":
        if (
            os.environ.get("AWS_PROFILE")
            or (
                os.environ.get("AWS_ACCESS_KEY_ID")
                and os.environ.get("AWS_SECRET_ACCESS_KEY")
            )
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
            or _get_proc_env("AWS_PROFILE")
            or (
                _get_proc_env("AWS_ACCESS_KEY_ID")
                and _get_proc_env("AWS_SECRET_ACCESS_KEY")
            )
            or _get_proc_env("AWS_BEARER_TOKEN_BEDROCK")
            or _get_proc_env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or _get_proc_env("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or _get_proc_env("AWS_WEB_IDENTITY_TOKEN_FILE")
        ):
            return "<authenticated>"

    return None
