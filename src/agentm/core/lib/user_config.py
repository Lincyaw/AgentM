"""Global user configuration from ``~/.agentm/config.toml``.

Reads once per process and caches. Missing file or parse errors fall back
to empty defaults — the config file is always optional.
"""

from __future__ import annotations

from loguru import logger
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict



class ModelBuildConfig(TypedDict, total=False):
    model: str
    base_url: str
    api_key: str
    name: str
    context_window: int
    max_output_tokens: int
    reasoning_effort: str
    extra_body: dict[str, Any]


@dataclass(frozen=True)
class ModelProfile:
    """A named model profile from ``[models.<name>]``."""

    provider: str
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None

    def to_build_config(self) -> ModelBuildConfig:
        """Build the config dict for ``ProviderRegistry.build()``."""
        config: ModelBuildConfig = {"model": self.model}
        if self.base_url:
            config["base_url"] = self.base_url
        if self.api_key:
            config["api_key"] = self.api_key
        if self.name:
            config["name"] = self.name
        if self.context_window:
            config["context_window"] = self.context_window
        if self.max_output_tokens:
            config["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort:
            config["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            config["extra_body"] = self.extra_body
        return config


@dataclass(frozen=True)
class UserConfig:
    """Top-level user config parsed from ``config.toml``."""

    default_model: str | None = None
    models: dict[str, ModelProfile] = field(default_factory=dict)


_EMPTY = UserConfig()
_cached: UserConfig | None = None


def agentm_home_dir() -> Path:
    """Return the AgentM home directory: ``$AGENTM_HOME`` or ``~/.agentm``."""
    home = os.environ.get("AGENTM_HOME")
    return Path(home) if home else Path.home() / ".agentm"


def _config_path() -> Path:
    return agentm_home_dir() / "config.toml"


def _parse_profile(key: str, raw: dict[str, Any]) -> ModelProfile | None:
    provider = raw.get("provider")
    model = raw.get("model")
    if not isinstance(provider, str) or not isinstance(model, str):
        logger.warning(f"config.toml: [models.{key}] missing required 'provider' or 'model'; skipped")
        return None
    name = raw.get("name")
    if not isinstance(name, str):
        name = key

    base_url = raw.get("base_url")
    api_key = raw.get("api_key")
    context_window = raw.get("context_window")
    max_output_tokens = raw.get("max_output_tokens")
    reasoning_effort = raw.get("reasoning_effort")
    extra_body = raw.get("extra_body")

    return ModelProfile(
        provider=provider,
        name=name,
        model=model,
        base_url=base_url if isinstance(base_url, str) else None,
        api_key=api_key if isinstance(api_key, str) else None,
        context_window=int(context_window) if context_window is not None else None,
        max_output_tokens=(
            int(max_output_tokens) if max_output_tokens is not None else None
        ),
        reasoning_effort=(
            reasoning_effort if isinstance(reasoning_effort, str) else None
        ),
        extra_body=extra_body if isinstance(extra_body, dict) else None,
    )


def load_user_config() -> UserConfig:
    """Load and cache ``~/.agentm/config.toml`` (or ``$AGENTM_HOME/config.toml``).

    Returns :data:`_EMPTY` when the file does not exist or cannot be parsed.
    """
    global _cached
    if _cached is not None:
        return _cached

    path = _config_path()
    if not path.is_file():
        _cached = _EMPTY
        return _cached

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        logger.opt(exception=True).warning(f"config.toml: failed to parse {path}; using defaults")
        _cached = _EMPTY
        return _cached

    default_model = data.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        logger.warning("config.toml: 'default_model' must be a string; ignored")
        default_model = None

    models: dict[str, ModelProfile] = {}
    raw_models = data.get("models")
    if isinstance(raw_models, dict):
        for key, value in raw_models.items():
            if not isinstance(value, dict):
                logger.warning(f"config.toml: [models.{key}] is not a table; skipped")
                continue
            profile = _parse_profile(key, value)
            if profile is not None:
                # Store under lower-cased key for case-insensitive lookup.
                models[key.lower()] = profile

    _cached = UserConfig(default_model=default_model, models=models)
    return _cached


def resolve_model_profile(model_name: str | None) -> ModelProfile | None:
    """Look up a model profile by name (case-insensitive).

    When *model_name* is ``None``, falls back to
    ``config.default_model``.  Returns ``None`` when no matching profile
    exists (callers should fall through to existing behaviour).
    """
    config = load_user_config()
    if model_name is None:
        if config.default_model is None:
            return None
        model_name = config.default_model
    return config.models.get(model_name.lower())


def apply_reasoning_effort(
    build_config: ModelBuildConfig, cli_flag: str | None
) -> ModelBuildConfig:
    """Apply reasoning-effort precedence: CLI flag > env > config.toml profile.

    Mutates and returns *build_config*. When no effort is resolved, any value
    already present from the profile is left untouched.
    """
    effort = (
        cli_flag
        or os.environ.get("AGENTM_REASONING_EFFORT")
        or build_config.get("reasoning_effort")
    )
    if effort:
        build_config["reasoning_effort"] = effort
    return build_config


def _reset_cache() -> None:
    """Clear the cached config. For testing only."""
    global _cached
    _cached = None
