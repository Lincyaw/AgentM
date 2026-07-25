"""Lightweight LLM calls without spawning child sessions.

Uses the AGENTM_HOME config.toml to resolve model credentials and
calls the OpenAI-compatible API directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

_CONFIG_CACHE: dict[str, tuple[str, str, str]] = {}


def _load_model_config(model_name: str | None = None) -> tuple[str, str, str] | None:
    """Return (api_key, base_url, model_id) from config.toml."""
    cache_key = model_name or "_default_"
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    try:
        import tomllib  # noqa: PLC0415

        agentm_home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
        config_path = Path(agentm_home) / "config.toml"
        if not config_path.is_file():
            config_path = Path.home() / ".agentm" / "config.toml"
        if not config_path.is_file():
            return None

        config = tomllib.loads(config_path.read_text())
        models = config.get("models", {})
        if not models:
            return None

        if model_name and model_name in models:
            mc = models[model_name]
        else:
            default = config.get("default_model", "")
            mc = models.get(default) or next(iter(models.values()))

        api_key = mc.get("api_key", "")
        base_url = mc.get("base_url", "")
        model_id = mc.get("model", "")
        if not api_key or not model_id:
            return None
        result = (api_key, base_url, model_id)
        _CONFIG_CACHE[cache_key] = result
        return result
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm: failed to load model config: {}", exc)
        return None


def call_llm(
    system: str,
    prompt: str,
    *,
    model_name: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0,
) -> str | None:
    """Make a direct LLM call. Returns response text or None on failure."""
    model_config = _load_model_config(model_name)
    if model_config is None:
        logger.warning("llm: no model config available")
        return None

    api_key, base_url, model_id = model_config

    try:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(api_key=api_key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm: call failed: {}", exc)
        return None
