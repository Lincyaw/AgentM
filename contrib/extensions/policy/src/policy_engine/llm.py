"""Lightweight LLM calls without spawning child sessions.

Uses the AGENTM_HOME config.toml to resolve model credentials and
calls the OpenAI-compatible API directly. This avoids creating PG
session records for every tagger/critic call.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from loguru import logger


@lru_cache(maxsize=1)
def _load_model_config() -> tuple[str, str, str] | None:
    """Return (api_key, base_url, model_id) from config.toml."""
    try:
        import tomllib  # noqa: PLC0415

        agentm_home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
        config_path = Path(agentm_home) / "config.toml"
        if not config_path.is_file():
            config_path = Path.home() / ".agentm" / "config.toml"
        if not config_path.is_file():
            return None

        config = tomllib.loads(config_path.read_text())
        default_model = config.get("default_model", "")
        models = config.get("models", {})
        if not models:
            return None

        mc = models.get(default_model) or next(iter(models.values()))
        api_key = mc.get("api_key", "")
        base_url = mc.get("base_url", "")
        model_id = mc.get("model", "")
        if not api_key or not model_id:
            return None
        return api_key, base_url, model_id
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm: failed to load model config: {}", exc)
        return None


def call_llm(
    system: str,
    prompt: str,
    *,
    max_tokens: int = 300,
    temperature: float = 0,
) -> str | None:
    """Make a direct LLM call. Returns response text or None on failure."""
    model_config = _load_model_config()
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
