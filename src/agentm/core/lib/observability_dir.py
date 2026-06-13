"""Resolve the observability output directory, honoring AGENTM_OBSERVABILITY_DIR."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_OBSERVABILITY_DIR = "AGENTM_OBSERVABILITY_DIR"


def file_export_requested() -> bool:
    """Return True when the user explicitly opted into local file export."""
    return bool(os.environ.get(_ENV_OBSERVABILITY_DIR))


def resolve_observability_dir(cwd: str | Path | None = None) -> Path:
    """Return the observability directory, honoring AGENTM_OBSERVABILITY_DIR.

    When the env var is set, returns ``Path(env_value)`` directly.
    When unset, falls back to ``<cwd>/.agentm/observability``.
    """
    env_dir = os.environ.get(_ENV_OBSERVABILITY_DIR)
    if env_dir:
        return Path(env_dir)
    base = Path(cwd) if cwd else Path.cwd()
    return base / ".agentm" / "observability"
