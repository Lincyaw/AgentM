"""Resolve local observability output paths for the OTel/file backend."""

from __future__ import annotations

import os
from pathlib import Path

from agentm.core.lib.paths import expand_path

_ENV_AGENTM_HOME = "AGENTM_HOME"
_ENV_OBSERVABILITY_DIR = "AGENTM_OBSERVABILITY_DIR"


def file_export_requested() -> bool:
    """Return True when the user explicitly opted into local file export."""
    return bool(os.environ.get(_ENV_OBSERVABILITY_DIR))


def resolve_observability_dir() -> Path:
    """Return the observability directory, honoring AGENTM_OBSERVABILITY_DIR.

    When the env var is set, expands environment variables and ``~``.
    When unset, falls back to ``$AGENTM_HOME/observability`` and then
    ``~/.agentm/observability``.
    """
    env_dir = os.environ.get(_ENV_OBSERVABILITY_DIR)
    if env_dir:
        return expand_path(env_dir)
    home = os.environ.get(_ENV_AGENTM_HOME)
    if home:
        return expand_path(home) / "observability"
    return Path.home() / ".agentm" / "observability"
