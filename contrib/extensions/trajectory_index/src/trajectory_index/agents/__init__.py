"""Agent manifest resolution for trajectory-index child sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Final

_AGENTS_DIR: Final = Path(__file__).resolve().parent


def extractor_scenario() -> str:
    """Absolute path to the entity-extractor agent directory."""
    return str(_AGENTS_DIR / "entity_extractor")
