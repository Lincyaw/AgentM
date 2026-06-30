"""Agent manifest resolution for cognitive-audit child sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Final

_AGENTS_DIR: Final = Path(__file__).parent


def auditor_scenario() -> str:
    """Absolute path to the auditor agent directory (scenario-resolvable)."""
    return str(_AGENTS_DIR / "auditor")


def tel_scenario() -> str:
    """Absolute path to the TEL agent directory (scenario-resolvable)."""
    return str(_AGENTS_DIR / "tel")


def analyst_scenario() -> str:
    """Absolute path to the analyst agent directory (scenario-resolvable)."""
    return str(_AGENTS_DIR / "analyst")


__all__ = ["analyst_scenario", "auditor_scenario", "tel_scenario"]
