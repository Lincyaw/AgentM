"""Agent manifest resolution for cognitive-audit child sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Final

_AGENTS_DIR: Final = Path(__file__).resolve().parent


def extractor_scenario() -> str:
    """Absolute path to the extractor agent directory (scenario-resolvable)."""
    return str(_AGENTS_DIR / "extractor")


def auditor_scenario() -> str:
    """Absolute path to the auditor agent directory (scenario-resolvable)."""
    return str(_AGENTS_DIR / "auditor")


__all__ = ["auditor_scenario", "extractor_scenario"]
