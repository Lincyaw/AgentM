"""System prompt loader for the extractor child session.

The framing text lives in markdown files under ``prompts/`` (sibling to
this module). Pick a variant by name (default ``"default"``) or by an
absolute path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

DEFAULT_PROMPT_NAME = "default"


def _resolve(name_or_path: str) -> Path:
    candidate = name_or_path.strip()
    if not candidate:
        raise ValueError("empty prompt spec for extractor")
    if "/" in candidate or "\\" in candidate:
        path = Path(candidate).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"prompt file not found: {path}")
        return path
    for fname in (f"{candidate}.md", f"extractor_{candidate}.md"):
        path = _PROMPTS_DIR / fname
        if path.is_file():
            return path
    available = sorted(p.name for p in _PROMPTS_DIR.glob("*.md"))
    raise FileNotFoundError(f"unknown extractor prompt {candidate!r}; available: {available}")


@lru_cache(maxsize=64)
def _read(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def load_extractor_prompt(name_or_path: str = DEFAULT_PROMPT_NAME) -> str:
    """Load the extractor framing text for the given variant."""
    return _read(str(_resolve(name_or_path)))


__all__ = [
    "DEFAULT_PROMPT_NAME",
    "load_extractor_prompt",
]
