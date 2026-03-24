"""Data structures for trajectory analysis scenario."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillCatalogEntry:
    """A discovered skill from the vault catalog.

    Holds frontmatter metadata from a SKILL.md file under
    ``skill/trajectory-analysis/``.
    """

    name: str
    description: str
    path: str  # vault path, e.g. "skill/trajectory-analysis/memory-extraction"
