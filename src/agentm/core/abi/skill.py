"""Skill record types — public ABI shared between the harness, the
``skill_loader`` atom, and any peer atom that contributes skill records via
the ``resources_discover`` event channel.

The dataclasses here carry no logic; they exist so atoms can speak the same
shape without reaching into ``core._internal``. The actual loading /
formatting helpers stay in ``_internal/skills.py`` and are exposed to atoms
through ``ExtensionAPI.skills``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    level: Literal["warning", "collision"]
    message: str
    path: str


@dataclass(frozen=True, slots=True)
class SkillRecord:
    name: str
    description: str
    file_path: str
    base_dir: str
    disable_model_invocation: bool
    source: str


__all__ = ["SkillDiagnostic", "SkillRecord"]
