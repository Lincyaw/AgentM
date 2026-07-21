"""Executable dependencies provisioned in remote agent environments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version


@dataclass(frozen=True, slots=True)
class ToolboxDependency:
    distribution: str
    executable: str

    @property
    def requirement(self) -> str:
        return f"{self.distribution}=={version(self.distribution)}"


REMOTE_DEPENDENCIES = (
    ToolboxDependency(distribution="ast-grep-cli", executable="ast-grep"),
)


__all__ = ["REMOTE_DEPENDENCIES", "ToolboxDependency"]
