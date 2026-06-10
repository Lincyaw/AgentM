"""Extension manifest — the module-level MANIFEST every atom exports."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ExtensionManifest(BaseModel):
    """Module-level declaration every atom exports as MANIFEST."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    registers: tuple[str, ...] = ()
    config_schema: dict[str, Any] | None = None
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    api_version: int = 1
    affects: tuple[str, ...] = ()
    tier: int = 1
    mountable_via_command: bool = False
    provides_role: tuple[str, ...] = ()


__all__ = [
    "ExtensionManifest",
]
