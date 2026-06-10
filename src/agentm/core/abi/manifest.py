"""Extension manifest — the module-level MANIFEST every atom exports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtensionManifest(BaseModel):
    """Module-level declaration every atom exports as MANIFEST."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    registers: tuple[str, ...] = ()
    config_schema: type[BaseModel] | None = None
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
