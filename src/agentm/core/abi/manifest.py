"""Extension manifest — the module-level MANIFEST every atom exports."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict


class ChannelEffects(BaseModel):
    """Declared handler effects on one event channel.

    Resource names are objective identifiers of shared session state the
    handler touches (e.g. ``"tools"``, ``"system"``). Three effect kinds:

    - ``mutates`` — rewrites or filters the resource; ordering against
      readers is significant.
    - ``reads`` — consumes the resource; on the same channel a reader must
      run after every declared mutator of that resource.
    - ``appends`` — adds content commutatively; appenders are unordered
      among themselves and do not count as mutators.

    Declarations are opt-in: an extension without ``effects`` is
    unconstrained. The loader validates the resolved, ordered extension
    list at load time (see ``agentm.extensions.validate``).
    """

    model_config = ConfigDict(frozen=True)

    mutates: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    appends: tuple[str, ...] = ()


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
    effects: Mapping[str, ChannelEffects] = {}
    """Optional per-channel effect declarations, keyed by event channel name.

    Empty mapping (the default) means no declaration and no ordering
    constraints — fully backward compatible.
    """


__all__ = [
    "ChannelEffects",
    "ExtensionManifest",
]
