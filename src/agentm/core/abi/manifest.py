"""Extension manifest — the module-level MANIFEST every atom exports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AtomInstallPriority:
    """Conventional install priority bands.

    Smaller numbers install earlier. ``requires`` is still the hard ordering
    constraint; priority only orders otherwise independent atoms.
    """

    OBSERVABILITY = 100
    SERVICE = 200
    POLICY = 300
    PROVIDER = 400
    TOOL = 500
    CONTEXT = 600
    NORMAL = 500


class ExtensionManifest(BaseModel):
    """Declarative identity for an installable atom.

    The manifest describes the atom itself. Scenario selection, config source
    precedence, and hard composition policy belong to ``AgentSessionConfig``
    and its ``ScenarioLoader``. ``priority`` is only a stable default for
    otherwise independent atoms; ``requires`` wins.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    registers: tuple[str, ...] = ()
    config_schema: type[BaseModel] | None = None
    requires: tuple[str, ...] = ()
    priority: int = AtomInstallPriority.NORMAL


__all__ = [
    "AtomInstallPriority",
    "ExtensionManifest",
]
