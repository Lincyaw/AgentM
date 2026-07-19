"""Extension manifest — the module-level MANIFEST every atom exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

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


CapabilityKind = Literal[
    "atom",
    "service",
    "tool",
    "event",
    "provider",
    "context_policy",
    "trigger_renderer",
    "trigger_codec",
    "operations",
    "resource",
    "trajectory",
    "catalog",
    "effect",
    "permission",
    "executor",
    "orchestrator",
    "query",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    """Normalized atom dependency/provision reference."""

    kind: CapabilityKind
    name: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"


def parse_capability_ref(value: str) -> CapabilityRef:
    """Parse a manifest capability string into a stable key.

    Strings with a prefix, such as ``tool:bash`` or ``service:operations``,
    keep that prefix. Bare values are interpreted as service capabilities; the
    factory also accepts ``atom:<name>`` for compatibility while manifests move
    toward capability-oriented dependencies.
    """

    kind, separator, name = value.partition(":")
    if not separator:
        return CapabilityRef(kind="service", name=value)
    if kind in {
        "atom",
        "service",
        "tool",
        "event",
        "provider",
        "context_policy",
        "trigger_renderer",
        "trigger_codec",
        "operations",
        "resource",
        "trajectory",
        "catalog",
        "effect",
        "permission",
        "executor",
        "orchestrator",
        "query",
    }:
        return CapabilityRef(kind=cast(CapabilityKind, kind), name=name)
    return CapabilityRef(kind="unknown", name=value)


def requirement_keys(value: str) -> tuple[str, ...]:
    """Return acceptable provider keys for one manifest requirement."""

    ref = parse_capability_ref(value)
    if ":" not in value:
        return (ref.key, f"atom:{value}")
    return (ref.key,)


def provided_capability_keys(
    *,
    atom_name: str,
    registers: tuple[str, ...],
) -> tuple[str, ...]:
    """Return all capability keys an atom contributes to dependency solving."""

    keys = [f"atom:{atom_name}"]
    keys.extend(parse_capability_ref(item).key for item in registers)
    return tuple(dict.fromkeys(keys))


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
    "CapabilityKind",
    "CapabilityRef",
    "ExtensionManifest",
    "parse_capability_ref",
    "provided_capability_keys",
    "requirement_keys",
]
