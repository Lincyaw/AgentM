"""Extension manifest — the module-level MANIFEST every atom exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

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


_PARSEABLE_KINDS: frozenset[str] = frozenset(get_args(CapabilityKind)) - {"unknown"}


def parse_capability_ref(value: str) -> CapabilityRef:
    """Parse one explicit ``kind:name`` capability reference."""

    kind, separator, name = value.partition(":")
    if not separator:
        raise ValueError(
            f"capability reference must use an explicit kind:name key: {value!r}"
        )
    if not name:
        raise ValueError(f"capability reference has an empty name: {value!r}")
    if kind in _PARSEABLE_KINDS:
        return CapabilityRef(kind=cast(CapabilityKind, kind), name=name)
    raise ValueError(f"unknown capability kind {kind!r} in {value!r}")


def requirement_key(value: str) -> str:
    """Return the normalized key for one manifest requirement."""

    return parse_capability_ref(value).key


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

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str
    description: str
    registers: tuple[str, ...] = ()
    config_schema: type[BaseModel] | None = None
    sensitive_config_fields: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    priority: int = AtomInstallPriority.NORMAL


__all__ = [
    "AtomInstallPriority",
    "CapabilityKind",
    "CapabilityRef",
    "ExtensionManifest",
    "parse_capability_ref",
    "provided_capability_keys",
    "requirement_key",
]
