"""Extension manifest — the module-level MANIFEST every atom exports."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Final, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict


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


_PARSEABLE_KINDS: Final[frozenset[str]] = frozenset(get_args(CapabilityKind)) - {
    "unknown"
}


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


def live_capability_keys(
    *,
    services: Collection[str] = (),
    atoms: Collection[str] = (),
    tools: Collection[str] = (),
    providers: Collection[str] = (),
    trigger_renderers: Collection[str] = (),
) -> set[str]:
    """Spell a live composition's capabilities the way manifests spell them.

    A requirement is written in manifest vocabulary — ``atom:<manifest name>``,
    not the dotted or content-addressed module the atom happens to load under.
    Anything solving requirements against a running session has to name what
    the session holds in that same vocabulary, so the spelling lives here once
    rather than being restated by each solver.
    """

    keys = {f"service:{name}" for name in services}
    keys |= {f"atom:{name}" for name in atoms}
    keys |= {f"tool:{name}" for name in tools}
    keys |= {f"provider:{name}" for name in providers}
    keys |= {f"trigger_renderer:{source}" for source in trigger_renderers}
    return keys


class ExtensionManifest(BaseModel):
    """Declarative identity for an installable atom.

    The manifest describes the atom itself. Scenario selection, config source
    precedence, and hard composition policy belong to ``AgentSessionConfig``
    and its ``ScenarioLoader``.

    An atom declares what it needs, never where it sits.  Two words, because
    one was doing two jobs and the pair of them could not be stated apart:

    * ``requires`` is a hard dependency.  It decides satisfaction *and* order:
      absent, the composition fails with a message that reads as a fix.
    * ``after`` is order only.  "If this one is here, I come after it" --
      absent, nothing happens and nothing is said.

    Conflating them meant an optional dependency could not be declared at all:
    naming it made it mandatory, so every deliberate probe-and-degrade went
    undeclared, which in turn made the declarations untrustworthy -- reading a
    manifest did not tell you what the atom would touch.

    Neither word says where the atom sits among peers it declares nothing
    about.  That is not knowable from here: it depends on which peers are
    present, which is a fact about the composition rather than about this atom,
    and it is read from the order the scenario lists its extensions in.

    ``requires``, ``after`` and ``registers`` hold explicit ``kind:name``
    references. For
    a service boundary that has a ``ServiceRole``, spell it ``ROLE.capability``
    so the role stays the only place its key is written; anything else — a
    tool, an atom, an event, a service with no role descriptor — is written
    out, which is also what out-of-tree atoms already do.
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
    after: tuple[str, ...] = ()


__all__ = [
    "CapabilityKind",
    "CapabilityRef",
    "ExtensionManifest",
    "live_capability_keys",
    "parse_capability_ref",
    "provided_capability_keys",
    "requirement_key",
]
