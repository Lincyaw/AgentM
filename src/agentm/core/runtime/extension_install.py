"""Installed atoms — which atoms a session holds, and how to replay them.

What an atom *registered* is not here.  It went into that atom's own
``AtomContext`` (``atom_context.py``), the session sees it because the context
is linked, and detaching unlinks.  Ownership is therefore "which node did the
write land in", and there is nothing here that could give a second answer.

What is here is the other half, which no table of the session can answer for:
which atoms were installed, in which order, from which spec.  A child or a fork
is rebuilt by replaying those specs, and ``replace=True`` finds the incarnation
to supersede by manifest name.

Rollback is by inverse rather than by snapshot.  One installation records
exactly one atom, so undoing it retires exactly that record; superseding one
hands its record to the caller that took it out, which gives it back if the
replacement never lands.

``TriggerCodecOwners`` is the one attribution table left, for the one write an
atom makes that a context table cannot hold — see its docstring.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.session_api import ExtensionSpec


@dataclass(frozen=True, slots=True)
class InstalledAtom:
    """One installed atom: what replays it, and what it calls itself.

    ``runtime`` marks an atom installed into a running session rather than
    composed before start.  The distinction matters for rebuilds: the active
    set recorded at creation covers the composed atoms only, and a rebuild that
    replayed a runtime atom would compute a different digest than the one the
    source session froze into its provider identity.
    """

    module_path: str
    spec: ExtensionSpec
    runtime: bool
    atom_name: str | None


@dataclass(frozen=True, slots=True)
class RetiredAtom:
    """One record ``retire`` took out, and where it goes back.

    The inverse of recording an installation, held by whoever performed the
    removal.  The position travels with it because the installed set has an
    order and a rebuild replays in that order: a superseded atom whose
    replacement fails must come back where it was, not at the end.
    """

    atom: InstalledAtom
    position: int


class InstalledAtoms:
    """The atoms installed into one session, in the order they were installed."""

    __slots__ = ("_atoms",)

    def __init__(self) -> None:
        self._atoms: list[InstalledAtom] = []

    # --- Recording ---

    def record(
        self,
        spec: ExtensionSpec,
        *,
        runtime: bool = False,
        atom_name: str | None = None,
    ) -> None:
        """Record one installed atom, at the end of the installed set."""

        self._atoms.append(
            InstalledAtom(
                module_path=spec.module_path,
                spec=ExtensionSpec(source=spec.source, config=spec.config),
                runtime=runtime,
                atom_name=atom_name,
            )
        )

    def retire(self, module_path: str) -> RetiredAtom | None:
        """Take one atom out of the installed set and hand back its inverse.

        ``None`` when nothing was installed under that path, which is also the
        answer a detach of an atom that is not there needs.
        """

        for position, atom in enumerate(self._atoms):
            if atom.module_path == module_path:
                del self._atoms[position]
                return RetiredAtom(atom=atom, position=position)
        return None

    def reinstate(self, retired: RetiredAtom) -> None:
        """Put back what ``retire`` took out, where it was."""

        self._atoms.insert(retired.position, retired.atom)

    # --- Reading ---

    @property
    def module_paths(self) -> list[str]:
        """Module paths of the installed atoms, in install order."""

        return [atom.module_path for atom in self._atoms]

    def spec_module_paths(self) -> tuple[str, ...]:
        """Module paths of the replayable specs, in install order."""

        return tuple(atom.spec.module_path for atom in self._atoms)

    def runtime_module_paths(self) -> frozenset[str]:
        """Module paths installed into the running session."""

        return frozenset(atom.module_path for atom in self._atoms if atom.runtime)

    def atom_names(self) -> dict[str, str]:
        """Manifest name to module path, for the atoms that carry a manifest."""

        return {
            atom.atom_name: atom.module_path
            for atom in self._atoms
            if atom.atom_name is not None
        }

    def installed_module_path(self, atom_name: str) -> str | None:
        """Module path installed under ``atom_name``, if one is.

        Keyed on the manifest name rather than derived from the module path:
        a file-backed atom is loaded under a content-addressed module name, so
        two revisions of one atom share a manifest name and nothing else.
        """

        return self.atom_names().get(atom_name)

    def installed_atom_names(self) -> frozenset[str]:
        """Manifest names of the installed atoms, as a requirement spells them."""

        return frozenset(self.atom_names())

    def composition_extensions(
        self,
        *,
        excluded_module_paths: Collection[str] = (),
        include_runtime: bool = False,
    ) -> list[ExtensionSpec]:
        """Specs to replay when rebuilding this session's composition.

        Runtime-installed atoms are excluded by default: a rebuild replays these
        specs before the active set is recorded, so including them would make a
        child's digest disagree with the identity its source froze. A caller
        that wants the live picture rather than the composed one passes
        ``include_runtime=True``.
        """

        return [
            ExtensionSpec(source=atom.spec.source, config=atom.spec.config)
            for atom in self._atoms
            if atom.module_path not in excluded_module_paths
            and (include_runtime or not atom.runtime)
        ]


class TriggerCodecOwners:
    """Which atom registered each trigger codec.

    The one attribution table that survives the context model, because it
    describes the one registration an atom makes that a context table cannot
    hold.  Every other write goes into the atom's own tables and leaves when
    they do; a trigger codec is registered on the session's shared
    ``CodecRegistry`` and deliberately stays there after the atom has gone,
    because a committed turn names its trigger source by name and a session
    that could no longer decode it would fail to resume.  A record that has to
    outlive the context cannot be the context.

    Two readers need the retained entry.  ``is_superseded`` tells a replacement
    that it may take a source over rather than collide with it — the owner
    having left the installed set is exactly that signal.  ``composition_codec``
    keeps an atom's sources out of a child's registry, because a child replays
    the atoms and gets their codecs from the replay.

    ``owner`` is ``None`` for a source the host registered, which is the same
    "belongs to nobody" the context tree means by it.
    """

    __slots__ = ("_owners",)

    def __init__(self) -> None:
        self._owners: dict[str, str | None] = {}

    def note(self, source: str, owner: str | None) -> None:
        self._owners[source] = owner

    def owner(self, source: str) -> str | None:
        return self._owners.get(source)

    def is_superseded(self, source: str, installed: Collection[str]) -> bool:
        """True when this source is owned by an atom no longer installed."""

        owner = self._owners.get(source)
        return owner is not None and owner not in installed

    def composition_codec(self, codec: CodecRegistry) -> CodecRegistry:
        """``codec`` without the sources an atom registered."""

        return codec.copy_without_trigger_sources(
            {source for source, owner in self._owners.items() if owner is not None}
        )

    def copy(self) -> TriggerCodecOwners:
        """A detached copy, for a caller holding the shape before an install."""

        copied = TriggerCodecOwners()
        copied._owners = dict(self._owners)
        return copied

    def replace_from(self, other: TriggerCodecOwners) -> None:
        """Take another copy's contents in place, as ``CodecRegistry`` does.

        In place because a failed installation's rollback restores the codec
        registry the same way, and the two have to be put back together.
        """

        self._owners = dict(other._owners)


__all__ = [
    "InstalledAtom",
    "InstalledAtoms",
    "RetiredAtom",
    "TriggerCodecOwners",
]
