"""Who registered each trigger codec, and what the next turn will say.

A failed installation restores nothing from here, or from anywhere.  There is
no snapshot of a session taken before an install, because there is no store an
installation writes into that it does not own: an atom's tools, policies,
renderers, services and subscriptions live in its own ``AtomContext``
(``atom_context.py``), and its two writes that land outside it — a trigger
codec and a provider registration — each record the inverse of themselves.
Unlinking the context is the rest.

That is not a tidiness argument.  A picture of a store, put back at a moment
other than the one it was taken at, cannot tell whose decision it is reversing.
It resurrected a context a third party had unlinked; it erased the codec and
the provider ownership of an installation that had *finished* while the failing
one was awaiting, leaving the session advertising an atom whose trigger source
it could no longer encode; and it undid an embedder's own writes, made from a
handler running re-entrantly on the install's stack.  Each of those is the same
defect, and an inverse cannot have any of them, because an inverse names what
it undoes.

Which atoms a session holds is likewise not here and has no table of its own.
Linking a context is recording the installation; unlinking it is retiring it;
the spec that replays an atom and the name its manifest gives it are fields of
that one record.

``TriggerCodecOwners`` stays because it describes the one *attribution* a
context table cannot hold — see its docstring.  ``AtomInstallJournal`` is the
other thing here that outlives a context on purpose: what the next committed
turn will say about the session's atom set.
"""

from __future__ import annotations

from collections.abc import Collection

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.session_api import ExtensionSpec
from agentm.core.abi.trajectory import AtomInstall


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

    def forget(self, source: str) -> None:
        """Drop a source's attribution; safe to repeat.

        The counterpart of ``CodecRegistry.forget_trigger_codec``, and for the
        same one case: an installation that never landed.  An atom that leaves
        keeps its entry, which is what lets its replacement take the source
        over.
        """

        self._owners.pop(source, None)

    def is_superseded(self, source: str, installed: Collection[str]) -> bool:
        """True when this source is owned by an atom no longer installed."""

        owner = self._owners.get(source)
        return owner is not None and owner not in installed

    def composition_codec(self, codec: CodecRegistry) -> CodecRegistry:
        """``codec`` without the sources an atom registered."""

        return codec.copy_without_trigger_sources(
            {source for source, owner in self._owners.items() if owner is not None}
        )


class AtomInstallJournal:
    """What the next committed turn will say about this session's atom set.

    The composition describes the atoms a session was created with, so an atom
    installed into a running one is carried by the turn that commits after it
    and by nothing else.  That makes the record a queue rather than a table:
    entries wait here until a turn takes them, and until then nothing durable
    has been said.

    Which is what makes leaving expressible.  An install still waiting is
    withdrawn -- no history was written, so there is none to correct, and a
    committed pair that cancelled out would be two records saying nothing.  An
    install a turn already carries is history, and history is appended to
    rather than rewritten, so the atom's departure is queued as a record of its
    own.  A resume keeps the last record per atom name, so what a session
    reopens with is what the trajectory last said about each atom.

    Withdrawing therefore has to ask what the history already says, and not
    only what is in the queue.  Supersede an atom that a turn has carried and
    the queue holds a *second* install of the same name; take the atom out and
    dropping that one leaves the first still standing in the record, so the
    session reopens running the version it replaced -- which is a version the
    user has not had since before the supersede.  So the drained names are kept
    and the queue answers against them: nothing committed means nothing to
    correct, and anything committed means the departure has to be said.
    """

    __slots__ = ("_committed", "_pending")

    def __init__(self) -> None:
        self._pending: list[AtomInstall] = []
        #: Atom names the record currently says this session holds. Written by
        #: ``drain``, because being carried by a turn is what makes an install
        #: history, and that is the only moment it becomes true.
        self._committed: set[str] = set()

    def pending_names(self) -> tuple[str, ...]:
        """Atom names awaiting a turn, in queue order."""

        return tuple(install.atom_name for install in self._pending)

    def note_install(self, atom_name: str, spec: ExtensionSpec) -> None:
        """Queue a durable record of one runtime install."""

        self._pending.append(
            AtomInstall(
                atom_name=atom_name,
                source_kind=spec.source.kind,
                location=spec.source.location,
                digest=spec.source.digest,
                config=spec.config,
            )
        )

    def note_retire(self, atom_name: str, spec: ExtensionSpec) -> None:
        """Withdraw an uncommitted install, and say so if the record needs it.

        The queued install goes either way: it describes an arrival that is
        over, and a committed pair that cancelled out would be two records
        saying nothing.  Whether a departure is queued in its place is a
        question about the *record*, not about the queue -- history is what a
        resume replays, and it is corrected by appending or not at all.
        """

        self._pending[:] = [
            install for install in self._pending if install.atom_name != atom_name
        ]
        if atom_name not in self._committed:
            return
        self._pending.append(
            AtomInstall(
                atom_name=atom_name,
                source_kind=spec.source.kind,
                location=spec.source.location,
                digest=spec.source.digest,
                config=spec.config,
                retired=True,
            )
        )

    def drain(self) -> tuple[AtomInstall, ...]:
        """Take everything awaiting a turn to be recorded on.

        Draining is the moment a queued record becomes history, so it is also
        the moment the drained names become what the record says -- an install
        puts the atom in, a retirement takes it out, which is exactly what the
        resume reads back off the last record per name.
        """

        drained = tuple(self._pending)
        self._pending.clear()
        for install in drained:
            if install.retired:
                self._committed.discard(install.atom_name)
            else:
                self._committed.add(install.atom_name)
        return drained


__all__ = ["AtomInstallJournal", "TriggerCodecOwners"]
