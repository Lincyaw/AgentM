"""What one installation has to put back, and who registered each codec.

Which atoms a session holds is not here and has no table of its own: an atom's
installation *is* its ``AtomContext`` (``atom_context.py``), so the session's
link list is the installed set, in the order the atoms joined.  Linking is
recording; unlinking is retiring; the spec that replays an atom and the name
its manifest gives it are fields of that one record.  A second list keyed by
module path would be a second answer to "which atoms does this session hold",
and two answers restored by two mechanisms can be driven apart — which is
exactly what happened, with no concurrency at all, when a third atom was
removed from inside an installation that then failed.

So what a failed installation restores is only what is here: the *contents* of
the stores an install writes into.  Everything about the shape of the context
tree is undone by an inverse instead.

``TriggerCodecOwners`` stays because it describes the one write an atom makes
that a context table cannot hold — see its docstring.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING
from dataclasses import dataclass

from agentm.core.abi.bus import EventBus
from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import ExtensionSpec
from agentm.core.abi.trajectory import AtomInstall

if TYPE_CHECKING:
    from agentm.core.runtime.session_core import SessionRuntime


@dataclass(frozen=True, slots=True)
class ExtensionInstallSnapshot:
    """The *contents* of the shared stores, before an install.

    Not the shape of the context tree.  A picture of a list put back at a
    moment other than the one it was taken at overwrites whatever anybody else
    decided in between, in both directions -- it resurrects a context a third
    party unlinked and erases one a third party linked -- so which contexts the
    session holds is undone by the inverse of what this install itself linked
    and unlinked instead.

    Nor the host's own tools, policies and renderers -- those three tables and
    no others, which is the whole of the claim and worth stating exactly,
    because the reachability it rests on is narrower than "a host table"
    sounds.  No path an *atom* can reach writes into any of them
    (``test_an_installation_writes_into_no_table_of_the_hosts``).  But every
    registration emits ``ApiRegisterEvent`` through ``bus.emit_sync``,
    re-entrantly, from inside the atom's registration call, so an embedder's
    **synchronous** handler runs on the install's stack and can write anywhere
    the session lets it.  Its writes into those three tables survive the
    install's failure; its ``services.register`` and its ``on`` do not,
    because the registry and the bus are still pictured below.  One handler,
    two fates, decided by which store it reached for.  An ``async def`` handler
    reaches neither -- ``emit_sync`` closes the coroutine and logs -- so the
    same embedder code is atomic, half-atomic or inert depending on ``def``
    versus ``async def`` and on the store.  Pinned by
    ``test_a_sync_register_handler_writes_into_the_host_during_an_install``.

    Nor the codec registry or the provider registry any more.  Those are the
    two stores an atom writes into that are not its own, and each write now
    records the inverse of itself -- a codec restores whatever source it
    displaced, a provider registration takes back its own ownership entry and
    lets the active name be resolved again.  Which is what a picture cannot
    do: an installation that failed while another finished beside it used to
    leave the survivor linked and installed with its codec and its provider
    ownership erased, so the session advertised an atom whose trigger source
    it could not encode.

    What is left is the two stores whose writes are not yet inverses: the
    bus's own subscriptions and the registry's own entries.  Both are host
    tables an atom does not write into -- an atom subscribes into its own
    segment and registers into its own registry, and both leave with the
    context -- so what these two put back is what the *host* wrote during the
    install, which is the re-entrant handler case above.  Two fields is the
    remaining distance between this rollback and an inverse throughout.
    """

    bus: EventBus
    services: ServiceRegistry

    @classmethod
    def of(cls, session: "SessionRuntime") -> "ExtensionInstallSnapshot":
        """Copy the stores an installation writes into, before it runs."""

        return cls(
            bus=session.bus.copy(),
            services=session.services.copy(),
        )

    def restore_into(self, session: "SessionRuntime") -> None:
        """Put the contents back, in place, leaving the link lists alone.

        In place throughout: the driver holds references to these same
        container objects, so rebinding an attribute would roll the session's
        view back while leaving the driver serving whatever the failed install
        appended.

        Contents only. The three lists that say which contexts the session
        holds -- the bus's segments, the registry's child registries, and the
        session's own -- are not restored from here at all. They move together
        through ``link_into``/``unlink_from``, and the rollback runs the
        inverse of what the installation itself linked and unlinked. A picture
        of them put back would undo whatever anybody else decided while the
        install was awaiting, in both directions.
        """

        session.bus.replace_from(self.bus)
        session.services.replace_from(self.services)


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
    """

    def __init__(self) -> None:
        self._pending: list[AtomInstall] = []

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
        """Withdraw an uncommitted install, or queue the atom's departure."""

        uncommitted = [
            install for install in self._pending if install.atom_name == atom_name
        ]
        if uncommitted:
            self._pending[:] = [
                install for install in self._pending if install.atom_name != atom_name
            ]
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
        """Take everything awaiting a turn to be recorded on."""

        drained = tuple(self._pending)
        self._pending.clear()
        return drained


__all__ = ["AtomInstallJournal", "ExtensionInstallSnapshot", "TriggerCodecOwners"]
