"""The atom's own context: the tables it writes into, linked into a session.

An atom is not handed a facade over the session with a switch on it.  It is
handed *its own context*: a set of tables it alone writes into, plus an effect
log for the writes no table describes.  The session sees those writes because
the context is **linked** into it, and the session's visible state is the union
of its own tables and every linked context's.  Detaching an atom unlinks the
context and moves its tables out; there is no flag anywhere saying whether a
write is allowed.

Everything the previous attempts needed a guard for falls out of that:

*Attribution is not a parameter.*  There is no ``owner=`` to pass, to default
wrong, or to forge, because the only table an atom can reach is the one it was
handed.  Writing under another atom's name is not refused — it is unsayable.

*Reads keep working after a detach.*  Reading resolves up the chain, and being
unlinked means the parent no longer aggregates *your* writes, not that you
cannot see the parent's.  A superseded ``atom_watch`` reads its follower key,
misses in its own (now empty) table, resolves up, and finds the successor's
registration — which is the handover protocol.  On a plain uninstall the same
read resolves up, finds nothing, and returns ``None`` — which is the stop
signal.  One model answers both.

*A departed atom's surviving task writes into its own unlinked tables.*  Nobody
reads them, the session's stores never receive them, there is no orphan row to
clean up, and the write neither succeeds visibly nor raises.

That last sentence is about *table* writes, and the limit of it is worth
stating rather than leaving to be discovered.  A table write is inert once the
table is out of the session, because the write is the row.  ``api.effect`` is
not a table write: the body is arbitrary code and it runs where it is called,
so a departed atom that records an effect changes the world exactly as it
always did.  What departure takes away is the *audience* — the inverse lands in
a log the session no longer aggregates, so ``composition_digest`` does not
report it and no uninstall will run it, because the uninstall already happened.
Refusing the call instead would mean the log asking whether its context is
still linked, which is a revocation bit read at a write, and that is the shape
this design exists to remove.  What the session does instead is keep hold of
the effect logs of the contexts it has taken out (``DepartedContexts`` below)
and undo, at shutdown, whatever such a write recorded.  Taken out covers both
ways a context leaves: an atom that is detached, and an installation that
failed and was rolled back.  The rollback reverts that context's residue for
the same reason the detach does, so an effect recorded through it afterwards is
under the same decision, just late.  Past the session's own lifetime nothing
can be promised, and nothing here promises it.

*A failed ``replace=True`` has nothing to un-revert.*  Superseding moves the
previous context's tables aside rather than running their inverses, so the
rollback puts them back and relinks.  Linking is symmetric, reversible data.

What the effect log is for
--------------------------

Not for the writes that have a table.  A tool lives in ``_tools``; moving that
list out *is* the undo, and recording a second copy of it as an inverse would
be two accounts of one fact that can disagree.  The log holds exactly the
writes with no table:

* whatever an atom records through ``api.effect`` — a task, a connection, a
  patch: the escape hatch the platform is not supposed to enumerate;
* a write the runtime performs into shared state and deliberately keeps, which
  is recorded with a ``retain`` reason so the residue is visible rather than
  silent.  A trigger codec is the only one: a committed turn names its trigger
  source, and a session that could no longer decode it would fail to resume.
"""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentm.core.abi.bus import BusSegment, EventBusObserver, Handler
from agentm.core.abi.codec import TriggerCodec
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.effects import EffectBody, EffectHandle, EffectLog
from agentm.core.abi.operations import BashOperations, EnvironmentOperations
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.roles import BASH_OPERATIONS_ROLE, ENVIRONMENT_OPERATIONS
from agentm.core.abi.services import ServiceRegistry, ServiceScope
from agentm.core.abi.session_api import ExtensionSpec
from agentm.core.abi.tool import Tool
from agentm.core.abi.trigger import TriggerRenderer

if TYPE_CHECKING:
    from agentm.core.runtime.session_core import SessionRuntime

_POLICY_ORDER = itertools.count()
"""Ties within one context-policy priority band, across every context.

Policies compose in list order and two contexts cannot see each other's lists,
so the position a policy would have had in a single sorted list has to come
from somewhere both of them read.  Same shape and same reason as the bus's
subscription sequence.
"""


@dataclass(frozen=True, slots=True)
class PolicyRow:
    """One registered context policy, with what decides its position."""

    policy: ContextPolicy
    priority: int
    order: int


def policy_row(policy: ContextPolicy, priority: int) -> PolicyRow:
    """A row for ``policy``, numbered so its position is fixed at the write."""

    return PolicyRow(policy=policy, priority=priority, order=next(_POLICY_ORDER))


def _policy_key(row: PolicyRow) -> tuple[int, int]:
    return (row.priority, row.order)


_RENDERER_ORDER = itertools.count()
"""Which of two bindings of one trigger source is the later one.

Same shape and same reason as the service registry's write counter.  A trigger
source *resolves*: two contexts may bind it and exactly one renderer is used,
so the number decides an outcome, which is the test ``ChainedTools`` states for
why a tool name needs no such number.  Without it a source would resolve by
link order, and an atom rebinding a source it already holds would silently have
no effect while a later-linked context held it.
"""


@dataclass(frozen=True, slots=True)
class RendererRow:
    """One bound trigger renderer, with what decides which binding wins."""

    renderer: TriggerRenderer
    order: int


def renderer_row(renderer: TriggerRenderer) -> RendererRow:
    """A row for ``renderer``, numbered so its position is fixed at the write."""

    return RendererRow(renderer=renderer, order=next(_RENDERER_ORDER))


@dataclass(slots=True)
class ContextTables:
    """The tables one context writes into.

    Held as one object so the whole set moves together: the reason a context
    can be unlinked and put back is that "what it holds" is a thing with a
    boundary, rather than rows scattered through the session's stores.
    """

    tools: list[Tool] = field(default_factory=list)
    policies: list[PolicyRow] = field(default_factory=list)
    renderers: dict[str, RendererRow] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AtomResidue:
    """What one context held when it was unlinked, and what to do with it.

    Its fate is the caller's to decide and it has exactly two: ``revert`` for a
    detach that is final, or handing it back to the context for a rollback that
    puts the atom where it was.  Dropping one strands whatever the effect log
    still holds, which is why nothing here does it implicitly.
    """

    context: AtomContext
    tables: ContextTables
    services: ServiceRegistry
    segment: BusSegment
    effects: EffectLog

    def revert(self) -> tuple[BaseException, ...]:
        """Run the inverses of the writes that had no table, newest first.

        The tables themselves need nothing done to them: they are out of the
        session and out of the context, which is what "undone" means for a
        write whose whole existence was a row in one of them.

        Writes marked ``retain`` stay: this is an atom leaving, and what they
        name -- a trigger source a committed turn carries -- outlives it.
        """

        return self.effects.revert()

    def withdraw(self) -> tuple[BaseException, ...]:
        """Revert, retained writes included, for an install that never landed.

        A retention is a promise to whatever already names the write. An
        installation that failed left nothing able to name anything, so the
        promise has no one to keep and the write goes with the rest.
        """

        return self.effects.withdraw()

    def dispose(self) -> tuple[BaseException, ...]:
        """Drop the inverses without running them, closing any queued body.

        For a session that is going away: nothing will read what these undo,
        and running them would tear down state whose owner is about to stop
        existing anyway.
        """

        return self.effects.dispose()


@dataclass(frozen=True, slots=True)
class AtomDeparture:
    """What removing one atom took out, for the caller that decides its fate.

    ``residue`` is what the atom's context held, reverted by a detach that is
    final and handed back by a supersede whose replacement never lands, which
    is why it is not acted on at the removal.  ``None`` when no context was
    holding that module path.

    ``preceded_by`` is the rest of the inverse: the contexts linked ahead of
    this one, so a relink puts it back among its neighbours.  An integer index
    would not survive the list moving underneath it, and the installed set's
    order is replayed into every child, so an atom put back in the wrong place
    corrupts the composition of every session spawned afterwards.
    """

    residue: AtomResidue | None
    preceded_by: tuple[AtomContext, ...]


class AtomContext:
    """One atom's context — its tables, its install record, and its link state.

    Constructed by ``install_extension`` for one installation and given to that
    atom.  Two incarnations of the same module path get two contexts, so a task
    surviving from the first writes into the first's tables and the second
    inherits nothing.

    The context also *is* the installation record.  What replays this atom (its
    spec), what it calls itself (its manifest name), and whether it was
    composed or installed into a running session are fields here rather than
    rows in a second list keyed by module path, because a second list is a
    second answer to "which atoms does this session hold" and the two can be
    driven apart.  Linking a context is recording the install; unlinking is
    retiring it; there is nothing else to keep in step.
    """

    __slots__ = (
        # A session that has unlinked a context keeps a weak hold on it, so
        # that an effect recorded through it afterwards still has its inverse
        # run at shutdown.
        "__weakref__",
        "_atom_name",
        "_effects",
        "_installed",
        "_runtime",
        "_segment",
        "_services",
        "_session",
        "_spec",
        "_tables",
    )

    def __init__(
        self,
        session: SessionRuntime,
        spec: ExtensionSpec,
        *,
        runtime: bool = False,
    ) -> None:
        self._session = session
        self._spec = spec
        self._runtime = runtime
        self._atom_name: str | None = None
        self._installed = False
        self._tables = ContextTables()
        self._services = ServiceRegistry(parent=session.services)
        self._services.set_write_observer(self._observe_service_write)
        self._segment = session.bus.segment(spec.module_path)
        self._effects = EffectLog()

    @property
    def module_path(self) -> str:
        """Which atom this context belongs to.

        Read for diagnostics and for naming an owner to a reader that has to
        print one.  It is never an argument to a write: the write goes where
        the object is, not where a string says.
        """

        return self._spec.module_path

    @property
    def spec(self) -> ExtensionSpec:
        """What replays this atom into a child, a fork, or a resume."""

        return self._spec

    @property
    def runtime(self) -> bool:
        """Whether this atom was installed into an already-running session.

        The distinction matters for rebuilds: the active set recorded at
        creation covers the composed atoms only, and a rebuild that replayed a
        runtime atom would compute a different digest than the one the source
        session froze into its provider identity.
        """

        return self._runtime

    @property
    def atom_name(self) -> str | None:
        """What the manifest calls this atom, once its install has finished.

        Keyed on separately from the module path because a file-backed atom is
        loaded under a content-addressed module name, so two revisions of one
        atom share a manifest name and nothing else.  ``None`` until the
        install finishes, and for an atom that carries no manifest.
        """

        return self._atom_name

    @property
    def installed(self) -> bool:
        """Whether this atom's ``install()`` ran to completion.

        A context is linked *before* ``install()`` runs, so that what the atom
        writes is part of the session while it is writing it.  Between those
        two moments the atom is held but not installed, and the readers that
        mean "finished" rather than "held" — the replayable composition, and
        whether a trigger source may be taken over — say so by reading this.
        It is a field of the one record, not a second record: nothing can
        answer for a context that is not here to answer.
        """

        return self._installed

    def note_installed(self, atom_name: str | None) -> None:
        """Mark this installation finished, under the name its manifest gives.

        Called last by ``install_extension``, after everything that could fail
        has run, so a context carrying this flag is one whose install landed.
        """

        self._atom_name = atom_name
        self._installed = True

    @property
    def services(self) -> ServiceRegistry:
        """This context's own registry, resolving up into the session's."""

        return self._services

    @property
    def tables(self) -> ContextTables:
        return self._tables

    @property
    def segment(self) -> BusSegment:
        return self._segment

    @property
    def effects(self) -> EffectLog:
        return self._effects

    # --- Linking ---

    def link_into(
        self,
        session: SessionRuntime,
        *,
        after: Sequence[AtomContext] | None = None,
    ) -> None:
        """Make this context part of what the session resolves.

        The session's own list goes first because it is the one that refuses a
        second live context for one module path; refused there, nothing else
        has been linked yet and there is nothing to unwind.

        ``after`` is for a relink that has to land where the context was rather
        than at the end — see ``SessionRuntime.link_context``.  Only the
        session's list takes it: the bus orders subscriptions by ``(priority,
        seq)`` and the registry resolves a key by write order, so neither reads
        the position of a segment or a child registry in its own link list.
        """

        session.link_context(self, after=after)
        session.services.link(self._services)
        session.bus.link(self._segment)

    def unlink_from(self, session: SessionRuntime) -> None:
        """Stop the session aggregating this context; safe to repeat."""

        session.services.unlink(self._services)
        session.bus.unlink(self._segment)
        session.unlink_context(self)

    def suspend(self) -> AtomResidue:
        """Move everything this context holds out of it, leaving it empty.

        Moved rather than undone, which is the whole difference between a
        detach and a rollback: the caller that is superseding an atom can put
        the residue back if the replacement does not land, and the caller that
        is removing one reverts it.

        Emptying the context's own tables is also what makes a superseded
        atom's *reads* resolve up to whatever took its place, instead of
        finding its own stale registration.
        """

        tables = self._tables
        self._tables = ContextTables()
        return AtomResidue(
            context=self,
            tables=tables,
            services=self._services.take_own(),
            segment=self._segment.take(),
            effects=self._effects.take(),
        )

    def resume(self, residue: AtomResidue) -> None:
        """Take back everything ``suspend`` moved out.

        Anything written since goes after what is put back, which is the order
        the writes actually happened in.  Every row goes back as the object it
        was, so a policy's position and a renderer's binding are the ones they
        had rather than fresh ones that would sort or resolve differently.
        """

        if residue.context is not self:
            raise ValueError("an atom residue belongs to the context it came from")
        self._tables.tools[:0] = residue.tables.tools
        self._tables.policies[:0] = residue.tables.policies
        self._tables.policies.sort(key=_policy_key)
        restored_renderers = dict(residue.tables.renderers)
        restored_renderers.update(self._tables.renderers)
        self._tables.renderers.clear()
        self._tables.renderers.update(restored_renderers)
        self._services.give_own(residue.services)
        self._segment.give(residue.segment)
        self._effects.give(residue.effects)

    # --- Writes ---

    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = 500,
    ) -> Callable[[], None]:
        """Subscribe into this context's own bus segment."""

        return self._segment.on(channel, handler, priority=priority)

    def add_observer(self, observer: EventBusObserver) -> Callable[[], None]:
        """Attach a bus observer into this context's own segment."""

        return self._segment.add_observer(observer)

    def register_tool(self, tool: Tool) -> None:
        """Add a tool to this context's table.

        The duplicate check reads the session, not the table: a tool name is
        what the model calls, and two contexts advertising one name is the
        collision, wherever the other one lives.
        """

        if any(existing.name == tool.name for existing in self._session.tools):
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tables.tools.append(tool)
        self._emit_register_event("tool", tool.name, {"tool": tool})

    def register_context_policy(
        self,
        policy: ContextPolicy,
        *,
        priority: int = 500,
    ) -> None:
        """Add a context policy to this context's table, at ``priority``."""

        if any(existing is policy for existing in self._session.context_policies):
            raise ValueError("context policy instance is already registered")
        row = policy_row(policy, priority)
        self._tables.policies.append(row)
        self._tables.policies.sort(key=_policy_key)
        if self._session.driver_running:
            try:
                self._session.bind_context_policy(policy, services=self._services)
            except BaseException:
                self._tables.policies[:] = [
                    held for held in self._tables.policies if held is not row
                ]
                raise
        self._emit_register_event(
            "context_policy",
            type(policy).__name__,
            {"policy": policy, "priority": priority},
        )

    def register_trigger_renderer(
        self,
        source: str,
        renderer: TriggerRenderer,
    ) -> None:
        """Bind a trigger source to a renderer in this context's table.

        The write lands here whatever else holds the source. Who the source
        then *belongs* to is not decided at the write and is not recorded
        anywhere: a source resolves to the highest write order in the chain, so
        a binding that lands under an existing later one changes nothing the
        session serves, and every reader resolves that for itself.
        """

        self._tables.renderers[source] = renderer_row(renderer)
        self._emit_register_event("trigger_renderer", source, {"renderer": renderer})

    def register_trigger_codec(self, source: str, codec: object) -> None:
        """Register a codec on the session's codec registry, and keep it there.

        The one write here that does not go into this context's own table. A
        codec cannot be a context table because a committed turn names its
        trigger source by name and must stay decodable after the atom that
        registered it has gone, so the write lands on the session and is
        recorded with the reason it stays.

        It hands back an inverse anyway, which is not a contradiction: the
        retention is a promise to whatever already names the source, and an
        installation that failed left nothing able to name it. So the write
        survives the atom *leaving* and is taken back when the atom never
        arrived -- ``revert`` and ``withdraw`` respectively. Restoring what was
        there before rather than merely dropping it, because a supersede
        registers over a source its previous incarnation owns.
        """

        if not isinstance(codec, TriggerCodec):  # code-health: ignore[AM025]
            raise TypeError("trigger codec must implement serialize and deserialize")
        session = self._session
        module_path = self.module_path

        def _register() -> Callable[[], None]:
            displaced = session.codec.trigger_codec(source)
            displaced_owner = session._codec_owners.owner(source)
            session.note_atom_trigger_codec(source, codec, module_path)
            self._emit_register_event("trigger_codec", source, {"codec": codec})

            def _undo() -> None:
                if displaced is None:
                    session.codec.forget_trigger_codec(source)
                    session._codec_owners.forget(source)
                    return
                session.codec.register_trigger_codec(source, displaced, replace=True)
                session._codec_owners.note(source, displaced_owner)

            return _undo

        self._effects.effect(
            _register,
            provides=f"trigger_codec:{source}",
            subject=codec,
            retain="a committed turn names this trigger source",
        )

    def register_operations(
        self,
        *,
        replace: bool = False,
        service_scope: ServiceScope = "session",
        **kwargs: object,
    ) -> None:
        """Register named operation services into this context's registry."""

        protocols: dict[str, type | None] = {
            "bash": BashOperations,
            "environment": EnvironmentOperations,
        }
        service_names = {
            "bash": BASH_OPERATIONS_ROLE.key,
            "environment": ENVIRONMENT_OPERATIONS.key,
        }
        for key, value in kwargs.items():
            service_name = service_names.get(key, f"operations:{key}")
            if self._services.has(service_name) and not replace:
                raise ValueError(f"operation {key!r} already registered")
            # No unregister first: writing this context's own key shadows
            # whatever the chain resolved to, and the value it shadows comes
            # back by itself when this context is unlinked.
            self._services.register(
                service_name,
                value,
                protocols.get(key),
                scope=service_scope,
            )
            self._emit_register_event(
                "operations",
                key,
                {"service_name": service_name, "service": value},
            )

    def register_provider(
        self,
        name: str,
        config: ProviderConfig,
        *,
        replace: bool = False,
    ) -> None:
        """Register an LLM provider whose backing service is this context's.

        The config lands in this context's own registry and leaves with it, but
        the registry's ownership index and its active name are the session's,
        so the write is recorded with the inverse that takes those back. An
        installation that fails then undoes its own provider registrations
        rather than a picture of the registry being put back over everyone's.
        """

        providers = self._session._providers

        def _register() -> Callable[[], None]:
            return providers.register(
                name,
                config,
                replace=replace,
                into=self._services,
                owner=self.module_path,
            )

        self._effects.effect(
            _register,
            provides=f"provider:{name}",
            subject=config,
            retain="the session keeps streaming through the model it resolved",
        )

    def effect(
        self,
        body: EffectBody,
        *,
        provides: str = "",
        retain: str = "",
        subject: object = None,
    ) -> EffectHandle:
        """Record a write the platform has no table for, with its inverse."""

        return self._effects.effect(
            body,
            provides=provides,
            retain=retain,
            subject=subject,
        )

    async def settle(self) -> None:
        """Run the async effect bodies queued in this context's log."""

        await self._effects.settle()

    # --- Internals ---

    def _observe_service_write(
        self,
        key: str,
        service: object,
        scope: ServiceScope,
        *,
        role_bind: bool,
    ) -> None:
        """Let the session announce one write of this context's registry.

        The session names this context because this observer belongs to this
        context; there is no argument saying so and no ambient state consulted.
        """

        self._session.announce_service_write(
            key,
            service,
            scope,
            role_bind=role_bind,
            context=self,
        )

    def _emit_register_event(
        self,
        kind: str,
        name: str,
        payload: dict[str, object],
    ) -> None:
        self._session.emit_register_event(
            kind,
            name,
            payload,
            owner=self.module_path,
        )

    def __repr__(self) -> str:
        return (
            f"AtomContext({self.module_path!r}, tools={len(self._tables.tools)}, "
            f"policies={len(self._tables.policies)}, "
            f"services={len(self._services.own_table())}, "
            f"effects={len(self._effects)})"
        )


class ChainedTools(Sequence[Tool]):
    """The session's tools: its own, then each linked context's, in link order.

    A live view rather than a rebuilt list.  The driver takes this once at
    start and re-reads it at every turn boundary, which is what makes an atom
    installed mid-run advertise its tools from the next turn.

    Position here is link position, not write order, and the difference is
    visible: a host tool registered after an atom sorts *before* that atom's,
    where a single appended list would have put it last.  It is left that way
    on purpose.  Services and policies are numbered per write because their
    number decides an outcome — which of two writes to one key wins, and where
    a policy sits in a composed chain.  A tool name resolves to nothing: two
    tools cannot share one, ``register_tool`` refuses the collision outright,
    and what the list order decides is only the sequence the model is shown
    them in.  Numbering every tool to reproduce an interleaving nothing reads
    would be a second ordering account to keep in step with this one.
    """

    __slots__ = ("_linked", "_own")

    def __init__(self, own: list[Tool], linked: list[AtomContext]) -> None:
        self._own = own
        self._linked = linked

    def _rows(self) -> list[Tool]:
        if not self._linked:
            return self._own
        rows = list(self._own)
        for context in self._linked:
            rows.extend(context.tables.tools)
        return rows

    def __len__(self) -> int:
        return len(self._rows())

    def __getitem__(self, index: int) -> Tool:  # type: ignore[override]
        return self._rows()[index]

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._rows())

    def __repr__(self) -> str:
        return f"ChainedTools({[tool.name for tool in self._rows()]!r})"


class ChainedPolicies(Sequence[ContextPolicy]):
    """The session's context policies, ordered by ``(priority, write order)``.

    Sorting across contexts rather than concatenating them: a policy's position
    is what decides how it composes, and two contexts registering at the same
    priority must interleave the way one list would have.
    """

    __slots__ = ("_linked", "_own")

    def __init__(self, own: list[PolicyRow], linked: list[AtomContext]) -> None:
        self._own = own
        self._linked = linked

    def _rows(self) -> list[PolicyRow]:
        if not self._linked:
            return self._own
        rows = list(self._own)
        for context in self._linked:
            rows.extend(context.tables.policies)
        rows.sort(key=_policy_key)
        return rows

    def priority_of(self, policy: ContextPolicy) -> int | None:
        """The priority one policy was registered at, or None if it is gone.

        A public read on the object ``session.context_policies`` hands an
        embedder: the view exposes policies, and a host that wants to know
        where one sits in the composed chain has nowhere else to ask.
        """

        for row in self._rows():
            if row.policy is policy:
                return row.priority
        return None

    def __len__(self) -> int:
        return len(self._rows())

    def __getitem__(self, index: int) -> ContextPolicy:  # type: ignore[override]
        return self._rows()[index].policy

    def __iter__(self) -> Iterator[ContextPolicy]:
        return iter(row.policy for row in self._rows())

    def __repr__(self) -> str:
        return f"ChainedPolicies({[type(p).__name__ for p in self]!r})"


class ChainedRenderers(Mapping[str, TriggerRenderer]):
    """The session's trigger renderers: the latest binding of each source.

    Resolved by write order across the host's table and every linked context's,
    the way services are, because a source resolves to exactly one renderer and
    the number is what picks it.  That is last-writer-wins, which is what the
    single table this replaced gave: an atom that rebinds a source it already
    holds takes it back, and linking a context cannot change what a source
    already resolved to except by writing it.  Unlinking gives the shadowed
    binding back.
    """

    __slots__ = ("_linked", "_own")

    def __init__(
        self,
        own: dict[str, RendererRow],
        linked: list[AtomContext],
    ) -> None:
        self._own = own
        self._linked = linked

    def _rows(self) -> dict[str, RendererRow]:
        if not self._linked:
            return self._own
        rows = dict(self._own)
        for context in self._linked:
            for source, row in context.tables.renderers.items():
                held = rows.get(source)
                if held is None or row.order > held.order:
                    rows[source] = row
        return rows

    def __getitem__(self, key: str) -> TriggerRenderer:
        return self._rows()[key].renderer

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows())

    def __len__(self) -> int:
        return len(self._rows())

    def __repr__(self) -> str:
        return f"ChainedRenderers({sorted(self._rows())!r})"


@dataclass(frozen=True, slots=True)
class ContextOwnership:
    """Who holds what, read off the context tree.

    The one account of ownership this design has: a thing belongs to the
    context whose table it is in.  Built fresh for each reader that has to
    state ownership — the composition digest, the provider registry's uncover
    path — rather than maintained, because a maintained index would have to be
    told every time unlinking a context hands a key back to whoever it shadowed.
    """

    tools: dict[int, str]
    policies: dict[int, str]
    renderers: dict[str, str]
    services: dict[str, str]

    def tool(self, tool: Tool) -> str | None:
        return self.tools.get(id(tool))

    def policy(self, policy: ContextPolicy) -> str | None:
        return self.policies.get(id(policy))

    def renderer(self, source: str) -> str | None:
        return self.renderers.get(source)

    def service(self, key: str) -> str | None:
        return self.services.get(key)


def ownership_of(
    own: ServiceRegistry,
    own_renderers: Mapping[str, RendererRow],
    linked: Sequence[AtomContext],
) -> ContextOwnership:
    """Index which linked context holds each tool, policy, renderer, service.

    Every table here is indexed the way that table is *resolved*, because an
    index that ordered its candidates differently from the reader it describes
    would name one context while the session served another's value.

    Services and renderers are the two tables where that is not link order.  A
    service key resolves to the highest ``ServiceEntry.order`` anywhere in the
    chain and a trigger source to the highest ``RendererRow.order`` — write
    order, so that linking a context cannot change what either already resolved
    to — and a context can rebind one long after a later-linked context wrote
    it.  So each winner is picked by the same number its reader picks it by, and
    the host's own tables take part: a host write later than every atom's wins
    and belongs to nobody, which is what a ``None`` owner means.

    Tools and policies need no tiebreak: they are keyed by identity, and no two
    contexts can hold the same object.
    """

    tools: dict[int, str] = {}
    policies: dict[int, str] = {}
    renderers: dict[str, str] = {}
    services: dict[str, str] = {}
    winning: dict[str, int] = {
        key: entry.order for key, entry in own.own_table().items()
    }
    bound: dict[str, int] = {source: row.order for source, row in own_renderers.items()}
    for context in linked:
        owner = context.module_path
        for tool in context.tables.tools:
            tools[id(tool)] = owner
        for row in context.tables.policies:
            policies[id(row.policy)] = owner
        for source, binding in context.tables.renderers.items():
            latest = bound.get(source)
            if latest is None or binding.order > latest:
                bound[source] = binding.order
                renderers[source] = owner
        for key, entry in context.services.own_table().items():
            held = winning.get(key)
            if held is None or entry.order > held:
                winning[key] = entry.order
                services[key] = owner
    return ContextOwnership(
        tools=tools,
        policies=policies,
        renderers=renderers,
        services=services,
    )


@dataclass(frozen=True, slots=True)
class _Departure:
    """One removed context: what may still write, and what it has written."""

    context: weakref.ref[AtomContext]
    effects: EffectLog

    def spent(self) -> bool:
        """Nothing can record another inverse here, and none is held."""

        return (
            self.context() is None
            and not self.effects.entries
            and not self.effects.unsettled
        )


class DepartedContexts:
    """The contexts a session has unlinked, and the inverses they still hold.

    Here for the one write of a departed atom that is not inert.  A table write
    into an unlinked context reaches nobody because the write *is* the row; an
    effect body is arbitrary code that runs where it is called, so recording
    one after departure changes something and leaves the inverse in a log the
    session no longer aggregates.  Refusing that call instead would be the log
    reading a revocation bit, which is the shape this design removes.

    So a departure is held two ways, and the difference between them is the
    difference between a write that may yet happen and one that already has.
    The context is held *weakly*: a context nobody holds has nobody left to
    write through it, and a session that supersedes an atom on a timer must not
    accumulate every incarnation it ever removed.  Its effect log is held
    *outright*, because the log is the promise — an inverse that has been
    recorded is owed whether or not anything can still reach the context that
    recorded it, and a weak hold on that would lose the undo of a write already
    made.  A departure is dropped only when both are spent.
    """

    __slots__ = ("_departures",)

    def __init__(self) -> None:
        self._departures: list[_Departure] = []

    def note(self, context: AtomContext) -> None:
        """Record one departure, dropping the ones with nothing left in them."""

        self._prune()
        self._departures.append(
            _Departure(context=weakref.ref(context), effects=context.effects)
        )

    def forget(self, context: AtomContext) -> None:
        """Drop one context, for a rollback that has linked it again.

        Its log goes with it, which is right: a context this session holds
        again is abandoned with the composition at shutdown like any other, not
        undone separately.
        """

        self._departures[:] = [
            departure
            for departure in self._departures
            if departure.context() is not context
        ]
        self._prune()

    def undo(self) -> tuple[BaseException, ...]:
        """Revert what each departed context recorded, newest departure first.

        Reverted rather than released, which is the difference from a context
        that is still linked when the session shuts down: that one is being
        abandoned along with the composition it belongs to, while this one's
        removal already decided that what it wrote comes back out — and a write
        made after that decision is under the same decision, just late.

        The log is reverted directly rather than through the context, because
        the context may be gone and the log is what the inverses are in.
        """

        failures: list[BaseException] = []
        for departure in reversed(self._departures):
            failures.extend(departure.effects.revert())
        self._departures.clear()
        return tuple(failures)

    def _prune(self) -> None:
        self._departures[:] = [
            departure for departure in self._departures if not departure.spent()
        ]


__all__ = [
    "AtomContext",
    "AtomDeparture",
    "AtomResidue",
    "ChainedPolicies",
    "ChainedRenderers",
    "ChainedTools",
    "ContextOwnership",
    "ContextTables",
    "DepartedContexts",
    "PolicyRow",
    "RendererRow",
    "ownership_of",
    "policy_row",
    "renderer_row",
]
