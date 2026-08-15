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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentm.core.abi.bus import BusSegment, EventBus, EventBusObserver, Handler
from agentm.core.abi.codec import TriggerCodec
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.effects import EffectBody, EffectHandle, EffectLog
from agentm.core.abi.operations import BashOperations, EnvironmentOperations
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.roles import BASH_OPERATIONS_ROLE, ENVIRONMENT_OPERATIONS
from agentm.core.abi.services import ServiceRegistry, ServiceScope
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


@dataclass(slots=True)
class ContextTables:
    """The tables one context writes into.

    Held as one object so the whole set moves together: the reason a context
    can be unlinked and put back is that "what it holds" is a thing with a
    boundary, rather than rows scattered through the session's stores.
    """

    tools: list[Tool] = field(default_factory=list)
    policies: list[PolicyRow] = field(default_factory=list)
    renderers: dict[str, TriggerRenderer] = field(default_factory=dict)


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
        """

        return self.effects.revert()

    def dispose(self) -> tuple[BaseException, ...]:
        """Drop the inverses without running them, closing any queued body.

        For a session that is going away: nothing will read what these undo,
        and running them would tear down state whose owner is about to stop
        existing anyway.
        """

        return self.effects.dispose()


class AtomContext:
    """One atom's context — its tables, its effect log, and its link state.

    Constructed by ``install_extension`` for one installation and given to that
    atom.  Two incarnations of the same module path get two contexts, so a task
    surviving from the first writes into the first's tables and the second
    inherits nothing.
    """

    __slots__ = (
        "_module_path",
        "_effects",
        "_segment",
        "_services",
        "_session",
        "_tables",
    )

    def __init__(self, session: SessionRuntime, module_path: str) -> None:
        self._session = session
        self._module_path = module_path
        self._tables = ContextTables()
        self._services = ServiceRegistry(parent=session.services)
        self._services.set_write_observer(self._observe_service_write)
        self._segment = session.bus.segment(module_path)
        self._effects = EffectLog()

    @property
    def module_path(self) -> str:
        """Which atom this context belongs to.

        Read for diagnostics and for the install ledger's parallel account of
        the same ownership.  It is never an argument to a write: the write goes
        where the object is, not where a string says.
        """

        return self._module_path

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

    def link_into(self, session: SessionRuntime) -> None:
        """Make this context part of what the session resolves."""

        session.services.link(self._services)
        session.bus.link(self._segment)
        session.link_context(self)

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
        the writes actually happened in.
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
        self._session._extensions.note_tool(tool, self._module_path)
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
        self._session._extensions.note_context_policy(
            policy,
            self._module_path,
            priority=priority,
        )
        if self._session.driver_running:
            try:
                self._session.bind_context_policy(policy, services=self._services)
            except BaseException:
                self._tables.policies[:] = [
                    held for held in self._tables.policies if held is not row
                ]
                self._session._extensions.drop_context_policy(policy)
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
        """Bind a trigger source to a renderer in this context's table."""

        self._tables.renderers[source] = renderer
        self._session._extensions.note_trigger_renderer(source, self._module_path)
        self._emit_register_event("trigger_renderer", source, {"renderer": renderer})

    def register_trigger_codec(self, source: str, codec: object) -> None:
        """Register a codec on the session's codec registry, and keep it there.

        The one write here that does not go into this context's own table. A
        codec cannot be a context table because a committed turn names its
        trigger source by name and must stay decodable after the atom that
        registered it has gone, so the write lands on the session and is
        recorded with the reason it stays.
        """

        if not isinstance(codec, TriggerCodec):  # code-health: ignore[AM025]
            raise TypeError("trigger codec must implement serialize and deserialize")
        session = self._session
        module_path = self._module_path

        def _register() -> None:
            # A superseded atom keeps its codec registered so committed turns
            # stay decodable; its replacement takes the source over rather than
            # colliding.
            session.codec.register_trigger_codec(
                source,
                codec,
                replace=session._extensions.trigger_codec_is_superseded(source),
            )
            session._extensions.note_trigger_codec(source, module_path)
            self._emit_register_event("trigger_codec", source, {"codec": codec})

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
        """Register an LLM provider whose backing service is this context's."""

        self._session._providers.register(
            name,
            config,
            replace=replace,
            into=self._services,
            owner=self._module_path,
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
        """Mirror one write of this context's registry into the session.

        The session attributes it to this context because this observer belongs
        to this context; there is no argument saying so and no ambient state
        consulted.
        """

        self._session.note_service_write(
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
            owner=self._module_path,
        )

    def __repr__(self) -> str:
        return (
            f"AtomContext({self._module_path!r}, tools={len(self._tables.tools)}, "
            f"policies={len(self._tables.policies)}, "
            f"services={len(self._services.own_names())}, "
            f"effects={len(self._effects)})"
        )


class ChainedTools(Sequence[Tool]):
    """The session's tools: its own, then each linked context's, in link order.

    A live view rather than a rebuilt list.  The driver takes this once at
    start and re-reads it at every turn boundary, which is what makes an atom
    installed mid-run advertise its tools from the next turn.
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
        """The priority one policy was registered at, or None if it is gone."""

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
    """The session's trigger renderers: its own, overlaid by linked contexts.

    A later-linked context wins a source the session or an earlier one already
    bound, which is what a single last-writer-wins table did.  Unlinking gives
    the shadowed binding back.
    """

    __slots__ = ("_linked", "_own")

    def __init__(
        self,
        own: dict[str, TriggerRenderer],
        linked: list[AtomContext],
    ) -> None:
        self._own = own
        self._linked = linked

    def _rows(self) -> dict[str, TriggerRenderer]:
        if not self._linked:
            return self._own
        rows = dict(self._own)
        for context in self._linked:
            rows.update(context.tables.renderers)
        return rows

    def __getitem__(self, key: str) -> TriggerRenderer:
        return self._rows()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows())

    def __len__(self) -> int:
        return len(self._rows())

    def __repr__(self) -> str:
        return f"ChainedRenderers({sorted(self._rows())!r})"


@dataclass(frozen=True, slots=True)
class ContextOwnership:
    """Who holds what, read off the context tree rather than off a ledger.

    The one owner table this design has: a thing belongs to the context whose
    table it is in.  Built for readers that need to state ownership — the
    composition digest, and the tests that check the install ledger's parallel
    account agrees with it.
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


def ownership_of(linked: Sequence[AtomContext]) -> ContextOwnership:
    """Index which linked context holds each tool, policy, renderer, service."""

    tools: dict[int, str] = {}
    policies: dict[int, str] = {}
    renderers: dict[str, str] = {}
    services: dict[str, str] = {}
    for context in linked:
        owner = context.module_path
        for tool in context.tables.tools:
            tools[id(tool)] = owner
        for row in context.tables.policies:
            policies[id(row.policy)] = owner
        for source in context.tables.renderers:
            renderers[source] = owner
        for key in context.services.own_names():
            services[key] = owner
    return ContextOwnership(
        tools=tools,
        policies=policies,
        renderers=renderers,
        services=services,
    )


def unlink_all(session_bus: EventBus, contexts: Sequence[AtomContext]) -> None:
    """Take every context out of a bus, for a session that is closing."""

    for context in contexts:
        session_bus.unlink(context.segment)


__all__ = [
    "AtomContext",
    "AtomResidue",
    "ChainedPolicies",
    "ChainedRenderers",
    "ChainedTools",
    "ContextOwnership",
    "ContextTables",
    "PolicyRow",
    "ownership_of",
    "policy_row",
    "unlink_all",
]
