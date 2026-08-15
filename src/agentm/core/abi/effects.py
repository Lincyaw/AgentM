"""Effects — a write that hands back the way to undo it.

The platform's other recovery paths work by enumeration: something knows the
kinds of thing an atom can register, and cleans up by walking that list.  A
write outside the list — a background task, an open connection, a monkeypatch —
is invisible to it and survives uninstall.

An effect inverts that.  The caller performs its own write and hands back the
inverse, so the platform needs to know nothing about what was written::

    def install(api, config):
        def _connect():
            client = Client.open(config.url)
            api.services.register("thing", client)
            return client.close          # the inverse

        api.effect(_connect, provides="service:thing")

Teardown is then derived rather than maintained beside the loading code, and
the inverse of a composite follows by composition: a body that calls helpers
which record their own effects has already handed back its inverse and need
not restate one.  ``EffectLog`` is the accumulator that holds them.

One log belongs to one atom context (``core/runtime/atom_context.py``), which
is what removes every question of attribution from this file.  There is no
``owner`` field, no ``owner=`` argument, and no way to record into a log other
than by holding it — an atom holds exactly its own.

Three properties are load-bearing and each shapes the code below.

*Recording is synchronous.*  ``effect()`` returns an ``EffectHandle`` without
awaiting, so a plain ``def install(api, config)`` keeps working.  A sync body
therefore runs to completion inline — past its first yield, not merely up to
it, because a write deferred past the end of ``install()`` would not be visible
to the atom's own code that expects it.  Only an async body has to wait, and it
waits for ``settle()``, which is a plain method on the log it was recorded
into: whoever holds the log can run what is queued in it, and a body left
queued is reported by ``unsettled`` rather than hidden.

*What it holds is what actually ran.*  Inverses are appended as they are
produced, so a body that fails half way leaves exactly the inverses of the part
that succeeded.  That is what lets one accumulator serve uninstall,
install-failure rollback and session shutdown alike, rather than one mechanism
per occasion.

*A write without an inverse is a type error.*  A body that produces no inverse,
records no nested effect, and names no ``retain`` reason raises
``EffectWithoutInverse`` where it was registered.  ``retain`` is the escape
hatch for a write that legitimately has no inverse — a trigger codec a
committed turn still names — and it is deliberately noisy at the call site.

An inverse is synchronous.  Detaching an atom is synchronous the whole way up
to ``AtomAPI.uninstall_extension``, so an inverse that had to be awaited could
not run on the one path that most needs it; an author whose teardown is
genuinely asynchronous cancels a task or schedules a close from a sync inverse,
which is what the session's own shutdown hooks already do.

An inverse is not a compensation.  It restores the state the write changed; it
cannot un-send a message or un-charge an API call.  Something whose undo is an
approximation belongs behind an explicit compensation, not here.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from typing import cast

from loguru import logger

type EffectInverse = Callable[[], None]
"""How to undo one write.  Synchronous — see the module docstring."""

type EffectOutcome = (
    EffectInverse
    | Generator[EffectInverse, None, None]
    | AsyncGenerator[EffectInverse, None]
    | None
)
"""What a body may hand back: one inverse, several, or none."""

type EffectBody = Callable[[], EffectOutcome]
"""A write, invoked by ``EffectLog.effect`` and expected to yield its inverse.

Three shapes, and the difference between them is only how many inverses the
write has and whether producing them needs to await:

* a plain callable returning one inverse, or ``None`` with a ``retain`` reason;
* a generator yielding one inverse per write it makes, run to completion inline;
* an async generator, run by ``settle()`` because it cannot run synchronously.

A body is not a context manager.  Its yields emit inverses rather than suspend
it, so teardown belongs inside the yielded inverse, never after the yield.
"""


class EffectWithoutInverse(TypeError):
    """A write was registered without a way to undo it.

    A ``TypeError`` on purpose: in this design "registered but not revertible"
    is a shape the API should not be able to express, not a discipline authors
    are asked to remember.
    """


@dataclass(frozen=True, slots=True)
class EffectEntry:
    """One recorded write: how to undo it, and what it was.

    ``provides`` and ``subject`` are what let the log answer "what does this
    context hold" without a separate ownership table — the provision label
    reads in a diagnostic, the subject is the object itself for a caller that
    needs to compare identities.  ``inverse`` is ``None`` only for a retained
    write, where ``retain`` carries the reason.
    """

    provides: str
    subject: object
    inverse: EffectInverse | None
    retain: str


class EffectHandle:
    """Receipt for one effect, handed back the moment it was recorded.

    ``settled`` is False only for an async body, which has produced nothing
    yet: everything a sync body writes has already happened by the time its
    caller holds this.
    """

    __slots__ = ("_entries", "_settled", "provides", "retain", "subject")

    def __init__(self, *, provides: str, retain: str, subject: object) -> None:
        self.provides = provides
        self.retain = retain
        self.subject = subject
        self._entries: list[EffectEntry] = []
        self._settled = False

    @property
    def settled(self) -> bool:
        """True once the body has finished producing inverses."""

        return self._settled

    @property
    def entries(self) -> tuple[EffectEntry, ...]:
        """The inverses this body produced, in the order it produced them."""

        return tuple(self._entries)

    def _add(self, entry: EffectEntry) -> None:
        self._entries.append(entry)

    def _settle(self) -> None:
        self._settled = True

    def __repr__(self) -> str:
        state = "settled" if self._settled else "pending"
        return (
            f"EffectHandle(provides={self.provides!r}, {state}, "
            f"inverses={len(self._entries)})"
        )


@dataclass(slots=True)
class _PendingBody:
    """An async body that has not run yet."""

    generator: AsyncGenerator[EffectInverse, None]
    handle: EffectHandle


class EffectLog:
    """The inverses of every write recorded through it, oldest first.

    Reverting runs them last-in-first-out, which is the only order that is
    correct in general: a later write may depend on an earlier one, and an
    inverse that runs while its dependency is already gone is undoing a state
    that no longer exists.

    A log belongs to one context.  It carries no attribution because there is
    nothing to attribute: the only writes in it are the ones made through the
    context that owns it.
    """

    __slots__ = ("_entries", "_pending")

    def __init__(self) -> None:
        self._entries: list[EffectEntry] = []
        self._pending: list[_PendingBody] = []

    # --- Recording ---

    def effect(
        self,
        body: EffectBody,
        *,
        provides: str = "",
        retain: str = "",
        subject: object = None,
    ) -> EffectHandle:
        """Run ``body`` and record the inverses it hands back.

        Returns synchronously.  A sync body has finished by the time this
        returns; an async body has not started and will run at the next
        ``settle()``.

        Raises whatever ``body`` raises, having kept the inverses it produced
        before it failed — a half-executed write is exactly the case a separate
        rollback mechanism exists to handle, and the point of the accumulator
        is that it needs no separate mechanism.
        """

        handle = EffectHandle(provides=provides, retain=retain, subject=subject)
        # Held rather than counted: a mark into either list is only an "since
        # here" if nothing is removed from it, and both lists are emptied by
        # ``revert``, which can run while an async body is awaiting. Holding the
        # objects also makes their addresses safe to compare, since none of them
        # can be freed and have its address handed to something new while this
        # tuple exists.
        held_entries = tuple(self._entries)
        held_pending = tuple(self._pending)
        outcome = body()
        if inspect.iscoroutine(outcome):
            # The most natural way to get this wrong, and file atoms are not
            # type-checked, so the error names the mistake. Closing the
            # coroutine here keeps it to one error: left to the collector it
            # would warn again later, from somewhere with no call site in it.
            outcome.close()
            raise TypeError(
                f"effect {provides or '<unnamed>'} was given a coroutine: an "
                "effect body must be a function returning an inverse, or a "
                "generator yielding one; `async def` is not supported -- use "
                "an async generator"
            )
        if inspect.isasyncgen(outcome):
            # ``isasyncgen`` narrows to the runtime type, which has forgotten
            # what the generator yields; ``_record`` checks each value anyway.
            generator = cast("AsyncGenerator[EffectInverse, None]", outcome)
            self._pending.append(_PendingBody(generator=generator, handle=handle))
            return handle
        if inspect.isgenerator(outcome):
            for inverse in outcome:
                self._record(handle, inverse)
        elif outcome is not None:
            self._record(handle, outcome)
        self._complete(handle, held_entries, held_pending)
        return handle

    def _record(self, handle: EffectHandle, inverse: object) -> None:
        if not callable(inverse):
            raise TypeError(
                f"effect {handle.provides!r} handed back "
                f"{type(inverse).__name__}, which is not callable; an effect "
                "yields the function that undoes it"
            )
        entry = EffectEntry(
            provides=handle.provides,
            subject=handle.subject,
            inverse=cast("EffectInverse", inverse),
            retain=handle.retain,
        )
        self._entries.append(entry)
        handle._add(entry)

    def _complete(
        self,
        handle: EffectHandle,
        held_entries: tuple[EffectEntry, ...],
        held_pending: tuple[_PendingBody, ...],
    ) -> None:
        """Close one body, refusing a write that left no way back.

        ``held_entries`` and ``held_pending`` are what the log held before the
        body ran, as objects rather than as lengths: what the waiver below asks
        is whether anything arrived *while the body ran*, and across an
        ``await`` a length cannot answer that.
        """

        handle._settle()
        if handle.entries:
            return
        if handle.retain:
            # Recorded even though it can never run: "what this context holds"
            # has to include the writes it deliberately leaves behind, or the
            # retention becomes invisible the moment it is granted.
            entry = EffectEntry(
                provides=handle.provides,
                subject=handle.subject,
                inverse=None,
                retain=handle.retain,
            )
            self._entries.append(entry)
            handle._add(entry)
            return
        arrived_entries = {id(entry) for entry in held_entries}
        arrived_pending = {id(pending) for pending in held_pending}
        if any(id(entry) not in arrived_entries for entry in self._entries) or any(
            id(pending) not in arrived_pending for pending in self._pending
        ):
            # The body wrote through helpers that recorded their own effects.
            # The inverse of a composite is the composition of theirs, so
            # asking for one here would be asking the author to restate it.
            #
            # A nested effect with an async body counts even though it has
            # produced no entry yet: it is queued in this same log, its own
            # inverse is checked when it runs, and refusing the outer body for
            # the inner one's lateness would reject a composite whose only work
            # is asynchronous.
            return
        raise EffectWithoutInverse(
            f"effect {handle.provides or '<unnamed>'} was registered without an "
            "inverse; hand back the function that undoes it, or pass retain= "
            "with the reason the write has to stay"
        )

    # --- Settling ---

    async def settle(self) -> None:
        """Run the async bodies queued in this log, oldest first.

        A plain method rather than a capability handed out and taken back: the
        log belongs to one context, so being able to settle it is the same
        thing as holding it, and there is no moment at which holding it stops
        meaning that.  The installation calls this once its ``install()`` has
        returned; an atom that queues a body afterwards can settle it itself.

        Drains until empty, so a body that queues another is run in the same
        settle rather than left for a later one that may never come.
        """

        while self._pending:
            pending = self._pending.pop(0)
            # Held after the pop, so what is "new" below is only what this body
            # queues while it runs -- a composite whose parts are themselves
            # asynchronous is waived by the same rule as a synchronous one.
            held_entries = tuple(self._entries)
            held_pending = tuple(self._pending)
            async for inverse in pending.generator:
                self._record(pending.handle, inverse)
            self._complete(pending.handle, held_entries, held_pending)

    # --- Undoing ---

    def revert(self) -> tuple[BaseException, ...]:
        """Run every inverse, newest first, and report what failed.

        Every inverse runs even when one raises: stopping at the first failure
        would strand the writes older than it, which is the opposite of what a
        teardown is for.  The failures are returned rather than raised so the
        caller can attribute them; dropping the returned tuple is the one way
        to use this that hides an error, and ``revert_or_raise`` exists so
        callers with nothing to add never have to.

        The log is emptied first, failures included: an inverse that already ran
        must not run again on a later revert, and an inverse that records an
        effect of its own is recording it into a log that is no longer being
        walked.
        """

        failures = list(self._close_pending())
        undoing = self._entries
        self._entries = []
        for entry in reversed(undoing):
            if entry.inverse is None:
                continue
            try:
                entry.inverse()
            except BaseException as exc:  # noqa: BLE001 - collected, not hidden
                logger.exception(
                    "inverse of {} failed while reverting",
                    entry.provides or "<unnamed effect>",
                )
                failures.append(exc)
        return tuple(failures)

    def revert_or_raise(self, message: str) -> None:
        """Revert, and surface any failure as a group rather than a first one.

        Reverting is a bulk operation and its failures are siblings; picking
        one to raise would hide the others.
        """

        failures = self.revert()
        if failures:
            raise BaseExceptionGroup(message, failures)

    def dispose(self) -> tuple[BaseException, ...]:
        """Drop the inverses without running them.

        For the one case where running them would be wrong: something else has
        already restored the state they describe, so running them would undo a
        write a second time.  Pending bodies are still closed, because a body
        suspended mid-setup holds resources whether or not its inverses are
        wanted.
        """

        failures = self._close_pending()
        self._entries.clear()
        return failures

    def _close_pending(self) -> tuple[BaseException, ...]:
        """Close every queued body, reporting rather than dropping a failure.

        A queued body has not started -- ``settle`` runs each one to completion
        before taking the next -- so its ``aclose`` finishes in a single step
        and needs no loop.  A body that needs more than that is reported rather
        than left to the collector, which is the whole difference between this
        and letting the log go out of scope.
        """

        failures: list[BaseException] = []
        while self._pending:
            pending = self._pending.pop()
            failure = _close_without_awaiting(pending)
            if failure is not None:
                failures.append(failure)
        return tuple(failures)

    # --- Queries ---

    @property
    def entries(self) -> tuple[EffectEntry, ...]:
        """Every recorded inverse, oldest first."""

        return tuple(self._entries)

    @property
    def unsettled(self) -> tuple[EffectHandle, ...]:
        """The handles of the bodies recorded but not yet run, oldest first.

        A body still here has no other occasion to run than a later settle, so
        this is state a caller checking what the session holds has to see.
        """

        return tuple(pending.handle for pending in self._pending)

    def provisions(self) -> tuple[str, ...]:
        """Provision labels, in the order they were written."""

        return tuple(entry.provides for entry in self._entries)

    def subjects(self) -> tuple[object, ...]:
        """The objects written, in the order they were written."""

        return tuple(entry.subject for entry in self._entries)

    # --- Moving ---

    def take(self) -> EffectLog:
        """Move everything into a fresh log, leaving this one empty.

        What a context's tables do when the context is unlinked: the writes go
        somewhere the caller decides the fate of — reverted for a detach, put
        back for a rollback — while the context itself stays usable and empty.
        """

        moved = EffectLog()
        moved._entries = self._entries
        moved._pending = self._pending
        self._entries = []
        self._pending = []
        return moved

    def give(self, other: EffectLog) -> None:
        """Take everything back from a log ``take`` moved it into.

        The reverse of ``take`` and its only caller's only need: a rollback
        that puts an unlinked context back has to put back what it held, in the
        order it held it, ahead of anything written since.
        """

        self._entries = other._entries + self._entries
        self._pending = other._pending + self._pending
        other._entries = []
        other._pending = []

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"EffectLog(entries={len(self._entries)}, unsettled={len(self._pending)})"
        )


def _close_without_awaiting(pending: _PendingBody) -> BaseException | None:
    """Drive one queued body's ``aclose`` as far as it goes without a loop."""

    name = pending.handle.provides or "<unnamed effect>"
    closing = pending.generator.aclose()
    try:
        closing.send(None)
    except StopIteration:
        return None
    except BaseException as exc:  # noqa: BLE001 - returned, not hidden
        logger.exception("could not close the queued effect {}", name)
        return exc
    logger.warning(
        "the queued effect {} awaits while closing and cannot finish "
        "unwinding in a synchronous teardown",
        name,
    )
    try:
        closing.close()
    except BaseException as exc:  # noqa: BLE001 - returned, not hidden
        logger.exception("could not abandon the close of the queued effect {}", name)
        return exc
    return None


__all__ = [
    "EffectBody",
    "EffectEntry",
    "EffectHandle",
    "EffectInverse",
    "EffectLog",
    "EffectOutcome",
    "EffectWithoutInverse",
]
