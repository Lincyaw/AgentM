"""The accumulator on its own: what it records, what it refuses, what it undoes.

Ported from ``feat/effect-primitive``'s ``tests/test_effect_primitive.py``,
keeping the tests whose subject survives the move to a per-context log: LIFO
order, a retained write, a failing inverse, an async body, the composite
waiver, and the refusal of a write with no way back.

Dropped with the machinery they were about: everything testing an owner
argument, an ``EffectSettlement`` capability, the window between a settle and
the end of an installation, and detaching one owner's slice out of a shared
log. A log now belongs to one context, so there is no owner to pass, no
capability to hold, and no slice to take.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import pytest

from agentm.core.abi.effects import (
    EffectInverse,
    EffectLog,
    EffectWithoutInverse,
)


def test_inverses_run_newest_first_and_a_retained_write_stays() -> None:
    """LIFO is the only order that is correct in general.

    A later write may depend on an earlier one, so undoing the earlier first
    would act on a state that no longer exists. A retained write has no inverse
    to run and is still recorded, or the residue would be invisible.
    """

    log = EffectLog()
    undone: list[str] = []
    log.effect(lambda: lambda: undone.append("first"), provides="first")
    log.effect(lambda: lambda: undone.append("second"), provides="second")
    log.effect(lambda: None, provides="kept", retain="a committed turn names it")

    assert log.provisions() == ("first", "second", "kept")
    assert log.revert() == ()
    assert undone == ["second", "first"]
    assert len(log) == 0


def test_a_failing_inverse_surfaces_and_the_rest_still_run() -> None:
    """Stopping at the first failure would strand every write older than it."""

    log = EffectLog()
    undone: list[str] = []
    log.effect(lambda: lambda: undone.append("outer"), provides="outer")

    def _explodes() -> EffectInverse:
        def _undo() -> None:
            raise RuntimeError("this inverse does not work")

        return _undo

    log.effect(_explodes, provides="broken")
    failures = log.revert()
    assert len(failures) == 1
    assert undone == ["outer"]

    log.effect(lambda: lambda: undone.append("again"), provides="again")
    log.effect(_explodes, provides="broken")
    with pytest.raises(BaseExceptionGroup):
        log.revert_or_raise("teardown failed")
    assert undone == ["outer", "again"]


def test_a_generator_body_keeps_the_inverses_of_what_it_did_land() -> None:
    """A body that fails half way leaves exactly the part that succeeded."""

    log = EffectLog()
    undone: list[str] = []

    def _two_then_fail() -> Generator[EffectInverse, None, None]:
        yield lambda: undone.append("one")
        yield lambda: undone.append("two")
        raise RuntimeError("half way")

    with pytest.raises(RuntimeError, match="half way"):
        log.effect(_two_then_fail, provides="composite")
    assert len(log) == 2
    log.revert()
    assert undone == ["two", "one"]


@pytest.mark.asyncio
async def test_an_async_body_runs_when_the_log_settles() -> None:
    """Nothing is handed out or taken back: settling is draining your own log."""

    log = EffectLog()
    ran: list[str] = []

    async def _body() -> AsyncGenerator[EffectInverse, None]:
        ran.append("ran")
        yield lambda: ran.append("undone")

    handle = log.effect(_body, provides="async")
    assert not handle.settled
    assert log.unsettled == (handle,)
    assert ran == []

    await log.settle()
    assert handle.settled
    assert log.unsettled == ()
    assert ran == ["ran"]
    log.revert()
    assert ran == ["ran", "undone"]


def test_a_body_that_only_calls_other_effects_needs_no_inverse() -> None:
    """The inverse of a composite is the composition of its parts'."""

    log = EffectLog()
    undone: list[str] = []

    def _composite() -> None:
        log.effect(lambda: lambda: undone.append("inner"), provides="inner")

    log.effect(_composite, provides="outer")
    assert log.provisions() == ("inner",)
    log.revert()
    assert undone == ["inner"]


@pytest.mark.asyncio
async def test_a_composite_whose_part_is_async_needs_no_inverse_either() -> None:
    """A nested async body counts even though it has produced nothing yet."""

    log = EffectLog()
    ran: list[str] = []

    async def _inner() -> AsyncGenerator[EffectInverse, None]:
        ran.append("inner")
        yield lambda: ran.append("inner undone")

    def _composite() -> None:
        log.effect(_inner, provides="inner")

    log.effect(_composite, provides="outer")
    await log.settle()
    assert ran == ["inner"]


def test_a_write_without_an_inverse_is_refused_where_it_was_made() -> None:
    """Registered-but-not-revertible is a shape the API should not express."""

    log = EffectLog()
    with pytest.raises(EffectWithoutInverse, match="without an inverse"):
        log.effect(lambda: None, provides="silent")
    assert len(log) == 0


def test_an_async_def_body_is_refused_by_name() -> None:
    """File atoms are not type-checked, so the error names the mistake."""

    log = EffectLog()

    async def _coroutine_body() -> None:
        return None

    with pytest.raises(TypeError, match="async generator"):
        log.effect(_coroutine_body, provides="wrong")  # type: ignore[arg-type]


def test_a_body_that_hands_back_a_non_callable_is_refused() -> None:
    log = EffectLog()
    with pytest.raises(TypeError, match="not callable"):
        log.effect(lambda: "not a function", provides="wrong")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_disposing_closes_a_queued_body_without_running_its_inverses() -> None:
    """A queued body holds a frame whether or not its inverses are wanted."""

    log = EffectLog()
    undone: list[str] = []
    closed: list[str] = []

    async def _body() -> AsyncGenerator[EffectInverse, None]:
        try:
            yield lambda: undone.append("never")
        finally:
            closed.append("closed")

    log.effect(lambda: lambda: undone.append("kept"), provides="sync")
    log.effect(_body, provides="queued")
    assert log.dispose() == ()
    assert undone == []
    # The body never started, so there is no ``try`` to unwind: closing one is
    # a single step, which is what makes a synchronous teardown able to do it.
    assert closed == []
    assert log.unsettled == ()
    assert len(log) == 0


def test_taking_and_giving_back_moves_the_log_without_running_it() -> None:
    """What unlinking a context does: set aside, not undone.

    The property a failed ``replace=True`` rests on -- there is nothing to
    un-revert, because nothing was reverted.
    """

    log = EffectLog()
    undone: list[str] = []
    log.effect(lambda: lambda: undone.append("held"), provides="held")

    moved = log.take()
    assert len(log) == 0
    assert moved.provisions() == ("held",)
    assert undone == []

    log.effect(lambda: lambda: undone.append("since"), provides="since")
    log.give(moved)
    assert log.provisions() == ("held", "since")
    assert len(moved) == 0
    log.revert()
    assert undone == ["since", "held"]
