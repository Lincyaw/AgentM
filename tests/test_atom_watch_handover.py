"""The supersede handover, driven on a real superseded context.

``tests/test_atom_watch.py`` drives the follower against a fake api holding a
raw ``ServiceRegistry``, which is why five rounds of review never caught that a
capability revoked at detach time broke the handover: the fake has no notion of
being superseded, so it answered every read.

These drive the *real* ``_ScenarioFollower``, installed by the real loader,
through the real ``AtomAPI`` facade, and then supersede it for real.  Three
things have to keep working on an atom that has just been replaced by itself:

* ``_resolve`` — reads the scenario loader, a host service, up the chain;
* ``_hand_over_if_superseded`` — reads the follower key, misses in its own
  (now emptied) table, resolves up, and finds the successor;
* ``_apply`` → ``_diagnose`` → ``api.bus.emit`` — the reload that superseded it
  reports itself through the api it kept.

The same shape is what ``tool_authoring`` does when an authored tool replaces
an earlier one: install through the kept api and then keep using it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from agentm.core.abi.session_api import (
    AgentSessionConfig,
    AtomAPI,
    ExtensionInput,
    ExtensionSpec,
)
from agentm.core.abi.stream import Model
from agentm.core.runtime.session_core import SessionRuntime
from agentm.extensions.builtin.atom_watch import (
    _FOLLOWER_SERVICE,
    _Handover,
    _ScenarioFollower,
)
from agentm.sdk import AgentSession
from agentm.testing import NeverStreams

_ATOM_WATCH = "agentm.extensions.builtin.atom_watch"
_PROBE_MODEL = Model(
    id="probe-model",
    provider="probe",
    context_window=128_000,
    max_output_tokens=4_096,
)


class _EmptyScenario:
    """A readable scenario that lists nothing.

    Readable matters: an unreadable one makes ``_resolve`` return None, and the
    follower's own bound treats that as "the successor could not have started
    either", which would make a handover look confirmed for the wrong reason.
    """

    def __call__(self, scenario: str) -> Sequence[ExtensionInput]:
        del scenario
        return []


async def _session(tmp_path: str) -> AgentSession:
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=tmp_path,
            scenario="demo",
            scenario_loader=_EmptyScenario(),
            extensions=[ExtensionSpec.from_module(_ATOM_WATCH)],
            stream_fn=NeverStreams(),
            model=_PROBE_MODEL,
        )
    )


def _follower(session: SessionRuntime) -> _ScenarioFollower:
    follower = session.services.get(_FOLLOWER_SERVICE)
    assert isinstance(follower, _ScenarioFollower)
    return follower


async def _stop(follower: _ScenarioFollower) -> None:
    task = follower._task
    follower._task = None
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_superseded_follower_finds_its_successor_on_the_first_pass(
    tmp_path,
) -> None:
    """20 -- RELEASED on the first pass, through the api it kept."""

    session = await _session(str(tmp_path))
    try:
        first = _follower(session)
        # Not superseded yet: it reads its own registration and stays.
        assert first._hand_over_if_superseded() is _Handover.NOT_SUPERSEDED

        await session.install_extension(
            ExtensionSpec.from_module(_ATOM_WATCH),
            trigger="scenario_watch",
            replace=True,
        )
        second = _follower(session)
        assert second is not first

        # The scenario still resolves through the api the old follower kept:
        # the loader is the host's service and reading goes up the chain.
        assert first._resolve() == []

        handover = first._hand_over_if_superseded()
        assert handover is _Handover.RELEASED
        assert second._task is not None
        await _stop(second)
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_a_detached_follower_reads_nothing_and_stops(tmp_path) -> None:
    """The other half of the same read: detached outright, so it stops.

    One model answers both. Superseded, the read resolves up and finds the
    successor; detached, the same read resolves up and finds nothing.
    """

    session = await _session(str(tmp_path))
    try:
        follower = _follower(session)
        assert session.uninstall_extension(ExtensionSpec.from_module(_ATOM_WATCH))
        assert follower._api.services.get(_FOLLOWER_SERVICE) is None
        assert follower._hand_over_if_superseded() is _Handover.RELEASED
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_the_follower_reloads_itself_and_still_reports_it(tmp_path) -> None:
    """20 -- ``_apply`` -> ``_diagnose`` -> ``api.bus.emit`` when the atom is itself.

    The install inside ``_apply`` supersedes the very follower running it, so
    everything after the ``await`` runs on an unlinked context. Emitting has to
    keep working, because a follower that could not report its own reload would
    fail the pass that just succeeded.
    """

    session = await _session(str(tmp_path))
    try:
        first = _follower(session)
        diagnostics: list[object] = []
        session.bus.on("diagnostic", diagnostics.append)

        landed = await first._apply(
            ExtensionSpec.from_module(_ATOM_WATCH),
            reloading=True,
        )
        assert landed is True
        assert diagnostics, "the reload reported nothing through the kept api"

        second = _follower(session)
        assert second is not first
        assert first._hand_over_if_superseded() is _Handover.RELEASED
        await _stop(second)
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_the_follower_detaches_an_atom_through_the_api_it_kept(
    tmp_path,
) -> None:
    """The ``_detach`` path: a superseded follower can still remove an atom.

    ``uninstall_extension`` on a kept api is a lifecycle call, not a write into
    the follower's own tables, so being unlinked must not take it away — the
    pass that hands over is also the pass that applies removals.
    """

    session = await _session(str(tmp_path))
    try:
        first = _follower(session)
        api = first._api
        assert isinstance(api, AtomAPI)
        await session.install_extension(
            ExtensionSpec.from_module(_ATOM_WATCH),
            trigger="scenario_watch",
            replace=True,
        )
        second = _follower(session)
        # The old follower, unlinked, still detaches the atom it was told to.
        first._detach(ExtensionSpec.from_module(_ATOM_WATCH))
        assert session.services.get(_FOLLOWER_SERVICE) is None
        await _stop(second)
    finally:
        await session.shutdown()
