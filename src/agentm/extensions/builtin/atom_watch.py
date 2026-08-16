# code-health: ignore-file[AM025] -- narrows a host-supplied loader, its two
# return shapes, and the follower a newer version of this atom registered
"""Follow the scenario while the session runs.

Compose this atom and the session's composition tracks its scenario: add an
extension to the scenario and it installs, remove one and it detaches, change
an atom's config or edit its source and it reloads. Compose without it and the
composition is fixed at start.

What is followed is the scenario, not a directory of source files: a scenario
states the order atoms install in, the config each one gets, and by omission
which ones should not be there at all.

There is no file watching here. The scenario is re-resolved on a timer and the
resolved specs are compared, so this follows a scenario wherever the host keeps
one, and an edited atom source shows up anyway: the loader digests the file it
names, so changing that file changes the spec.

An atom that does not install settles rather than being retried on the timer:
the session keeps running the version it already has, and the attempt comes
back when the spec changes -- which is what editing the file does -- or when a
pass is rebuilding that position anyway, since what an atom needs to install
can be supplied by a change somewhere else in the composition. The timer
converges on the scenario; it does not hammer a broken one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, Field

from agentm.core.abi import (
    SCENARIO_LOADER_SERVICE,
    AtomAPI,
    DiagnosticEvent,
    ExtensionManifest,
    ExtensionInput,
    ExtensionSpec,
    ScenarioLoader,
    ScenarioSpec,
    SessionReadyEvent,
    SessionShutdownEvent,
    normalize_extension_spec,
)


_FOLLOWER_SERVICE: Final = "atom_watch_follower"
"""Which follower instance owns the session's polling loop.

A version of this atom that supersedes another registers over this key, so the
loop the previous version is running can see that it is no longer the session's
follower and hand over.
"""

_MAX_UNCONFIRMED_HANDOVERS: Final = 3
"""How many passes in a row may find a successor that has not taken over.

A superseded follower keeps its loop until the successor confirms, because the
reasons a successor declines -- a scenario file caught half written, a loader
not registered yet -- are the ones the next pass recovers from. Those clear
within a pass or two, so three of them is a wide margin.

The bound is what makes keeping the loop safe rather than permanent. Nothing
else can stop this task once another version holds the service: superseding
detaches this atom's handlers, the shutdown handler among them, and shutdown
only emits on the bus. A successor that will never confirm -- a value that is
not a follower at all, a ``take_over`` that raises on every call -- would
otherwise leave this loop installing and detaching atoms against a session
that has already ended.
"""


class AtomWatchConfig(BaseModel):
    """How often to re-read the scenario, and whether omission detaches."""

    interval_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Seconds between scenario reads. The staleness bound.",
    )
    apply_removals: bool = Field(
        default=True,
        description=(
            "Detach atoms the scenario stops listing. Off leaves a removed "
            "atom running, which is the safer default for a session whose "
            "scenario is edited by something other than its developer."
        ),
    )


MANIFEST = ExtensionManifest(
    name="atom_watch",
    description="Track the scenario's composition while the session runs.",
    registers=(
        "event:session_ready",
        "event:session_shutdown",
        f"service:{_FOLLOWER_SERVICE}",
    ),
    config_schema=AtomWatchConfig,
)


@dataclass(slots=True)
class _Settled:
    """The part of a follower's state nothing else records.

    What the session *holds* is not here: ``AtomAPI.installed_atoms`` answers
    that, and reading it is what stops this from becoming a second account of
    the composition.  It used to be one, and it went stale exactly where you
    would expect -- an atom removed by anybody other than this follower left
    the remembered picture naming it, and nothing corrected that until the
    scenario file changed.

    ``failed`` is a version that did not install, keyed by location.  Retrying
    it every tick would report the same breakage twice a second, so it is
    attempted again when its spec changes or when anything else lands -- an
    install fails for reasons outside its own spec.

    ``mine`` is which locations this follower put there.  It scopes removals:
    the scenario dropping an atom is a reason to detach the one *this* follower
    installed, and not a licence to detach whatever else the session happens to
    hold.
    """

    failed: dict[str, ExtensionSpec] = field(default_factory=dict)
    mine: set[str] = field(default_factory=set)

    def copy(self) -> _Settled:
        return _Settled(failed=dict(self.failed), mine=set(self.mine))


class _Handover(Enum):
    """What one pass concluded about a newer version of this atom."""

    NOT_SUPERSEDED = auto()
    """No other version holds the follower service. Keep the loop."""

    UNCONFIRMED = auto()
    """Another version holds the service but is not following yet.

    Keep the loop -- giving it up here would end scenario following for the
    session, since the loop that would retry is the one that returned -- but
    count it: consecutive ones are how a handover that will never land is told
    apart from one that is a pass away.
    """

    RELEASED = auto()
    """This follower is no longer the session's. The loop is over."""


@runtime_checkable
class _Follower(Protocol):
    """What one version of this atom needs from the version replacing it."""

    def take_over(self, settled: _Settled) -> bool:
        """Follow the scenario from here on; report a loop now running.

        ``settled`` is the part of the caller's state nothing else records:
        which versions it tried and could not install, and which atoms it put
        there itself. What the session *holds* is not handed over, because the
        session answers that.

        Reporting False leaves the caller following, because the reasons a
        successor does not start -- a half-written scenario file, a loader not
        yet registered -- are the ones a later pass recovers from.
        """
        ...


class _ScenarioFollower:
    """Re-reads the scenario and moves the session's composition toward it."""

    def __init__(self, api: AtomAPI, config: AtomWatchConfig) -> None:
        self._api = api
        self._interval = config.interval_seconds
        self._apply_removals = config.apply_removals
        self._task: asyncio.Task[None] | None = None
        self._settled = _Settled()

    # --- Lifecycle ---

    def on_session_ready(self, _event: SessionReadyEvent) -> None:
        # The session factory has just installed the plan, so every position
        # holds the version the scenario names.
        self._start(applied=None)

    def take_over(self, settled: _Settled) -> bool:
        """Follow the scenario in place of the version this one replaced.

        ``SessionReadyEvent`` fires once per session, so a version installed
        into a running session cannot start from it. The version being replaced
        starts this one instead, from the loop it is about to leave, and hands
        over what it converged the session to.
        """

        return self._start(applied=settled)

    def _start(self, *, applied: _Settled | None) -> bool:
        """Start the polling loop; report whether one is running afterwards."""

        if self._task is not None:
            return True
        resolved = self._resolve()
        if resolved is None:
            logger.info(
                "atom watch idle: this session has no scenario to follow",
            )
            return False
        # Starting fresh means the factory has just installed the plan, so
        # every position the scenario names is one this follower manages --
        # which is what scopes removals. Taking over from a previous version
        # inherits its answer instead.
        self._settled = (
            _Settled(mine={_key(spec) for spec in resolved})
            if applied is None
            else applied.copy()
        )
        self._task = asyncio.create_task(self._loop(), name="agentm-atom-watch")
        logger.info(
            "following scenario {} every {}s ({} extensions)",
            self._api.ctx.scenario,
            self._interval,
            len(resolved),
        )
        return True

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # --- Scenario ---

    def _resolve(self) -> list[ExtensionSpec] | None:
        scenario = self._api.ctx.scenario
        if not scenario:
            return None
        loader = self._api.services.get(SCENARIO_LOADER_SERVICE)
        if not isinstance(loader, ScenarioLoader):
            logger.warning(
                "atom watch has no scenario loader; composition cannot be "
                "followed for scenario {}",
                scenario,
            )
            return None
        try:
            result = loader(scenario)
        except Exception as exc:  # noqa: BLE001 - a bad edit must not kill the loop
            logger.warning("scenario {} did not resolve: {}", scenario, exc)
            return None
        if isinstance(result, ScenarioSpec):
            inputs: Sequence[ExtensionInput] = result.extensions
        else:
            inputs = result
        return [normalize_extension_spec(item) for item in inputs]

    # --- Loop ---

    async def _loop(self) -> None:
        unconfirmed = 0
        while True:
            await asyncio.sleep(self._interval)
            # Handing over runs the replacement's code, which the host just
            # loaded from a file someone is editing, so it gets a guard of its
            # own. Sharing the pass's guard would cost the session its tick
            # every time the handover raised, and a successor raises for
            # reasons that do not change by themselves -- a signature this
            # version does not call the way that one declares it -- so the
            # session would keep its loop and converge on nothing.
            try:
                handover = self._hand_over_if_superseded()
            except Exception as exc:  # noqa: BLE001 - a dev loop must not die
                logger.warning("atom watch could not hand the loop over: {}", exc)
                handover = _Handover.UNCONFIRMED
            if handover is _Handover.RELEASED:
                return
            if handover is _Handover.NOT_SUPERSEDED:
                unconfirmed = 0
            else:
                unconfirmed += 1
                # A successor declines for exactly one reason: its own
                # _resolve() returned None. When ours returns None too, the
                # scenario is momentarily unreadable for both of us -- a
                # half-written file, a loader not yet registered -- and the
                # bound must not fire on it. The rest of this module already
                # treats an unreadable scenario as a no-op rather than a fatal
                # event, and a loop retained in that state installs and
                # detaches nothing, so the orphan the bound guards against is
                # absent. What the bound is for is a successor structurally
                # unable to take over, and that one still stops us in k passes.
                # Resolved here rather than above so a readable scenario costs
                # no second load: _tick() below resolves it again anyway.
                if unconfirmed >= _MAX_UNCONFIRMED_HANDOVERS:
                    if self._resolve() is None:
                        unconfirmed -= 1
                        logger.debug(
                            "atom watch keeping the loop: the scenario is not "
                            "readable, so its replacement could not take it "
                            "over either"
                        )
                    else:
                        logger.error(
                            "atom watch stopping: superseded, and its "
                            "replacement has not taken the scenario loop over "
                            "in {} passes. This session no longer follows its "
                            "scenario.",
                            unconfirmed,
                        )
                        return
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dev loop must not die
                logger.warning("atom watch pass failed: {}", exc)

    def _hand_over_if_superseded(self) -> _Handover:
        """Offer the loop to a newer version of this atom; report the outcome.

        Superseding an atom detaches its handlers, including the shutdown
        handler that cancels this task, so a follower that reloaded itself has
        to notice on its own that the session moved on. The version installed
        over this one registered itself as the session's follower and has no
        session-ready event left to start from, so it is started here.

        The loop is only given up once the successor confirms it is following.
        A successor that could not start leaves this one running: the session
        would otherwise stop tracking its scenario for good, since the loop
        that would have retried is the one that just returned. Retention is the
        caller's to bound -- see ``_MAX_UNCONFIRMED_HANDOVERS``, since one that
        will never confirm has no other way to stop this task.
        """

        follower = self._api.services.get(_FOLLOWER_SERVICE)
        if follower is self:
            return _Handover.NOT_SUPERSEDED
        if follower is None:
            # Nothing registered over this atom; it was detached outright.
            # A supersede leaves the key absent too, but only between
            # remove_atom_registrations and load_extension, which
            # install_extension runs with no await between them -- so this
            # task cannot observe that window. See the note on
            # SessionRuntime.remove_atom_registrations.
            logger.info(
                "atom watch stopping: this session no longer follows a scenario",
            )
            return _Handover.RELEASED
        if not isinstance(follower, _Follower):
            logger.warning(
                "atom watch keeping the scenario loop: {} took the follower "
                "service over but cannot follow",
                type(follower).__name__,
            )
            return _Handover.UNCONFIRMED
        if not follower.take_over(self._settled):
            logger.warning(
                "atom watch keeping the scenario loop: its replacement has "
                "not started following",
            )
            return _Handover.UNCONFIRMED
        logger.info("atom watch handed the scenario loop to its replacement")
        return _Handover.RELEASED

    async def _tick(self) -> None:
        resolved = self._resolve()
        if resolved is None:
            return
        settled = self._settled
        # What the session holds, asked of the session. This used to be
        # remembered, which meant an atom anybody else removed stayed in the
        # picture and was never reinstalled: the follower believed it was
        # there. Reading it makes a missing atom just another position whose
        # spec disagrees.
        running = {_key(spec): spec for spec in self._api.installed_atoms()}

        # What this pass does is decided per position, from that position's own
        # spec. It used to be decided from the first disagreement onward:
        # everything after it was reinstalled too, even an atom whose own spec
        # had not changed, because bus handlers dispatched in subscription
        # order and leaving the tail alone composed differently from a cold
        # start. That is no longer true -- dispatch orders by (priority, rank,
        # seq), layers fold by rank, and a replacement takes the position it
        # superseded -- so a reload no longer costs every atom after it its
        # in-memory state. Order that genuinely matters is declared with
        # ``after``, which does not depend on where either atom sits in a file.
        pending: list[tuple[str, ExtensionSpec, ExtensionSpec | None, str]] = []
        for spec in resolved:
            key = _key(spec)
            here = running.get(key)
            if here == spec:
                state = "settled"
            elif settled.failed.get(key) == spec:
                # This exact version already failed here. The attempt comes
                # back when the spec does -- for a file that is its digest, so
                # an edit lifts it -- or when something else lands, since an
                # install also fails for reasons outside its own spec.
                state = "blocked"
            else:
                state = "reload" if here is not None else "install"
            pending.append((key, spec, here, state))

        landing = any(state in {"install", "reload"} for _k, _s, _h, state in pending)

        for key, spec, here, state in pending:
            if state == "settled":
                # The scenario came back to what runs here, so any version that
                # failed against this position is moot.
                settled.failed.pop(key, None)
                settled.mine.add(key)
                continue
            if state == "blocked" and not landing:
                continue
            if await self._apply(spec, reloading=here is not None):
                settled.failed.pop(key, None)
                settled.mine.add(key)
            else:
                settled.failed[key] = spec

        if self._apply_removals:
            listed = {_key(spec) for spec in resolved}
            for key in sorted(settled.mine - listed):
                held = running.get(key)
                if held is not None:
                    self._detach(held)
                settled.mine.discard(key)
                settled.failed.pop(key, None)

    async def _apply(self, spec: ExtensionSpec, *, reloading: bool) -> bool:
        # One attempt per version, so what is reported below is reported once
        # per broken version rather than once a tick: _tick calls this again
        # only after the spec changes.
        #
        # The bracket is held only across the install: it clears the session's
        # idle flag, so holding it for the whole watch would mean a session
        # that never reports idle.
        with self._api.track_background():
            try:
                await self._api.install_extension(
                    spec,
                    trigger="scenario_watch",
                    replace=True,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                where = spec.source.location
                verb = "reload" if reloading else "install"
                logger.warning("scenario atom {} did not {}: {}", where, verb, exc)
                await self._diagnose("warning", f"{where} failed to {verb}: {exc}")
                return False
        verb = "reloaded" if reloading else "installed"
        logger.info("{} scenario atom {}", verb, spec.source.location)
        await self._diagnose(
            "info",
            f"{spec.source.location} {verb} from the scenario; "
            "its tools apply from the next turn",
        )
        return True

    def _detach(self, spec: ExtensionSpec) -> None:
        if self._api.uninstall_extension(spec):
            logger.info(
                "detached atom {}: the scenario stopped listing it",
                spec.source.location,
            )

    # --- Helpers ---

    async def _diagnose(self, level: str, message: str) -> None:
        await self._api.bus.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="warning" if level == "warning" else "info",
                source="atom_watch",
                message=message,
            ),
        )


def _key(spec: ExtensionSpec) -> str:
    """Identity of an atom across edits: what it is, not which version."""

    return f"{spec.source.kind}:{spec.source.location}"


def install(api: AtomAPI, config: AtomWatchConfig) -> None:
    """Follow the scenario from session ready, or from the version replaced."""

    follower = _ScenarioFollower(api, config)
    api.services.register(_FOLLOWER_SERVICE, follower, scope="session")
    api.on(SessionReadyEvent.CHANNEL, follower.on_session_ready)
    api.on(SessionShutdownEvent.CHANNEL, follower.on_session_shutdown)
