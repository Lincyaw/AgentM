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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _Applied:
    """What the session holds at one position of the followed composition.

    ``running`` is the version actually installed there, which after a reload
    that did not take is still the previous one: a failed install rolls back
    and leaves what was there. It is ``None`` at a position nothing ever
    reached.

    ``failed`` is the version that did not apply, kept so a pass that would
    otherwise leave the composition alone does not attempt it again. Retrying
    it there would set the composition rebuilding from this position on every
    tick, so every healthy atom after it would lose its in-memory state twice a
    second for as long as the source stayed broken. A pass that is rebuilding
    this position anyway does retry it, because an install fails for reasons
    outside the spec that failed.
    """

    running: ExtensionSpec | None
    failed: ExtensionSpec | None = None


@dataclass(frozen=True, slots=True)
class _Composition:
    """Followed scenario positions, keyed by which atom rather than version.

    ``location`` identifies the atom across edits: a file keeps its path while
    its digest changes, and a module keeps its dotted name while its config
    does. It is also the only identity an atom whose module body raises still
    has, since a spec that could not be loaded has no manifest name. The entry
    held against that key carries the versions.
    """

    entries: dict[str, _Applied]

    @classmethod
    def of(cls, resolved: Sequence[ExtensionSpec]) -> _Composition:
        return cls(entries={_key(spec): _Applied(running=spec) for spec in resolved})


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

    def take_over(self, applied: Mapping[str, _Applied]) -> bool:
        """Follow the scenario from here on; report a loop now running.

        ``applied`` is what the caller converged the session to. It cannot be
        derived from the scenario, which says what should be installed rather
        than what is: a position may hold an older version, or nothing at all
        where an install failed.

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
        self._applied = _Composition(entries={})

    # --- Lifecycle ---

    def on_session_ready(self, _event: SessionReadyEvent) -> None:
        # The session factory has just installed the plan, so every position
        # holds the version the scenario names.
        self._start(applied=None)

    def take_over(self, applied: Mapping[str, _Applied]) -> bool:
        """Follow the scenario in place of the version this one replaced.

        ``SessionReadyEvent`` fires once per session, so a version installed
        into a running session cannot start from it. The version being replaced
        starts this one instead, from the loop it is about to leave, and hands
        over what it converged the session to.
        """

        return self._start(applied=applied)

    def _start(self, *, applied: Mapping[str, _Applied] | None) -> bool:
        """Start the polling loop; report whether one is running afterwards."""

        if self._task is not None:
            return True
        resolved = self._resolve()
        if resolved is None:
            logger.info(
                "atom watch idle: this session has no scenario to follow",
            )
            return False
        self._applied = (
            _Composition.of(resolved)
            if applied is None
            else _Composition(entries=dict(applied))
        )
        self._task = asyncio.create_task(self._loop(), name="agentm-atom-watch")
        logger.info(
            "following scenario {} every {}s ({} extensions)",
            self._api.ctx.scenario,
            self._interval,
            len(self._applied.entries),
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
        if not follower.take_over(self._applied.entries):
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
        applied = self._applied.entries

        # Install and reload in scenario order, so an atom that requires
        # another still arrives after it.
        #
        # From the first position where the session and the scenario disagree,
        # every atom after it is reinstalled too, even one whose own spec did
        # not change. Bus handlers dispatch in subscription order, so leaving
        # the tail alone would compose differently from a cold start of the
        # same scenario. Reinstalled atoms lose whatever they held in memory;
        # a session whose scenario did not change reinstalls nothing.
        applied_order = list(applied)
        rebuilding = False
        landed: dict[str, _Applied] = {}
        for position, spec in enumerate(resolved):
            key = _key(spec)
            previous = applied.get(key)
            if not rebuilding:
                # Whether this position is undisturbed is the first thing to
                # settle, because both of the ways to leave it alone below
                # depend on it: an atom the scenario moved is one this pass
                # rebuilds, latched or not.
                in_place = (
                    position < len(applied_order) and applied_order[position] == key
                )
                if in_place and previous is not None and previous.failed == spec:
                    # This exact version already failed here and nothing before
                    # it has moved. The session runs what it ran before, so the
                    # position is as satisfied as it is going to get: the
                    # scenario disagreeing with it forever would rebuild every
                    # atom after it on every pass. The attempt comes back when
                    # the spec does -- for a file that is its digest, so an edit
                    # is what lifts this -- or when a pass is rebuilding this
                    # position anyway, since an install also fails for reasons
                    # outside its own spec: a requirement no atom offers yet, an
                    # environment the composition around it had not set up. Both
                    # keep failures to one report per broken version rather than
                    # one a tick.
                    landed[key] = previous
                    continue
                if in_place and previous is not None and previous.running == spec:
                    # The scenario came back to what is running here, so any
                    # version that failed against this position is moot.
                    landed[key] = _Applied(running=spec)
                    continue
                rebuilding = True
            running = None if previous is None else previous.running
            if await self._apply(spec, reloading=running is not None):
                landed[key] = _Applied(running=spec)
            else:
                # A reload that failed leaves the previous version running, so
                # the applied picture keeps naming it rather than the edit that
                # did not take -- and names the edit as the one not to retry.
                landed[key] = _Applied(running=running, failed=spec)

        if self._apply_removals:
            for key, entry in applied.items():
                if key in landed or entry.running is None:
                    continue
                self._detach(entry.running)

        self._applied = _Composition(entries=landed)

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
