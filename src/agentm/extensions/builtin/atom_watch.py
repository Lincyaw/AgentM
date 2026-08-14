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
back when the spec changes -- which is what editing the file does. The timer
converges on the scenario; it does not hammer a broken one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
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


@runtime_checkable
class _Follower(Protocol):
    """What one version of this atom needs from the version replacing it."""

    def take_over(self) -> None:
        """Follow the scenario from here on, in place of the caller."""
        ...


@dataclass(frozen=True, slots=True)
class _Applied:
    """What the session holds at one position of the followed composition.

    ``running`` is the version actually installed there, which after a reload
    that did not take is still the previous one: a failed install rolls back
    and leaves what was there. It is ``None`` at a position nothing ever
    reached.

    ``failed`` is the version that did not apply, kept so the same version is
    not attempted again on every pass. Retrying it would set the composition
    rebuilding from this position on every tick, so every healthy atom after it
    would lose its in-memory state twice a second for as long as the source
    stayed broken.
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
        self._start()

    def take_over(self) -> None:
        """Follow the scenario in place of the version this one replaced.

        ``SessionReadyEvent`` fires once per session, so a version installed
        into a running session cannot start from it. The version being replaced
        starts this one instead, from the loop it is about to leave.
        """

        self._start()

    def _start(self) -> None:
        if self._task is not None:
            return
        resolved = self._resolve()
        if resolved is None:
            logger.info(
                "atom watch idle: this session has no scenario to follow",
            )
            return
        self._applied = _Composition.of(resolved)
        self._task = asyncio.create_task(self._loop(), name="agentm-atom-watch")
        logger.info(
            "following scenario {} every {}s ({} extensions)",
            self._api.ctx.scenario,
            self._interval,
            len(self._applied.entries),
        )

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
        while True:
            await asyncio.sleep(self._interval)
            if self._hand_over_if_superseded():
                return
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dev loop must not die
                logger.warning("atom watch pass failed: {}", exc)

    def _hand_over_if_superseded(self) -> bool:
        """Give the loop to a newer version of this atom; report having done so.

        Superseding an atom detaches its handlers, including the shutdown
        handler that cancels this task, so a follower that reloaded itself has
        to notice on its own that the session moved on. The version installed
        over this one registered itself as the session's follower and has no
        session-ready event left to start from, so it is started here.
        """

        follower = self._api.services.get(_FOLLOWER_SERVICE)
        if follower is self:
            return False
        if isinstance(follower, _Follower):
            logger.info("atom watch handing the scenario loop to its replacement")
            follower.take_over()
        else:
            logger.info(
                "atom watch stopping: this session no longer follows a scenario",
            )
        return True

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
            if previous is not None and previous.failed == spec:
                # This exact version already failed here. The session runs what
                # it ran before, so the position is as satisfied as it is going
                # to get: the scenario disagreeing with it forever would rebuild
                # every atom after it on every pass. The attempt comes back when
                # the spec does -- for a file that is its digest, so an edit is
                # what lifts this. Failures are reported from _apply, which this
                # keeps to one report per broken version rather than one a tick.
                landed[key] = previous
                continue
            if not rebuilding:
                in_place = (
                    position < len(applied_order) and applied_order[position] == key
                )
                if previous is not None and previous.running == spec and in_place:
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
