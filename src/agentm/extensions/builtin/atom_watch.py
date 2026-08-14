# code-health: ignore-file[AM025] -- narrows a host-supplied loader and its two return shapes
"""Follow the scenario while the session runs.

Compose this atom and the session's composition tracks its scenario: add an
extension to the scenario and it installs, remove one and it detaches, change
an atom's config or edit its source and it reloads. Compose without it and the
composition is fixed at start.

That is the whole of the mode distinction, and it deliberately is not a flag. A
development session differs from a recorded one by which atoms were composed,
the same way a package's dev script differs from its start script.

Watching the scenario rather than a directory of source files is what makes
this a composition change rather than a pile of modules. A scenario states the
order atoms install in, the config each one gets, and — by omission — which
ones should not be there at all. A directory can say none of those things: it
has no order worth honoring, no place to put config, and no way to express a
removal.

There is no file watching here. The scenario is re-resolved on a timer and the
resolved specs are compared, so this follows a scenario wherever the host keeps
one, and an edited atom source shows up for free: the loader digests the file
it names, so changing that file changes the spec.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

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
    registers=("event:session_ready", "event:session_shutdown"),
    config_schema=AtomWatchConfig,
)


@dataclass(frozen=True, slots=True)
class _Composition:
    """Resolved scenario extensions, keyed by which atom rather than version.

    ``location`` identifies the atom across edits: a file keeps its path while
    its digest changes, and a module keeps its dotted name while its config
    does. The spec held against that key carries the version.
    """

    specs: dict[str, ExtensionSpec]

    @classmethod
    def of(cls, resolved: Sequence[ExtensionSpec]) -> _Composition:
        return cls(specs={_key(spec): spec for spec in resolved})


class _ScenarioFollower:
    """Re-reads the scenario and moves the session's composition toward it."""

    def __init__(self, api: AtomAPI, config: AtomWatchConfig) -> None:
        self._api = api
        self._interval = config.interval_seconds
        self._apply_removals = config.apply_removals
        self._task: asyncio.Task[None] | None = None
        self._applied = _Composition(specs={})

    # --- Lifecycle ---

    def on_session_ready(self, _event: SessionReadyEvent) -> None:
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
            len(self._applied.specs),
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
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dev loop must not die
                logger.warning("atom watch pass failed: {}", exc)

    async def _tick(self) -> None:
        resolved = self._resolve()
        if resolved is None:
            return
        current = _Composition.of(resolved)
        applied = self._applied.specs

        # Install and reload in scenario order, so an atom that requires
        # another still arrives after it.
        #
        # From the first position where the session and the scenario disagree,
        # every atom after it is reinstalled too, even one whose own spec did
        # not change. Bus handlers dispatch in the order they subscribed, so
        # reloading only the edited atom would leave it running after atoms the
        # scenario lists later, and a channel where position decides the
        # outcome -- the system prompt is last-writer-wins, the tool list is
        # mapped by each handler in turn -- would compose differently from a
        # cold start of the same scenario. Reinstalling the tail costs those
        # atoms whatever they were holding in memory; a development loop can
        # pay that, and a session that reloads nothing never does.
        applied_order = list(applied)
        rebuilding = False
        landed: list[ExtensionSpec] = []
        for position, spec in enumerate(resolved):
            previous = applied.get(_key(spec))
            if not rebuilding:
                in_place = position < len(applied_order) and applied_order[
                    position
                ] == _key(spec)
                if previous is not None and previous == spec and in_place:
                    landed.append(spec)
                    continue
                rebuilding = True
            if await self._apply(spec, reloading=previous is not None):
                landed.append(spec)
            elif previous is not None:
                # A reload that failed leaves the previous version running, so
                # the applied picture keeps naming it rather than the edit that
                # did not take.
                landed.append(previous)

        if self._apply_removals:
            for key, spec in applied.items():
                if key not in current.specs:
                    self._detach(spec)

        self._applied = _Composition.of(landed)

    async def _apply(self, spec: ExtensionSpec, *, reloading: bool) -> bool:
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
    """Start following the scenario once the session is ready."""

    follower = _ScenarioFollower(api, config)
    api.on(SessionReadyEvent.CHANNEL, follower.on_session_ready)
    api.on(SessionShutdownEvent.CHANNEL, follower.on_session_shutdown)
