"""The scenario follower converges: a broken atom settles instead of churning.

Position matters to a composition, so the follower rebuilds every atom after
the first one that disagrees with the scenario. That is what makes a failing
atom expensive: retrying it on the timer would reinstall the whole tail every
pass and every one of those atoms would lose what it held in memory. These
lock down that a failed apply is attempted once, that the tail is left alone
until the failing source changes, and that a failed atom is still detachable.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator, Sequence

import pytest

from agentm.core.abi import (
    SCENARIO_LOADER_SERVICE,
    EventBus,
    ExtensionInput,
    ExtensionSpec,
    ServiceRegistry,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.extensions.builtin.atom_watch import (
    AtomWatchConfig,
    _ScenarioFollower,
)


def _spec(name: str, version: str = "v1") -> ExtensionSpec:
    digest = hashlib.sha256(f"{name}:{version}".encode()).hexdigest()
    return ExtensionSpec.from_file(f"/atoms/{name}.py", digest=f"sha256:{digest}")


class _FakeLoader:
    """Whatever the scenario currently says, rewritable between passes."""

    def __init__(self, specs: Sequence[ExtensionSpec]) -> None:
        self.specs = list(specs)

    def __call__(self, scenario: str) -> Sequence[ExtensionInput]:
        del scenario
        return list(self.specs)


class _FakeContext:
    scenario = "demo"


class _CountingAPI:
    """Counts what the follower asks the session to do."""

    def __init__(self, loader: _FakeLoader) -> None:
        self.ctx = _FakeContext()
        self.bus = EventBus()
        self.services = ServiceRegistry()
        self.services.register(SCENARIO_LOADER_SERVICE, loader)
        self.installed: list[ExtensionSpec] = []
        self.detached: list[ExtensionSpec] = []
        self.broken: set[str] = set()

    @contextlib.contextmanager
    def track_background(self) -> Iterator[None]:
        yield

    async def install_extension(
        self,
        extension: ExtensionSpec,
        config: None = None,
        *,
        trigger: str = "runtime",
        replace: bool = False,
    ) -> None:
        del config, trigger, replace
        self.installed.append(extension)
        if extension.source.digest in self.broken:
            raise RuntimeError("module body raised")

    def uninstall_extension(self, atom: ExtensionSpec) -> bool:
        self.detached.append(atom)
        return True


async def _following(
    specs: Sequence[ExtensionSpec],
) -> tuple[_CountingAPI, _FakeLoader, _ScenarioFollower]:
    """A follower seeded with ``specs`` as what the session is running."""

    loader = _FakeLoader(specs)
    api = _CountingAPI(loader)
    follower = _ScenarioFollower(api, AtomWatchConfig())  # type: ignore[arg-type]
    follower.on_session_ready(SessionReadyEvent())
    # The passes are driven by hand from here, so the timer goes away.
    await follower.on_session_shutdown(SessionShutdownEvent())
    return api, loader, follower


def _names(specs: Sequence[ExtensionSpec]) -> list[str]:
    return [spec.source.location for spec in specs]


@pytest.mark.asyncio
async def test_unchanged_scenario_installs_nothing() -> None:
    api, _loader, follower = await _following([_spec(f"a{i}") for i in range(3)])

    await follower._tick()
    await follower._tick()

    assert api.installed == []


@pytest.mark.asyncio
async def test_broken_atom_is_attempted_once_and_leaves_the_tail_alone() -> None:
    seeded = [_spec(f"a{i}") for i in range(8)]
    api, loader, follower = await _following(seeded)
    broken = _spec("a3", "v2")
    api.broken.add(str(broken.source.digest))
    loader.specs[3] = broken

    await follower._tick()
    after_first_pass = list(api.installed)
    for _ in range(4):
        await follower._tick()

    # The first pass rebuilds the tail, because position 3 disagreed and every
    # position after it composes differently once it is reinstalled.
    assert _names(after_first_pass) == [f"/atoms/a{i}.py" for i in range(3, 8)]
    # The four passes after it touch nothing: the broken version is attempted
    # once, and positions 4-8 are left holding what they hold.
    assert api.installed == after_first_pass
    assert _names(api.installed).count("/atoms/a3.py") == 1


@pytest.mark.asyncio
async def test_rewriting_the_broken_source_retries_once() -> None:
    seeded = [_spec(f"a{i}") for i in range(8)]
    api, loader, follower = await _following(seeded)
    broken = _spec("a3", "v2")
    api.broken.add(str(broken.source.digest))
    loader.specs[3] = broken
    await follower._tick()
    api.installed.clear()

    loader.specs[3] = _spec("a3", "v3")
    await follower._tick()
    retried = list(api.installed)
    await follower._tick()

    # The edit is what lifts the block: one retry, the tail rebuilt once
    # behind it, and nothing after that.
    assert _names(retried) == [f"/atoms/a{i}.py" for i in range(3, 8)]
    assert api.installed == retried


@pytest.mark.asyncio
async def test_dropping_an_atom_that_failed_still_detaches_it() -> None:
    seeded = [_spec(f"a{i}") for i in range(4)]
    api, loader, follower = await _following(seeded)
    broken = _spec("a2", "v2")
    api.broken.add(str(broken.source.digest))
    loader.specs[2] = broken
    await follower._tick()

    del loader.specs[2]
    await follower._tick()

    # The version still running is the one detached, not the edit that never
    # took.
    assert api.detached == [seeded[2]]
