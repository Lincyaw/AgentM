"""The scenario follower converges: a broken atom settles instead of churning.

Position matters to a composition, so the follower rebuilds every atom after
the first one that disagrees with the scenario. That is what makes a failing
atom expensive: retrying it on the timer would reinstall the whole tail every
pass and every one of those atoms would lose what it held in memory. These
lock down that a failed apply is attempted once, that the tail is left alone
until the failing source changes, and that a failed atom is still detachable.

Settling is not giving up: an atom fails to install for reasons outside its own
source, so a pass that rebuilds its position attempts it again, and a follower
handing the loop to a newer version of itself hands over what the session is
actually running rather than what the scenario asks for.
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
    _FOLLOWER_SERVICE,
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
        self.raises = False

    def __call__(self, scenario: str) -> Sequence[ExtensionInput]:
        del scenario
        if self.raises:
            raise RuntimeError("scenario file is half written")
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
        # An atom that only installs once another one is there, the way a
        # requirement is only solvable once the atom offering it is composed.
        self.requires: dict[str, str] = {}

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
        required = self.requires.get(extension.source.location)
        satisfied = required is None or required in _names(self.installed)
        self.installed.append(extension)
        if extension.source.digest in self.broken:
            raise RuntimeError("module body raised")
        if not satisfied:
            raise RuntimeError(f"nothing offers {required}")

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


@pytest.mark.asyncio
async def test_a_rebuild_retries_an_atom_the_composition_can_now_satisfy() -> None:
    api, loader, follower = await _following([_spec("a0"), _spec("a1")])
    api.requires["/atoms/x.py"] = "/atoms/dep.py"
    loader.specs.append(_spec("x"))
    await follower._tick()
    assert _names(api.installed) == ["/atoms/x.py"]
    api.installed.clear()

    loader.specs.insert(0, _spec("dep"))
    await follower._tick()

    # x failed for a reason outside its own source, and the scenario just
    # supplied it. The pass rebuilds from position 0, so x is attempted again
    # on the way past rather than staying settled against a cause that is gone.
    assert _names(api.installed) == [
        "/atoms/dep.py",
        "/atoms/a0.py",
        "/atoms/a1.py",
        "/atoms/x.py",
    ]


@pytest.mark.asyncio
async def test_a_replacement_that_cannot_start_leaves_the_loop_following() -> None:
    api, loader, follower = await _following([_spec("a0")])
    successor = _ScenarioFollower(api, AtomWatchConfig())  # type: ignore[arg-type]
    api.services.register(_FOLLOWER_SERVICE, successor)
    loader.raises = True

    # Mid-edit the scenario resolves to nothing, which is a pass to skip and
    # not a reason to stop following: handing the loop over here would end
    # scenario following for the session, since nothing restarts it.
    handed_over = follower._hand_over_if_superseded()
    loader.raises = False
    confirmed = follower._hand_over_if_superseded()
    await successor.on_session_shutdown(SessionShutdownEvent())

    assert handed_over is False
    assert confirmed is True


@pytest.mark.asyncio
async def test_a_replacement_inherits_what_is_running_not_what_is_asked_for() -> None:
    api, loader, follower = await _following([_spec("a0"), _spec("a1")])
    broken = _spec("x")
    api.broken.add(str(broken.source.digest))
    loader.specs.append(broken)
    await follower._tick()

    successor = _ScenarioFollower(api, AtomWatchConfig())  # type: ignore[arg-type]
    assert successor.take_over(follower._applied.entries) is True
    await successor.on_session_shutdown(SessionShutdownEvent())
    del loader.specs[-1]
    await successor._tick()

    # x never installed, so the scenario dropping it detaches nothing. A
    # successor that read the scenario instead would believe x was running.
    assert api.detached == []
