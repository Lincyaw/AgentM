"""The attribution matrix: who owns a write, on every path that can make one.

An atom is handed its own context — its own tables, linked into the session —
so "who wrote this" is answered by *which table the write landed in*, and never
by a parameter, a flag, or anything ambient.  These run one cell per control
path and check three things each: what the context tree says, what the install
ledger's parallel account says, and whether the session can actually see the
write.

The two accounts have to agree.  The ledger is still maintained (removal no
longer reads it, but a later batch retires it and until then a divergence is a
bug in one of them), so every cell that asserts an owner asserts it twice.

Cells are numbered to match the acceptance list they were written against.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from agentm.core.abi.effects import EffectInverse
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.roles import PROVIDER_RESOLVER_SERVICE, RESOURCE_TXN_SERVICE
from agentm.core.abi.session_api import AtomAPI, ExtensionSpec
from agentm.core.abi.stream import Model
from agentm.core.runtime.composition_digest import composition_digest
from agentm.core.runtime.session_core import SessionRuntime
from agentm.testing import NeverStreams, digest_differences, probe_session

# --- Reading the two accounts -----------------------------------------------


def _tree_owner(session: SessionRuntime, key: str) -> str | None:
    """Who holds a service key, according to the context tree."""

    return session.ownership().service(key)


def _ledger_owner(session: SessionRuntime, key: str) -> str | None:
    """Who holds a service key, according to the install ledger."""

    return session._extensions.capture().service_owners.get(key)


def _owner(session: SessionRuntime, key: str) -> str | None:
    """The owner both accounts agree on; fails the test when they do not."""

    tree = _tree_owner(session, key)
    ledger = _ledger_owner(session, key)
    assert tree == ledger, (
        f"context tree says {key} belongs to {tree!r} and the install ledger "
        f"says {ledger!r}"
    )
    return tree


def _tool_owner(session: SessionRuntime, name: str) -> str | None:
    ownership = session.ownership()
    ledger = session._extensions.capture()
    for tool in session.tools:
        if tool.name == name:
            tree_owner = ownership.tool(tool)
            assert tree_owner == ledger.tool_owners.get(id(tool))
            return tree_owner
    raise AssertionError(f"no tool named {name}")


def _file_atom(root: Path, name: str, source: str) -> ExtensionSpec:
    path = root / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return ExtensionSpec.from_file(str(path), digest=digest)


def _composition_differences(
    before: object,
    after: object,
) -> tuple[str, ...]:
    """Digest fields that differ, ignoring which tasks happen to be pending.

    ``background_tasks`` is residue the digest reports on purpose, but it also
    moves when a task the test is deliberately driving finishes, which is not
    what these cells are about.
    """

    return tuple(
        difference.field
        for difference in digest_differences(before, after)  # type: ignore[arg-type]
        if difference.field != "background_tasks"
    )


def _kept(session: SessionRuntime, key: str = "kept_api") -> AtomAPI:
    """The api object an atom published, so a test can write as it would."""

    api = session.services.get(key)
    assert isinstance(api, AtomAPI)
    return api


# --- Atom sources ------------------------------------------------------------

_MANIFEST = """\
from agentm.core.abi.manifest import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="{name}",
    description="attribution matrix probe",
)
"""

# 1 -- a plain synchronous install body.
_SYNC = (
    _MANIFEST
    + """

from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.core.abi.messages import TextContent


def install(api, config):
    del config
    api.services.register("kept_api", api, scope="session")
    api.services.register("sync_write", "installed", scope="session")
    # A subscription and a renderer as well, so a digest comparison over this
    # atom covers the tables whose *position* matters and not only the keyed
    # ones.
    api.on("attribution.ping", lambda event: None, priority=300)
    api.register_trigger_renderer("{name}_source", lambda trigger: [])

    async def _probe(args):
        del args
        # 6 -- a write made from inside a tool call.
        api.services.register("tool_write", "from-tool", scope="session")
        return ToolResult(content=(TextContent(type="text", text="ok"),))

    api.register_tool(
        FunctionTool(
            name="{name}_probe",
            description="writes through the api that installed it",
            parameters={{"type": "object", "properties": {{}}}},
            fn=_probe,
        )
    )
"""
)

# 2 / 14 -- an install body that awaits, and a task started inside that window.
_ASYNC = (
    _MANIFEST
    + """

import asyncio


async def install(api, config):
    del config
    api.services.register("kept_api", api, scope="session")
    started = asyncio.Event()

    async def _in_window():
        # 14 -- runs while install() is still awaiting.
        api.services.register("window_write", "from-window", scope="session")
        started.set()

    task = asyncio.create_task(_in_window())
    await started.wait()
    await task
    # 2 -- a write after the first await of an async install body.
    api.services.register("async_write", "after-await", scope="session")

    async def _late():
        # An async effect body: queued here, run by the settle the
        # installation performs once install() returns.
        api.services.register("settled_write", "from-settle", scope="session")
        yield lambda: None

    api.effect(_late, provides="settled")
"""
)

# 14b -- a task started during install that emits once install() has returned.
_LATE_EMITTER = (
    _MANIFEST
    + """

import asyncio


async def install(api, config):
    del config
    ready = asyncio.Event()
    done = asyncio.Event()

    async def _after():
        await ready.wait()
        api.bus.emit_sync("attribution.late", object())
        api.services.register("late_emit", "emitted", scope="session")
        done.set()

    asyncio.create_task(_after())
    api.services.register("late_ready", ready, scope="session")
    api.services.register("late_done", done, scope="session")
    await asyncio.sleep(0)
"""
)


# 3 / 4 / 19 -- a background task that outlives the installation.
_SURVIVOR = (
    _MANIFEST
    + """

import asyncio


def install(api, config):
    del config
    api.services.register("kept_api", api, scope="session")
    permission = asyncio.Event()
    done = asyncio.Event()

    async def _loop():
        api.services.register("early_write", "before-detach", scope="session")
        await permission.wait()
        api.services.register("late_write", "after-detach", scope="session")
        done.set()

    # Deliberately untracked: a bare task is the write the platform has no
    # table for, and this cell is about what it can and cannot reach.
    api.services.register("survivor_task", asyncio.create_task(_loop()), scope="session")
    api.services.register("survivor_gate", permission, scope="session")
    api.services.register("survivor_done", done, scope="session")
"""
)

# 5 / 5x -- a bus handler that writes when it fires.
_HANDLER = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.register("kept_api", api, scope="session")

    def _on(event):
        del event
        api.services.register("handler_write", "{name}", scope="session")

    api.on("attribution.ping", _on)
    api.on("api.register", _on)
"""
)

# 5x -- an atom whose install emits on the channel the handler above listens to.
_EMITTER = (
    _MANIFEST
    + """

def install(api, config):
    del config
    # Registering emits ``ApiRegisterEvent`` synchronously, which dispatches
    # into the other atom's handler from inside *this* install(). That handler's
    # write must be attributed to the atom that subscribed it, not to this one.
    api.services.register("emitter_write", "emitter", scope="session")
"""
)

# 7 -- a write from a SessionShutdownEvent handler.
_SHUTDOWN_WRITER = (
    _MANIFEST
    + """

from agentm.core.abi.events import SessionShutdownEvent


def install(api, config):
    del config

    def _on(event):
        del event
        api.services.register("shutdown_write", "from-shutdown", scope="session")

    api.on(SessionShutdownEvent.CHANNEL, _on, priority=100)
"""
)

# 10 / 11 -- an atom trying to name somebody else.
_FORGER = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.register("kept_api", api, scope="session")

    def _on(event):
        del event

    # An explicit owner on the bus surface. EventBus.on takes one, so this is
    # sayable; it must not be honoured.
    api.bus.on("attribution.forge", _on, owner="agentm.victim")
    # Writing a key another atom already owns: allowed, but it lands in this
    # atom's own table, so it is this atom's.
    api.services.register("sync_write", "forged", scope="session")
"""
)

# 12 / 13 -- provider registration.
_PROVIDER = (
    _MANIFEST
    + """

from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.stream import Model


class Stream:
    def __init__(self, label):
        self.label = label

    async def __call__(self, **kwargs):
        raise AssertionError("a probe provider never streams")


def install(api, config):
    del config
    api.register_provider(
        "shared_provider",
        ProviderConfig(
            stream_fn=Stream("{name}"),
            model=Model(
                id="probe-model",
                provider="probe",
                context_window=1000,
                max_output_tokens=100,
            ),
            name="shared_provider",
        ),
        replace=True,
    )
"""
)

# 18 -- register_operations, twice over the same key.
_OPERATIONS = (
    _MANIFEST
    + """

from agentm.core.abi.operations import ExecResult


class Bash:
    def __init__(self, label):
        self.label = label

    async def exec(
        self,
        cmd,
        *,
        cwd,
        timeout=None,
        env=None,
        stdin=None,
        on_data=None,
        signal=None,
        log_path=None,
    ):
        del cmd, cwd, timeout, env, stdin, on_data, signal, log_path
        return ExecResult(
            stdout=b"{name}", stderr=b"", exit_code=0, timed_out=False
        )


def install(api, config):
    del config
    api.register_operations(bash=Bash("{name}"), replace=True)
"""
)

# 16 / 21 -- an atom that writes, starts a task, and then fails.
_FAILING = (
    _MANIFEST
    + """

import asyncio


KEPT = []


def install(api, config):
    del config
    KEPT.append(api)
    api.services.register("doomed_write", "written", scope="session")
    gate = asyncio.Event()

    async def _loop():
        await gate.wait()
        api.services.register("doomed_late", "after-rollback", scope="session")

    task = asyncio.create_task(_loop())
    api.effect(lambda: task.cancel, provides="doomed-task", subject=task)
    raise RuntimeError("this atom refuses to install")
"""
)

# 22 -- a context policy that writes through the context it was bound with.
_POLICY = (
    _MANIFEST
    + """

class Probe:
    def __init__(self):
        self.bound = None

    def bind(self, ctx):
        self.bound = ctx
        ctx.services.register("policy_write", "from-policy", scope="session")

    async def transform(self, messages, turns):
        del turns
        return messages


def install(api, config):
    del config
    api.register_context_policy(Probe())
"""
)

# A second install of one module path: same file, so the same module object and
# the same module-level counter, which is what tells the two incarnations apart.
_TWICE = (
    _MANIFEST
    + """

_INCARNATIONS = []


def install(api, config):
    del config
    _INCARNATIONS.append(1)
    api.services.register(
        "dup_key", f"incarnation-{{len(_INCARNATIONS)}}", scope="session"
    )
    api.on("attribution.dup", lambda event: None)
"""
)

# Two atoms contesting one service key, each keeping its api so the test can
# write through it after both are installed.
_CONTESTED = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.register("{name}_api", api, scope="session")
    api.services.register("contested", "{name}-first", scope="session")
"""
)

# An atom that publishes its api and nothing else.
_KEEPER = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.register("kept_api", api, scope="session")
"""
)

# 22 -- an atom that installs another atom, the way atom_watch and
# tool_authoring do: from its own code after its installation is over.
_NESTED = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.register("outer_api", api, scope="session")
    api.services.register("outer_write", "outer", scope="session")
"""
)


def _atom(root: Path, name: str, template: str) -> ExtensionSpec:
    return _file_atom(root, name, template.format(name=name))


# --- 1, 6, 9: the host writes as nobody; the atom writes as itself -----------


@pytest.mark.asyncio
async def test_cell_1_and_6_and_9_sync_install_tool_call_and_host_writes(
    tmp_path: Path,
) -> None:
    """1 sync install body, 6 tool call, 9 host subscription and host write."""

    spec = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        module = spec.module_path

        # 1
        assert _owner(session, "sync_write") == module
        assert session.services.get("sync_write") == "installed"
        assert _tool_owner(session, "sync_atom_probe") == module

        # 6 -- a tool call writes through the api that installed the tool.
        tool = next(t for t in session.tools if t.name == "sync_atom_probe")
        await tool.fn({})
        assert _owner(session, "tool_write") == module

        # 9 -- the host's own writes belong to nobody and stay behind.
        session.services.register("host_write", "host", scope="session")
        session.on("attribution.host", lambda event: None)
        assert _owner(session, "host_write") is None
        assert composition_digest(session).effects == ()

        assert session.uninstall_extension(spec)
        assert session.services.get("sync_write") is None
        assert session.services.get("tool_write") is None
        assert session.services.get("host_write") == "host"


# --- 2, 14: an async install body, and a task inside its window -------------


@pytest.mark.asyncio
async def test_cell_2_and_14_async_body_and_the_window_inside_it(
    tmp_path: Path,
) -> None:
    """2 write after an await, 14 write from a task while install still runs."""

    spec = _atom(tmp_path, "async_atom", _ASYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        module = spec.module_path

        assert _owner(session, "async_write") == module
        assert _owner(session, "window_write") == module
        # The async effect body ran at the installation's own settle.
        assert _owner(session, "settled_write") == module
        assert all(record.settled for record in composition_digest(session).effects)

        assert session.uninstall_extension(spec)
        for key in ("async_write", "window_write", "settled_write"):
            assert session.services.get(key) is None


# --- 3, 4: a surviving task, before and after the detach ---------------------


@pytest.mark.asyncio
async def test_cell_3_and_4_surviving_task_before_and_after_uninstall(
    tmp_path: Path,
) -> None:
    """3 the task's write lands; 4 the same task's later write reaches nobody."""

    spec = _atom(tmp_path, "survivor_atom", _SURVIVOR)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        module = spec.module_path
        await asyncio.sleep(0)

        # 3
        assert _owner(session, "early_write") == module

        gate = session.services.get("survivor_gate")
        done = session.services.get("survivor_done")
        assert isinstance(gate, asyncio.Event)
        assert isinstance(done, asyncio.Event)

        before = composition_digest(session)
        assert session.uninstall_extension(spec)
        after_detach = composition_digest(session)

        # 4 -- let the task run its second write, with the atom gone.
        gate.set()
        await asyncio.wait_for(done.wait(), timeout=1.0)
        assert session.services.get("late_write") is None
        assert _tree_owner(session, "late_write") is None
        assert _ledger_owner(session, "late_write") is None
        # Nothing the session holds moved because of that write.
        assert _composition_differences(after_detach, composition_digest(session)) == ()
        del before
        assert session.services.get("early_write") is None


@pytest.mark.asyncio
async def test_cell_14b_a_task_of_an_installation_may_emit_once_it_is_over(
    tmp_path: Path,
) -> None:
    """The window a lifecycle flag used to leave open, asserted shut.

    ``activate`` is what opens the emit surface, and it runs with no await
    between it and the atom's own code finishing. A task the atom started is
    otherwise refused something it is entitled to, for as long as the
    installation is still writing bookkeeping.
    """

    spec = _atom(tmp_path, "late_emitter", _LATE_EMITTER)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        ready = session.services.get("late_ready")
        done = session.services.get("late_done")
        assert isinstance(ready, asyncio.Event)
        assert isinstance(done, asyncio.Event)
        ready.set()
        await asyncio.wait_for(done.wait(), timeout=1.0)
        assert _owner(session, "late_emit") == spec.module_path


# --- 5, 5x: a handler firing later, and one firing inside another install ----


@pytest.mark.asyncio
async def test_cell_5_and_5x_bus_handler_attribution(tmp_path: Path) -> None:
    """5 a handler firing later; 5x a handler dispatched inside another install."""

    handler = _atom(tmp_path, "handler_atom", _HANDLER)
    emitter = _atom(tmp_path, "emitter_atom", _EMITTER)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(handler)

        # 5 -- fired long after the installation that subscribed it.
        session.bus.emit_sync("attribution.ping", object())
        assert _owner(session, "handler_write") == handler.module_path

        # 5x -- dispatched from inside another atom's install. The write is the
        # subscribing atom's, not the installing one's.
        await session.install_extension(emitter)
        assert _owner(session, "emitter_write") == emitter.module_path
        assert _owner(session, "handler_write") == handler.module_path

        assert session.uninstall_extension(handler)
        # The subscription went with the context: nothing dispatches now.
        assert session.services.get("handler_write") is None
        session.bus.emit_sync("attribution.ping", object())
        assert session.services.get("handler_write") is None


# --- 7: a write from a shutdown handler -------------------------------------


@pytest.mark.asyncio
async def test_cell_7_write_from_a_session_shutdown_handler(tmp_path: Path) -> None:
    """A teardown handler is still the atom's, and still linked when it fires."""

    spec = _atom(tmp_path, "shutdown_atom", _SHUTDOWN_WRITER)
    session = await probe_session(str(tmp_path)).__aenter__()
    seen: list[str | None] = []
    try:
        await session.install_extension(spec)
        session.bus.on(
            SessionShutdownEvent.CHANNEL,
            lambda event: seen.append(_tree_owner(session, "shutdown_write")),
            priority=900,
        )
    finally:
        await session.shutdown()
    assert seen == [spec.module_path]


# --- 8: the driver's per-turn write must not grow anything ------------------


@pytest.mark.asyncio
async def test_cell_8_a_per_turn_host_write_does_not_grow_the_log(
    tmp_path: Path,
) -> None:
    """The driver rebinds a resource transaction every turn, as the host does.

    Twenty of them must leave the recorded effects exactly where they were: a
    write with no atom behind it records no inverse, because there is no
    context to hold one and nothing that would ever run it.
    """

    spec = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        before = composition_digest(session)
        for _ in range(20):
            session.services.register(RESOURCE_TXN_SERVICE, object(), scope="session")
            session.services.unregister(RESOURCE_TXN_SERVICE)
        after = composition_digest(session)
        assert after.effects == before.effects
        assert len(after.services) == len(before.services)


# --- 10, 11: naming somebody else --------------------------------------------


@pytest.mark.asyncio
async def test_cell_10_and_11_an_atom_cannot_name_another(tmp_path: Path) -> None:
    """10 an explicit owner= is dropped; 11 forging is not a thing to refuse."""

    victim = _atom(tmp_path, "sync_atom", _SYNC)
    forger = _atom(tmp_path, "forger_atom", _FORGER)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(victim)
        await session.install_extension(forger)

        # 10 -- the subscription is filed under the atom that made it.
        owners = {
            subscription.owner
            for subscription in session.bus.subscriptions("attribution.forge")
        }
        assert owners == {forger.module_path}
        assert "agentm.victim" not in owners

        # 11 -- writing a key the other atom already holds shadows it, under
        # the forger's own name. There is no way to write under the victim's.
        assert _owner(session, "sync_write") == forger.module_path
        assert session.services.get("sync_write") == "forged"

        # And the victim's value is still there, uncovered by unlinking.
        assert session.uninstall_extension(forger)
        assert _owner(session, "sync_write") == victim.module_path
        assert session.services.get("sync_write") == "installed"

        # There is no owner-selecting surface on what an atom holds.
        api = _kept(session)
        assert not hasattr(api.services, "for_owner")
        assert not hasattr(api, "for_owner")


# --- 12, 13: a provider, and the restore path with a previous owner ---------


@pytest.mark.asyncio
async def test_cell_12_and_13_provider_forward_and_restore_with_previous_owner(
    tmp_path: Path,
) -> None:
    """12 the forward path; 13 the restore path with two atoms, not a fresh one."""

    first = _atom(tmp_path, "provider_one", _PROVIDER)
    second = _atom(tmp_path, "provider_two", _PROVIDER)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(first)
        # 12
        assert _owner(session, "provider:shared_provider") == first.module_path
        assert composition_digest(session).providers[0].owner == first.module_path

        await session.install_extension(second)
        assert _owner(session, "provider:shared_provider") == second.module_path
        second_config = session.get_provider("shared_provider")
        assert second_config is not None

        # 13 -- the second leaves and the first's registration is uncovered,
        # under the first's name rather than as nobody's.
        assert session.uninstall_extension(second)
        assert _owner(session, "provider:shared_provider") == first.module_path
        restored = session.get_provider("shared_provider")
        assert restored is not None
        assert restored is not second_config
        assert composition_digest(session).providers[0].owner == first.module_path


# --- 18: register_operations restore with a previous owner ------------------


@pytest.mark.asyncio
async def test_cell_18_operations_restore_with_a_previous_owner(
    tmp_path: Path,
) -> None:
    """The second atom's operations shadow the first's and give them back."""

    first = _atom(tmp_path, "ops_one", _OPERATIONS)
    second = _atom(tmp_path, "ops_two", _OPERATIONS)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(first)
        key = "operations:bash"
        assert _owner(session, key) == first.module_path
        await session.install_extension(second)
        assert _owner(session, key) == second.module_path

        assert session.uninstall_extension(second)
        assert _owner(session, key) == first.module_path
        bash = session.services.get(key)
        assert bash is not None
        result = await bash.exec("x", cwd=str(tmp_path))
        assert result.stdout == b"ops_one"


# --- 15: composition replay into a child ------------------------------------


@pytest.mark.asyncio
async def test_cell_15_composition_replay_into_a_spawned_child(
    tmp_path: Path,
) -> None:
    """A child replays the specs and gets its own contexts, not the parent's."""

    spec = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        child = await session.spawn(purpose="probe")
        try:
            assert isinstance(child, SessionRuntime)
            assert _owner(child, "sync_write") == spec.module_path
            assert _kept(child) is not _kept(session)
            # The parent keeps its own; nothing moved between the two trees.
            assert _owner(session, "sync_write") == spec.module_path
        finally:
            await child.shutdown()


# --- 16: a task of a failed installation ------------------------------------


@pytest.mark.asyncio
async def test_cell_16_a_failed_installations_task_writes_to_nobody(
    tmp_path: Path,
) -> None:
    """Its writes are rolled back, and the task it left is cancelled."""

    spec = _atom(tmp_path, "failing_atom", _FAILING)
    async with probe_session(str(tmp_path)) as session:
        before = composition_digest(session)
        with pytest.raises(Exception, match="refuses to install"):
            await session.install_extension(spec)

        # The rollback cancelled the task the effect was recorded for; let the
        # cancellation land before comparing.
        await asyncio.sleep(0)
        assert _composition_differences(before, composition_digest(session)) == ()
        assert session.services.get("doomed_write") is None

        # The module the failed installation left in sys.modules still holds the
        # api it was given. Writing through it after the rollback lands in an
        # unlinked context and reaches nobody.
        kept = sys.modules[spec.module_path].__dict__["KEPT"]
        assert kept
        kept[0].services.register("post_rollback", "x", scope="session")
        assert session.services.get("post_rollback") is None
        assert _composition_differences(before, composition_digest(session)) == ()


# --- 17: writing after the session has shut down ----------------------------


@pytest.mark.asyncio
async def test_cell_17_a_surviving_task_writes_after_shutdown(
    tmp_path: Path,
) -> None:
    """Shutdown unlinks every context; a write afterwards reaches nobody."""

    spec = _atom(tmp_path, "sync_atom", _SYNC)
    session = await probe_session(str(tmp_path)).__aenter__()
    await session.install_extension(spec)
    api = _kept(session)
    await session.shutdown()

    api.services.register("post_shutdown", "x", scope="session")
    assert session.services.get("post_shutdown") is None
    assert session.linked_contexts() == ()


# --- 19: a second incarnation of the same module path -----------------------


@pytest.mark.asyncio
async def test_cell_19_a_second_incarnation_inherits_nothing(
    tmp_path: Path,
) -> None:
    """The first incarnation's surviving task must not write into the second's."""

    spec = _atom(tmp_path, "survivor_atom", _SURVIVOR)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        first_api = _kept(session)
        gate = session.services.get("survivor_gate")
        assert isinstance(gate, asyncio.Event)

        assert session.uninstall_extension(spec)
        await session.install_extension(spec)
        second_api = _kept(session)
        assert second_api is not first_api

        first_api.services.register("stale", "from-first", scope="session")
        assert session.services.get("stale") is None
        assert _tree_owner(session, "stale") is None

        # The second incarnation is intact and is what the session resolves.
        assert _owner(session, "kept_api") == spec.module_path
        assert session.services.get("kept_api") is second_api

        gate.set()
        await asyncio.sleep(0)


# --- 21: a failed replace=True leaves the survivor whole --------------------


@pytest.mark.asyncio
async def test_cell_21_a_failed_replacement_leaves_the_survivor_working(
    tmp_path: Path,
) -> None:
    """Reads, writes, entries and digest all come back — not merely reported so.

    The survivor is not restored by un-revoking anything: it was set aside, not
    undone, so putting it back is putting its tables back and relinking.
    """

    survivor = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(survivor)
        api = _kept(session)
        before = composition_digest(session)

        broken = tmp_path / "sync_atom_v2.py"
        broken.write_text(
            _SYNC.format(name="sync_atom") + "\n\nraise RuntimeError('v2 is broken')\n",
            encoding="utf-8",
        )
        failing = ExtensionSpec.from_file(
            str(broken),
            digest="sha256:" + hashlib.sha256(broken.read_bytes()).hexdigest(),
        )

        with pytest.raises(Exception, match="v2 is broken"):
            await session.install_extension(failing, replace=True)

        after = composition_digest(session)
        assert digest_differences(before, after) == ()

        # Reads through the kept api still resolve.
        assert api.services.get("sync_write") == "installed"
        # Writes through it still land in the session.
        api.services.register("after_failed_replace", "yes", scope="session")
        assert session.services.get("after_failed_replace") == "yes"
        assert _owner(session, "after_failed_replace") == survivor.module_path
        # Its tool is still advertised, in the position it was in.
        assert _tool_owner(session, "sync_atom_probe") == survivor.module_path
        # And it is still detachable, which a revoked capability would not be.
        assert session.uninstall_extension(survivor)
        assert session.services.get("sync_write") is None


# --- 22: control paths the list did not name --------------------------------


@pytest.mark.asyncio
async def test_cell_22_resume_replay_is_a_supersede_of_the_same_spec(
    tmp_path: Path,
) -> None:
    """``session.py`` resumes by installing the same spec with replace=True.

    Structurally a supersede of an atom by itself, at the same module path, so
    the context looked up for the detach must be the one actually linked and
    the replacement must not inherit or collide with it.
    """

    spec = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        first = _kept(session)
        before = composition_digest(session)

        await session.install_extension(spec, trigger="resume", replace=True)
        second = _kept(session)

        assert second is not first
        assert len(session.linked_contexts()) == 1
        assert [tool.name for tool in session.tools] == ["sync_atom_probe"]
        assert _owner(session, "sync_write") == spec.module_path
        after = composition_digest(session)
        assert [entry.name for entry in after.tools] == [
            entry.name for entry in before.tools
        ]

        first.services.register("from_the_old_one", "x", scope="session")
        assert session.services.get("from_the_old_one") is None


@pytest.mark.asyncio
async def test_cell_22_a_policy_writes_through_the_context_that_registered_it(
    tmp_path: Path,
) -> None:
    """``PolicyContext.services`` is the registering atom's, not the session's.

    A policy is bound by the session at start, long after install() returned,
    which is exactly the shape that used to lose attribution. Bound with the
    session's own registry it would outlive the atom that put it there.
    """

    spec = _atom(tmp_path, "policy_atom", _POLICY)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        session.start()
        assert _owner(session, "policy_write") == spec.module_path
        assert session.uninstall_extension(spec)
        assert session.services.get("policy_write") is None


@pytest.mark.asyncio
async def test_cell_22_an_atom_installing_an_atom_gets_its_own_context(
    tmp_path: Path,
) -> None:
    """Nesting is one more context linked into the session, not a sub-tree.

    Deliberate, per the target shape: withdrawal does not cascade, so removing
    the outer atom leaves the inner one installed and readable rather than
    silently taking it away.
    """

    inner = _atom(tmp_path, "sync_atom", _SYNC)
    outer_source = _NESTED.format(name="nested_atom")
    outer = _file_atom(tmp_path, "nested_atom", outer_source)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(outer)
        await _kept(session, "outer_api").install_extension(inner)
        assert _owner(session, "outer_write") == outer.module_path
        assert _owner(session, "sync_write") == inner.module_path
        assert {context.module_path for context in session.linked_contexts()} == {
            outer.module_path,
            inner.module_path,
        }

        assert session.uninstall_extension(outer)
        assert session.services.get("outer_write") is None
        assert _owner(session, "sync_write") == inner.module_path
        assert session.uninstall_extension(inner)


@pytest.mark.asyncio
async def test_cell_22_a_write_during_a_spawn_from_inside_an_install(
    tmp_path: Path,
) -> None:
    """A child built while the parent is mid-install sees the linked contexts.

    The parent's composition snapshot is taken from the ledger's replayable
    specs, so an atom that has not been recorded yet is not replayed into the
    child -- but everything already linked is, with its own contexts there.
    """

    spec = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        api = _kept(session)
        child = await api.spawn(purpose="probe")
        try:
            assert isinstance(child, SessionRuntime)
            assert _owner(child, "sync_write") == spec.module_path
        finally:
            await child.shutdown()


@pytest.mark.asyncio
async def test_an_unlinked_context_reads_the_session_it_left(
    tmp_path: Path,
) -> None:
    """The property the whole model turns on, stated on its own.

    Reading resolves up the chain; unlinking means the session no longer
    aggregates *your* writes, not that you cannot see the session's.
    """

    spec = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        api = _kept(session)
        assert session.uninstall_extension(spec)

        session.services.register("written_after", "host", scope="session")
        assert api.services.get("written_after") == "host"
        assert api.services.get("sync_write") is None


# --- The cardinality, the order, and the reach of a write --------------------


@pytest.mark.asyncio
async def test_one_module_path_gets_one_live_context(tmp_path: Path) -> None:
    """A second live incarnation is refused, so removal can reach every one.

    Everything that answers for an atom is keyed on its module path and holds
    one entry: the ledger's attribution, its replayable spec, ``context_for``.
    Two linked contexts under one path would leave removal unlinking one while
    the ledger dropped the path, and the other linked with nothing to name it.
    """

    spec = _atom(tmp_path, "twice_atom", _TWICE)
    async with probe_session(str(tmp_path)) as session:
        pristine = composition_digest(session)
        await session.install_extension(spec)

        with pytest.raises(ValueError, match="already installed"):
            await session.install_extension(spec)

        # The refused install left nothing: one context, one incarnation.
        assert [context.module_path for context in session.linked_contexts()] == [
            spec.module_path
        ]
        assert session.services.get("dup_key") == "incarnation-1"

        assert session.uninstall_extension(spec) is True
        assert session.uninstall_extension(spec) is False
        # Nothing of the atom is left dispatching or resolving.
        assert session.services.get("dup_key") is None
        assert session.bus.subscriptions("attribution.dup") == []
        assert _composition_differences(pristine, composition_digest(session)) == ()


@pytest.mark.asyncio
async def test_the_service_index_names_the_context_a_key_resolves_out_of(
    tmp_path: Path,
) -> None:
    """Ownership is decided by write order, the way resolution is.

    A key resolves to the highest write order anywhere in the chain, so an atom
    that rebinds a key a later-linked atom took wins it back. An index built by
    walking the contexts in link order would name the other one, and the digest
    would then report one context's value under another's name.
    """

    spec_a = _atom(tmp_path, "contest_a", _CONTESTED)
    spec_b = _atom(tmp_path, "contest_b", _CONTESTED)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec_a)
        await session.install_extension(spec_b)
        assert session.services.get("contested") == "contest_b-first"
        assert _owner(session, "contested") == spec_b.module_path

        api_a = _kept(session, "contest_a_api")
        api_a.services.register("contested", "a-later", scope="session")
        assert session.services.get("contested") == "a-later"
        assert _owner(session, "contested") == spec_a.module_path
        entry = next(
            row
            for row in composition_digest(session).services
            if row.key == "contested"
        )
        assert entry.owner == spec_a.module_path

        # The host writing last owns it as nobody, for the same reason.
        session.services.register("contested", "host-last", scope="session")
        assert session.services.get("contested") == "host-last"
        assert _owner(session, "contested") is None

        # And the loser is uncovered when the winner leaves, under its own name.
        assert session.uninstall_extension(spec_a) is True
        assert session.services.get("contested") == "host-last"
        assert _owner(session, "contested") is None


@pytest.mark.asyncio
async def test_a_departed_atoms_effect_runs_and_is_undone_at_shutdown(
    tmp_path: Path,
) -> None:
    """The one write of a departed atom that is not inert, and what bounds it.

    A table write into an unlinked context reaches nobody because the write is
    the row. An effect body is arbitrary code that runs where it is called, so
    it changes the world whatever the context's link state -- refusing it would
    be a revocation bit read at a write. What departure costs it is the
    audience: the session does not report it, and no uninstall will undo it,
    because the uninstall already happened. The session undoes it at shutdown
    instead, which is as far as anything here can promise.
    """

    spec = _atom(tmp_path, "keeper_atom", _KEEPER)
    world: list[str] = []

    def _write() -> EffectInverse:
        world.append("changed")
        return lambda: world.remove("changed")

    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        api = _kept(session)
        assert session.uninstall_extension(spec) is True

        api.effect(_write, provides="after-departure")
        assert world == ["changed"]
        assert all(
            record.provides != "after-departure"
            for record in composition_digest(session).effects
        )

    assert world == []


class _ResolverThatAnswersOnce:
    """Picks a provider once, then stops, so activation fails after the write.

    The registry checks that a provider is resolvable *before* it writes and
    activates it after, so a resolver that stops answering between the two is
    how a test reaches the rollback at all. The rollback is the subject; this
    is only the way in.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self.calls = 0

    def resolve_provider(self, providers: Mapping[str, ProviderConfig]) -> str | None:
        del providers
        self.calls += 1
        return self._name if self.calls == 1 else None


@pytest.mark.asyncio
async def test_a_failed_provider_registration_leaves_one_account_of_it(
    tmp_path: Path,
) -> None:
    """The undo of a shadowing write is removing it, not copying what it hid.

    The previous registration lives in the previous atom's own table. Writing
    it back into the *new* atom's table restores the value and moves the
    ownership: the key would then resolve out of the atom whose registration
    failed, and the three accounts of who owns it -- the context tree, the
    ledger, the provider index -- would give two different answers.
    """

    spec_a = _atom(tmp_path, "prov_a", _PROVIDER)
    spec_b = _atom(tmp_path, "prov_keeper", _KEEPER)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec_a)
        await session.install_extension(spec_b)
        assert _owner(session, "provider:shared_provider") == spec_a.module_path

        resolver = _ResolverThatAnswersOnce("shared_provider")
        session.services.register(PROVIDER_RESOLVER_SERVICE, resolver, scope="session")
        before = composition_digest(session)

        api_b = _kept(session)
        with pytest.raises(LookupError):
            api_b.register_provider(
                "shared_provider",
                ProviderConfig(
                    stream_fn=NeverStreams(),
                    model=Model(
                        id="probe-model",
                        provider="probe",
                        context_window=1000,
                        max_output_tokens=100,
                    ),
                    name="shared_provider",
                ),
                replace=True,
            )

        assert resolver.calls == 2
        assert _owner(session, "provider:shared_provider") == spec_a.module_path
        assert session._providers.owners()["shared_provider"] == spec_a.module_path
        assert _composition_differences(before, composition_digest(session)) == ()
