"""The attribution matrix: who owns a write, on every path that can make one.

An atom is handed its own context — its own tables, linked into the session —
so "who wrote this" is answered by *which table the write landed in*, and never
by a parameter, a flag, or anything ambient.  These run one cell per control
path and check two things each: what the context tree says, and whether the
session can actually see the write.

There used to be a second account to check this one against.  There is not any
more, which is why the helpers below read the tree and nothing else: a cell
that disagreed with it would have nowhere to disagree from.

Cells are numbered to match the acceptance list they were written against.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import sys
import weakref
from collections.abc import Mapping
from pathlib import Path

import pytest

from agentm.core.abi.effects import EffectInverse
from agentm.core.abi.events import ApiRegisterEvent, SessionShutdownEvent
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.roles import PROVIDER_RESOLVER_SERVICE, RESOURCE_TXN_SERVICE
from agentm.core.abi.messages import TextContent
from agentm.core.abi.session_api import AtomAPI, ExtensionSpec
from agentm.core.abi.stream import Model
from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.core.runtime.composition_digest import composition_digest
from agentm.core.runtime.session_core import SessionRuntime
from agentm.testing import NeverStreams, digest_differences, probe_session

# --- Reading the one account -------------------------------------------------


def _tree_owner(session: SessionRuntime, key: str) -> str | None:
    """Who holds a service key, according to the context tree."""

    return session.ownership().service(key)


_owner = _tree_owner
"""The owner of a service key.  One account, so one reader."""


def _renderer_owner(session: SessionRuntime, source: str) -> str | None:
    """Which context's binding of a trigger source resolves."""

    return session.ownership().renderer(source)


def _tool_owner(session: SessionRuntime, name: str) -> str | None:
    ownership = session.ownership()
    for tool in session.tools:
        if tool.name == name:
            return ownership.tool(tool)
    raise AssertionError(f"no tool named {name}")


async def _ok_tool(args: Mapping[str, object]) -> ToolResult:
    """A tool body for a host tool a test only has to be able to register."""

    del args
    return ToolResult(content=(TextContent(type="text", text="ok"),))


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
    api.services.register("{name}_api", api, scope="session")
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

# A trigger renderer two atoms contest, each keeping its api so the test can
# rebind the source through the atom that bound it first.
_RENDERER = (
    _MANIFEST
    + """

class Renderer:
    def __init__(self, label):
        self.label = label

    def render(self, trigger):
        del trigger
        return []


def install(api, config):
    del config
    api.services.register("{name}_api", api, scope="session")
    api.register_trigger_renderer("shared_src", Renderer("{name}"))
"""
)

# An atom whose install records an effect, so a rollback that puts it back can
# be asked whether the session still considers it departed.
_EFFECTFUL = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.register("kept_api", api, scope="session")
    marks = api.services.get("effect_marks")

    def _write():
        marks.append("{name}")
        return lambda: marks.remove("{name}")

    api.effect(_write, provides="{name}-mark")
"""
)

# A replacement that loads (so the supersede runs) and then refuses.
_REFUSES = (
    _MANIFEST
    + """

def install(api, config):
    del api, config
    raise RuntimeError("v2 broke")
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

# A plain atom with a manifest name, for a test that removes it by that name.
_VICTIM = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.register("{name}_write", "victim", scope="session")
"""
)

# An install body that reaches a third atom through an api another atom
# published, removes it, and then refuses -- one task, no awaits.
_SABOTEUR = (
    _MANIFEST
    + """

def install(api, config):
    del config
    api.services.require("kept_api", object).uninstall_extension("{victim}")
    raise RuntimeError("saboteur refuses to install")
"""
)

# An install body that awaits, so another task runs while it is in flight, and
# then refuses.  The gate lets the test decide when the failure lands.
_SLOW_REFUSES = (
    _MANIFEST
    + """

import asyncio


async def install(api, config):
    del config
    api.services.require("install_gate", asyncio.Event).set()
    await asyncio.sleep(0.05)
    raise RuntimeError("{name} broke")
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


# Every registration path an atom can reach from ``api``, in one install body.
# The probe for the claim the host's own tables rest on: an atom writes into
# its own context, so a table of the session's is a table no installation can
# move.
_EVERY_WRITE = (
    _MANIFEST
    + """

from agentm.core.abi.bus import EventBusObserver
from agentm.core.abi.messages import TextContent
from agentm.core.abi.operations import ExecResult
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.stream import Model
from agentm.core.abi.tool import FunctionTool, ToolResult


class Renderer:
    def render(self, trigger):
        del trigger
        return []


class Codec:
    def serialize(self, trigger):
        del trigger
        return {{}}

    def deserialize(self, data):
        del data
        raise AssertionError("a probe codec never decodes")


class Policy:
    async def transform(self, messages, turns):
        del turns
        return messages


class Bash:
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
        return ExecResult(stdout=b"", stderr=b"", exit_code=0, timed_out=False)


class Stream:
    async def __call__(self, **kwargs):
        raise AssertionError("a probe provider never streams")


class Observer(EventBusObserver):
    def on_emit_start(self, channel, event, dispatch_id):
        del channel, event, dispatch_id


async def _probe(args):
    del args
    return ToolResult(content=(TextContent(type="text", text="ok"),))


def install(api, config):
    del config
    api.services.register("every_write_api", api, scope="session")
    api.register_tool(
        FunctionTool(
            name="{name}_tool",
            description="every registration path an atom has",
            parameters={{"type": "object", "properties": {{}}}},
            fn=_probe,
        )
    )
    api.register_context_policy(Policy(), priority=123)
    api.register_trigger_renderer("{name}_source", Renderer())
    api.register_trigger_codec("{name}_source", Codec())
    api.register_operations(bash=Bash(), replace=True)
    api.register_provider(
        "{name}_provider",
        ProviderConfig(
            stream_fn=Stream(),
            model=Model(
                id="probe-model",
                provider="probe",
                context_window=1000,
                max_output_tokens=100,
            ),
            name="{name}_provider",
        ),
        replace=True,
    )
    api.services.register("{name}_service", "value", scope="session")
    api.on("{name}.channel", lambda event: None)
    api.bus.add_observer(Observer())
    api.effect(lambda: (lambda: None), provides="{name}-effect")
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

    The parent's composition snapshot is taken from the replayable specs of
    the installed set, so an atom that has not been recorded yet is not
    replayed into the child -- but everything already linked is, with its own
    contexts there.
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
    one entry: its replayable spec, ``retire``, ``context_for``.  Two linked
    contexts under one path would leave removal unlinking one while the
    installed set dropped the path, and the other linked with nothing to name
    it.
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
    """The undo of a shadowing write puts the key back where it resolved from.

    Two branches, and the same failure in each: a rollback that decides who
    owns a key by anything other than asking the tree afterwards leaves the two
    accounts of it -- the context tree and the provider index -- giving
    different answers, and a *failed* call having changed what the session
    serves.

    The first branch is a write into a table that held nothing for the key: the
    previous registration lives in the previous atom's own table, so writing it
    back into the new atom's would move the ownership to the atom whose
    registration failed. The second is a write into a table that already held
    the key -- an atom re-registering its own provider. Putting that entry back
    with ``register`` restores the value at a *fresh* write order, so the entry
    comes back newer than it was and wins a key it had lost to a later atom.
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

    # The other branch: the failing registration goes into a table that already
    # holds the key, and the entry it shadows has already lost that key to a
    # later atom.
    spec_x = _atom(tmp_path, "prov_x", _PROVIDER)
    spec_y = _atom(tmp_path, "prov_y", _PROVIDER)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec_x)
        await session.install_extension(spec_y)
        key = "provider:shared_provider"
        assert _owner(session, key) == spec_y.module_path
        assert session._providers.owners()["shared_provider"] == spec_y.module_path
        served = session.get_provider("shared_provider")
        assert served is not None
        assert served.stream_fn.label == "prov_y"
        assert session._providers.stream_fn.label == "prov_y"

        resolver = _ResolverThatAnswersOnce("shared_provider")
        session.services.register(PROVIDER_RESOLVER_SERVICE, resolver, scope="session")
        before = composition_digest(session)

        api_x = _kept(session, "prov_x_api")
        with pytest.raises(LookupError):
            api_x.register_provider(
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

        # A failed call changed nothing: not what resolves, not who owns it in
        # any of the three accounts, not what the session would stream through.
        assert resolver.calls == 2
        restored = session.get_provider("shared_provider")
        assert restored is not None
        assert restored.stream_fn.label == "prov_y"
        assert session._providers.stream_fn.label == "prov_y"
        assert _owner(session, key) == spec_y.module_path
        assert session._providers.owners()["shared_provider"] == spec_y.module_path
        assert _composition_differences(before, composition_digest(session)) == ()

        # And the entry that was put back is still the one it was: uninstalling
        # the winner uncovers x's registration rather than nothing.
        assert session.uninstall_extension(spec_y)
        uncovered = session.get_provider("shared_provider")
        assert uncovered is not None
        assert uncovered.stream_fn.label == "prov_x"
        assert _owner(session, key) == spec_x.module_path


class _LabelledRenderer:
    """A trigger renderer a test can tell from another one."""

    def __init__(self, label: str) -> None:
        self.label = label

    def render(self, trigger: object) -> list[object]:
        del trigger
        return []


@pytest.mark.asyncio
async def test_a_trigger_source_resolves_to_the_last_binding_written(
    tmp_path: Path,
) -> None:
    """A source is a key that resolves, so it resolves by write order.

    Two contexts can bind one source and exactly one renderer is used, which is
    the test a table has to meet before its rows need numbering -- the same one
    ``ChainedTools`` states for why a tool name does not. Resolving by link
    order instead would mean an atom rebinding a source it already holds has
    silently no effect while a later-linked context holds it.
    """

    spec_a = _atom(tmp_path, "rend_a", _RENDERER)
    spec_b = _atom(tmp_path, "rend_b", _RENDERER)
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec_a)
        await session.install_extension(spec_b)
        assert session.trigger_renderers["shared_src"].label == "rend_b"
        assert _renderer_owner(session, "shared_src") == spec_b.module_path

        # The earlier-linked atom rebinds the source it lost. The write is the
        # latest, so it takes the source back, and both accounts say so.
        api_a = _kept(session, "rend_a_api")
        api_a.register_trigger_renderer("shared_src", _LabelledRenderer("rend_a-again"))
        assert session.trigger_renderers["shared_src"].label == "rend_a-again"
        assert _renderer_owner(session, "shared_src") == spec_a.module_path

        # The host writing last owns it as nobody, the way a service does.
        session.register_trigger_renderer("shared_src", _LabelledRenderer("host"))
        assert session.trigger_renderers["shared_src"].label == "host"
        assert _renderer_owner(session, "shared_src") is None

        # And the loser is uncovered under its own name when the winner leaves.
        assert session.uninstall_extension(spec_a)
        assert session.trigger_renderers["shared_src"].label == "host"
        assert _renderer_owner(session, "shared_src") is None


@pytest.mark.asyncio
async def test_a_failed_installations_effect_is_undone_at_shutdown(
    tmp_path: Path,
) -> None:
    """A failed install's context departs, so what it records later comes out.

    ``api.effect`` runs its body where it is called, whatever the context's
    link state, so an atom that kept its api across a failed installation can
    still change the world through it. The rollback already decided that this
    context's residue comes back out; a write made after that decision is under
    the same decision, just late.
    """

    spec = _atom(tmp_path, "failing_atom", _FAILING)
    world: list[str] = []

    def _write() -> EffectInverse:
        world.append("changed")
        return lambda: world.remove("changed")

    async with probe_session(str(tmp_path)) as session:
        with pytest.raises(Exception, match="refuses to install"):
            await session.install_extension(spec)
        await asyncio.sleep(0)

        kept = sys.modules[spec.module_path].__dict__["KEPT"]
        assert kept
        kept[0].effect(_write, provides="after-a-failed-install")
        assert world == ["changed"]
        # Recorded into a log the session does not aggregate, so it is not
        # reported as something the composition holds.
        assert all(
            record.provides != "after-a-failed-install"
            for record in composition_digest(session).effects
        )

    assert world == []


@pytest.mark.asyncio
async def test_a_departed_contexts_recorded_inverse_outlives_the_context(
    tmp_path: Path,
) -> None:
    """Holding the context weakly must not mean holding the inverse weakly.

    A context nobody holds has nobody left to write *through* it, which is why
    the context itself is held weakly. That is a statement about future writes
    and says nothing about one already made: an inverse that has been recorded
    is owed, and letting the collector take it would leave the world changed
    with the undo gone.
    """

    spec = _atom(tmp_path, "keeper_atom", _KEEPER)
    world: list[str] = []

    def _write() -> EffectInverse:
        world.append("changed")
        return lambda: world.remove("changed")

    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(spec)
        api = _kept(session)
        watch = weakref.ref(session.context_for(spec.module_path))
        assert session.uninstall_extension(spec) is True

        api.effect(_write, provides="after-departure")
        assert world == ["changed"]

        # Nobody can write through it again, and nothing holds it: the context
        # itself is gone, and only what it recorded is left.
        del api
        gc.collect()
        assert watch() is None

    assert world == []


@pytest.mark.asyncio
async def test_a_restored_survivor_is_no_longer_held_as_departed(
    tmp_path: Path,
) -> None:
    """A rollback that relinks a context takes back the departure with it.

    A superseded atom is noted as departed when it is unlinked, and put back by
    the restore when its replacement fails to land. If the departure outlived
    the restore, the survivor would be simultaneously linked and pending an
    undo, and whether its effects survived would rest on which of the two
    shutdown loops ran first -- an ordering nothing states and nothing checks.
    """

    marks: list[str] = []
    spec = _atom(tmp_path, "eff_atom", _EFFECTFUL)
    async with probe_session(str(tmp_path)) as session:
        session.services.register("effect_marks", marks, scope="session")
        await session.install_extension(spec)
        assert marks == ["eff_atom"]
        survivor = session.context_for(spec.module_path)
        assert survivor is not None

        # Loads under the same manifest name, so the supersede really runs and
        # the survivor really departs before the replacement refuses.
        failing = _file_atom(tmp_path, "eff_atom_v2", _REFUSES.format(name="eff_atom"))
        with pytest.raises(Exception, match="v2 broke"):
            await session.install_extension(failing, replace=True)

        assert session.context_for(spec.module_path) is survivor
        # Stated as behaviour rather than as bookkeeping: undoing the
        # departures now, in either order relative to anything else, must not
        # reach a context the session is holding.
        assert session._departed.undo() == ()
        assert marks == ["eff_atom"]
        assert session.services.get("kept_api") is not None


@pytest.mark.asyncio
async def test_an_installation_writes_into_no_table_of_the_hosts(
    tmp_path: Path,
) -> None:
    """The claim the host's own tables rest on, run rather than reasoned about.

    An atom is handed its own context, so every registration path it can reach
    lands in that context's tables.  The session's own three -- tools,
    policies, renderers -- are therefore the embedder's alone, which is what
    lets a rollback stop copying them and a child inherit them directly.  This
    drives every path ``AtomAPI`` offers, from install, from a tool call, and
    from an atom installing another atom, and asserts the three tables are the
    objects they were.
    """

    spec = _atom(tmp_path, "every_write", _EVERY_WRITE)
    nested = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        host_tools = list(session._own_tools)
        host_policies = list(session._own_policies)
        host_renderers = dict(session._own_renderers)

        await session.install_extension(spec)
        api = _kept(session, "every_write_api")
        await api.install_extension(nested)
        tool = next(t for t in session.tools if t.name == "every_write_tool")
        await tool.fn({})

        assert session._own_tools == host_tools
        assert session._own_policies == host_policies
        assert session._own_renderers == host_renderers

        # The control: the writes really happened, in the atom's own tables.
        assert _owner(session, "every_write_service") == spec.module_path
        assert _tool_owner(session, "every_write_tool") == spec.module_path
        assert _renderer_owner(session, "every_write_source") == spec.module_path
        assert _owner(session, "sync_write") == nested.module_path


@pytest.mark.asyncio
async def test_a_failed_install_on_a_started_session_leaves_the_installed_set(
    tmp_path: Path,
) -> None:
    """The rollback is an inverse now, so the two shapes of it are asserted.

    An installation records exactly one atom, so undoing it retires exactly
    that record -- and because the record is written last, an installation that
    failed has none to retire. A supersede is the other shape: it takes the
    record of the atom it displaced *out*, so the rollback has to put that one
    back, at the position it held. Nothing reconstructs the installed set from
    a picture of it taken beforehand, which is why the position is carried by
    the inverse rather than recovered.

    Driven on a started session because that is where a runtime install differs
    from a composed one, and where the failure path also writes a diagnostic.
    """

    marks: list[str] = []
    first = _atom(tmp_path, "keeper_atom", _KEEPER)
    middle = _atom(tmp_path, "eff_atom", _EFFECTFUL)
    last = _atom(tmp_path, "sync_atom", _SYNC)
    async with probe_session(str(tmp_path)) as session:
        session.services.register("effect_marks", marks, scope="session")
        await session.install_extension(first)
        await session.install_extension(middle)
        await session.install_extension(last)
        session.start()
        installed = list(session.installed_extensions)
        assert installed == [first.module_path, middle.module_path, last.module_path]
        before = composition_digest(session)

        # An installation that fails before it records anything.
        failing = _atom(tmp_path, "failing_atom", _FAILING)
        with pytest.raises(Exception, match="refuses to install"):
            await session.install_extension(failing)
        await asyncio.sleep(0)
        assert session.installed_extensions == installed
        assert _composition_differences(before, composition_digest(session)) == ()

        # A supersede whose replacement refuses: the displaced record comes
        # back in the middle, not at the end.
        replacement = _file_atom(
            tmp_path, "eff_atom_v2", _REFUSES.format(name="eff_atom")
        )
        with pytest.raises(Exception, match="v2 broke"):
            await session.install_extension(replacement, replace=True)
        await asyncio.sleep(0)
        assert session.installed_extensions == installed
        assert session.installed_atom_module_path("eff_atom") == middle.module_path
        assert _composition_differences(before, composition_digest(session)) == ()

        # And the survivor is still an atom, not merely a row: it detaches.
        assert session.uninstall_extension(middle)
        assert marks == []


# --- The installed set is the link list -------------------------------------


@pytest.mark.asyncio
async def test_an_atom_removed_during_a_failed_install_is_not_left_half_held(
    tmp_path: Path,
) -> None:
    """A third atom removed from inside a failing install stays removed.

    No concurrency and no awaits: ``pub`` publishes its own activated api as a
    service, which this repo's own fixtures do, and ``sab``'s plain synchronous
    ``install()`` calls ``uninstall_extension`` through it and then raises. One
    task, one stack.

    The wedge this reproduces was a rollback that put the link list back from a
    picture taken before the install while the record of which atoms were
    installed was undone per-record: ``vic`` came back linked with nothing left
    naming it, so it could be neither removed nor installed again, and the only
    recovery re-recorded it at the end of the set, corrupting composition order
    for the session and every child of it.

    Either end state is defensible and this asserts the one the inverse
    produces: the removal was a real removal by a party that is not this
    install, so it is not undone. ``vic`` is gone and installable again. The
    composition digest cannot tell the two apart -- the removal suspends the
    context before the failure, so the resurrected one is empty -- which is why
    this asks the session what it holds rather than diffing a digest.
    """

    pub = _atom(tmp_path, "pub", _KEEPER)
    vic = _atom(tmp_path, "vic", _VICTIM)
    sab = _file_atom(tmp_path, "sab", _SABOTEUR.format(name="sab", victim="vic"))
    async with probe_session(str(tmp_path)) as session:
        await session.install_extension(pub)
        await session.install_extension(vic)
        assert session.services.get("vic_write") == "victim"

        with pytest.raises(Exception, match="saboteur refuses to install"):
            await session.install_extension(sab)

        assert session.installed_extensions == [pub.module_path]
        assert session.context_for(vic.module_path) is None
        assert session.installed_atom_module_path("vic") is None
        assert session.services.get("vic_write") is None
        # Gone means installable, and installable means back in the set once.
        await session.install_extension(vic)
        assert session.installed_extensions == [pub.module_path, vic.module_path]
        assert session.services.get("vic_write") == "victim"
        assert session.uninstall_extension(vic)


@pytest.mark.asyncio
async def test_an_atom_detached_by_another_task_is_not_resurrected(
    tmp_path: Path,
) -> None:
    """The concurrent shape of the same rule, and the pre-existing half of it.

    The failing install awaits, and another task detaches an atom the install
    never touched while it is awaiting. A rollback that restored the link list
    from a picture taken before the install put that atom back -- reversing a
    decision that was never this install's to reverse, and relinking a context
    whose effects had already been run backwards.

    True before the installed set was collapsed as well, which is why it is
    stated as its own cell: a detach by a third party survives the rollback,
    and what the rollback undoes is what this install itself did.
    """

    victim = _atom(tmp_path, "vic", _VICTIM)
    failing = _atom(tmp_path, "slow_atom", _SLOW_REFUSES)
    gate = asyncio.Event()
    async with probe_session(str(tmp_path)) as session:
        session.services.register("install_gate", gate, scope="session")
        await session.install_extension(victim)

        async def _detach_once_installing() -> None:
            await gate.wait()
            assert session.uninstall_extension(victim)

        detach = asyncio.create_task(_detach_once_installing())
        with pytest.raises(Exception, match="slow_atom broke"):
            await session.install_extension(failing)
        await detach

        assert session.installed_extensions == []
        assert session.context_for(victim.module_path) is None
        assert session.services.get("vic_write") is None


@pytest.mark.asyncio
async def test_a_failed_supersede_puts_the_atom_back_among_its_neighbours(
    tmp_path: Path,
) -> None:
    """Order under failure is the order a cold start would produce.

    ``a``, ``b``, ``c`` composed; a supersede of ``b`` that awaits and then
    refuses, with ``a`` detached by another task inside that window. The
    survivor has to come back *between its neighbours*, not at the index it
    happened to occupy: the list moved under it, so an integer position would
    put ``b`` after ``c`` and every child spawned afterwards would replay the
    composition in an order no cold start produces.
    """

    first = _atom(tmp_path, "a_atom", _VICTIM)
    middle = _atom(tmp_path, "b_atom", _VICTIM)
    last = _atom(tmp_path, "c_atom", _VICTIM)
    gate = asyncio.Event()
    async with probe_session(str(tmp_path)) as session:
        session.services.register("install_gate", gate, scope="session")
        for spec in (first, middle, last):
            await session.install_extension(spec)
        assert session.installed_extensions == [
            first.module_path,
            middle.module_path,
            last.module_path,
        ]

        async def _detach_once_installing() -> None:
            await gate.wait()
            assert session.uninstall_extension(first)

        # Loads under b_atom's manifest name, so the supersede really runs.
        replacement = _file_atom(
            tmp_path, "b_atom_v2", _SLOW_REFUSES.format(name="b_atom")
        )
        detach = asyncio.create_task(_detach_once_installing())
        with pytest.raises(Exception, match="b_atom broke"):
            await session.install_extension(replacement, replace=True)
        await detach

        assert session.installed_extensions == [middle.module_path, last.module_path]
        assert session.installed_atom_module_path("b_atom") == middle.module_path
        assert [
            spec.module_path for spec in session.composition_snapshot().extensions
        ] == [middle.module_path, last.module_path]
        assert session.uninstall_extension(middle)


@pytest.mark.asyncio
async def test_a_replacement_takes_the_position_it_superseded(
    tmp_path: Path,
) -> None:
    """Reloading an atom must not reorder the composition a child replays.

    The failure path was hardened first, which left the common one wrong: a
    supersede that *lands* used to append, so ``a, b, c`` with ``b`` reloaded
    became ``a, c, b``. Reloading is not the rare case -- it is what
    ``atom_watch`` does every time a scenario file changes -- and the installed
    set's order is what every child replays, so a long-running session drifted
    away from what a cold start of the same atoms produces.

    Stated against a cold start rather than against a literal, because that is
    the property: the two sessions hold the same atoms, so they must compose
    their children the same way.
    """

    first = _atom(tmp_path, "a_atom", _VICTIM)
    middle = _atom(tmp_path, "b_atom", _VICTIM)
    last = _atom(tmp_path, "c_atom", _VICTIM)
    # Same manifest name, different file, so it really supersedes.
    replacement = _file_atom(tmp_path, "b_atom_v2", _VICTIM.format(name="b_atom"))

    async with probe_session(str(tmp_path)) as reloaded:
        for spec in (first, middle, last):
            await reloaded.install_extension(spec)
        await reloaded.install_extension(replacement, replace=True)
        after_reload = [
            spec.module_path for spec in reloaded.composition_snapshot().extensions
        ]

    async with probe_session(str(tmp_path)) as cold:
        for spec in (first, replacement, last):
            await cold.install_extension(spec)
        from_cold = [
            spec.module_path for spec in cold.composition_snapshot().extensions
        ]

    assert after_reload == from_cold
    assert after_reload == [
        first.module_path,
        replacement.module_path,
        last.module_path,
    ]


# An atom that registers a tool -- which emits ``ApiRegisterEvent`` through
# ``emit_sync``, re-entrantly -- and then refuses to install.
_EMITS_THEN_FAILS = (
    _MANIFEST
    + """

from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.core.abi.messages import TextContent


async def _noop(args):
    del args
    return ToolResult(content=(TextContent(type="text", text="ok"),))


def install(api, config):
    del config
    api.register_tool(
        FunctionTool(
            name="{name}_tool",
            description="fires a register event before the refusal",
            parameters={{"type": "object", "properties": {{}}}},
            fn=_noop,
        )
    )
    raise RuntimeError("{name} refuses to install")
"""
)


@pytest.mark.asyncio
async def test_a_sync_register_handler_writes_into_the_host_during_an_install(
    tmp_path: Path,
) -> None:
    """The real edge on "an installation writes into no table of the host's".

    No path an *atom* reaches writes into a host table. But a registration
    emits ``ApiRegisterEvent`` through ``bus.emit_sync``, re-entrantly, from
    inside the atom's own registration call, so an embedder's handler runs on
    the install's stack and can write into the host's tables from there. The
    host's tables are not in the install snapshot, so such a write survives the
    install's failure.

    That is the documented behaviour rather than a defect this branch fixes,
    and it is pinned here because the alternative -- putting the three host
    tables back into the snapshot -- would restore a whole-state picture beside
    an inverse-shaped rollback and rebuild the hybrid this round removes.

    Two halves make it worth writing down rather than leaving to be found.
    One handler gets two fates, decided by which store it reached for: its
    write into a host table survives, and its service registration and its
    subscription do not, because the registry and the bus are still restored
    from a picture. And the same embedder code is atomic, half-atomic or inert
    depending on ``def`` versus ``async def``, because ``emit_sync`` closes an
    async handler's coroutine and logs.
    """

    spec = _atom(tmp_path, "emitter_atom", _EMITS_THEN_FAILS)
    async with probe_session(str(tmp_path)) as session:
        host_tool = FunctionTool(
            name="host_tool",
            description="registered by the embedder from inside an install",
            parameters={"type": "object", "properties": {}},
            fn=_ok_tool,
        )
        fired: list[str] = []
        never: list[str] = []
        heard: list[str] = []

        def _sync_handler(event: object) -> None:
            del event
            if fired:
                return
            fired.append("sync")
            session.register_tool(host_tool)
            session.services.register("host_from_handler", "written mid-install")
            session.on("host.channel", lambda event: heard.append("host"))

        async def _async_handler(event: object) -> None:
            del event
            never.append("async")

        session.on(ApiRegisterEvent.CHANNEL, _sync_handler)
        session.on(ApiRegisterEvent.CHANNEL, _async_handler)

        with pytest.raises(Exception, match="emitter_atom refuses to install"):
            await session.install_extension(spec)

        # It ran on the install's stack, and what it wrote into a host table is
        # still there.
        assert fired == ["sync"]
        assert host_tool in session._own_tools
        assert [tool.name for tool in session.tools] == ["host_tool"]
        # The same handler's other two writes are gone, because the registry
        # and the bus are still restored from a picture. This is the asymmetry
        # the snapshot's docstring names; it is pinned rather than endorsed.
        assert session.services.get("host_from_handler") is None
        session.bus.emit_sync("host.channel", object())
        assert heard == []
        # The atom's own write went with the rollback, which is the control.
        assert session.installed_extensions == []
        # And the async handler never ran at all: emit_sync closed it.
        assert never == []
