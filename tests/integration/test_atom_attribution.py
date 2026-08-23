"""Uninstall removes what the atom put there, and nothing else does.

An atom's registrations live in that atom's own context, and uninstalling
unlinks it. Anything that landed somewhere else survives uninstall forever, so
each test here installs one kind of registration and asserts it is gone again.
The rest guard the other direction: what an uninstall must not leave behind in
the session's own state, which capabilities each dependency solver is fed, and
which writes reach the event stream.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agentm import AgentSession, AgentSessionConfig, ExtensionSpec, Model
from agentm.core.abi.events import ApiRegisterEvent, TurnCommittedEvent
from agentm.core.abi.messages import TextContent
from agentm.core.abi.provider import ProviderSessionIdentity
from agentm.core.abi.roles import PROVIDER_SESSION_IDENTITY, RESOURCE_TXN_SERVICE
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, TurnMeta
from agentm.core.abi.trigger import UserInput
from agentm.core.runtime.session_core import SessionRuntime
from agentm.core.runtime.session_factory import _service_capabilities


_SYSTEM_PROMPT = "agentm.extensions.builtin.system_prompt"


def _model() -> Model:
    return Model(
        id="stub-model",
        provider="stub",
        context_window=128_000,
        max_output_tokens=4_096,
    )


class _StubProvider:
    """Never streams — these tests exercise composition, not turns."""

    async def __call__(
        self,
        *,
        messages: object,
        model: object,
        tools: object,
        system: object = None,
        signal: object = None,
        thinking: str = "off",
    ) -> object:
        raise AssertionError("attribution tests do not run turns")


def _file_atom(root: Path, name: str, source: str) -> ExtensionSpec:
    """Write one file atom and address it the way the loader does."""

    path = root / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return ExtensionSpec.from_file(str(path), digest=digest)


async def _session(tmp_path: Path, *extensions: ExtensionSpec) -> AgentSession:
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=list(extensions),
            stream_fn=_StubProvider(),
            model=_model(),
        )
    )


_PLAIN_SERVICE_ATOM = """\
from agentm.core.abi.manifest import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="plain_service_atom",
    description="Registers a service without binding it to a role.",
    registers=("service:attribution_probe",),
)


class Probe:
    label = "probe"


def install(api, config):
    del config
    api.services.register("attribution_probe", Probe(), scope="session")
"""


_OPERATIONS_ATOM = """\
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.abi.operations import ExecResult


MANIFEST = ExtensionManifest(
    name="operations_atom",
    description="Registers bash operations.",
    registers=("operations:bash",),
)


class StubBash:
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
        return ExecResult(stdout="", stderr="", exit_code=0)


def install(api, config):
    del config
    api.register_operations(bash=StubBash())
"""


_PROVIDER_ATOM = """\
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.stream import Model


MANIFEST = ExtensionManifest(
    name="provider_atom",
    description="Registers one named LLM provider.",
    registers=("provider:probe_provider",),
)


class ProbeStream:
    async def __call__(
        self,
        *,
        messages,
        model,
        tools,
        system=None,
        signal=None,
        thinking="off",
    ):
        raise AssertionError("probe provider never streams")


def install(api, config):
    del config
    api.register_provider(
        "probe_provider",
        ProviderConfig(
            stream_fn=ProbeStream(),
            model=Model(
                id="probe-model",
                provider="probe",
                context_window=1000,
                max_output_tokens=100,
            ),
            name="probe_provider",
        ),
    )
"""


_OBSERVER_ATOM = """\
from agentm.core.abi.bus import EventBusObserver
from agentm.core.abi.manifest import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="observer_atom",
    description="Attaches a bus observer and publishes what it saw.",
    registers=("service:observer_sink",),
)


class SinkObserver(EventBusObserver):
    def __init__(self, sink):
        self._sink = sink

    def on_emit_start(self, channel, event, dispatch_id):
        del event, dispatch_id
        self._sink.append(channel)


def install(api, config):
    del config
    sink = []
    api.services.register("observer_sink", sink, scope="session")
    api.bus.add_observer(SinkObserver(sink))
"""


_REQUIRES_ATOM_BY_NAME = """\
from agentm.core.abi.manifest import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="needs_system_prompt",
    description="Depends on another atom by its manifest name.",
    requires=("atom:system_prompt",),
)


def install(api, config):
    del api, config
"""


@pytest.mark.asyncio
async def test_uninstall_removes_a_plainly_registered_service(tmp_path: Path) -> None:
    spec = _file_atom(tmp_path, "plain_service_atom", _PLAIN_SERVICE_ATOM)
    session = await _session(tmp_path, spec)
    try:
        assert "attribution_probe" in session.services.names()
        assert session.uninstall_extension(spec)
        assert "attribution_probe" not in session.services.names()
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_uninstall_removes_registered_operations(tmp_path: Path) -> None:
    spec = _file_atom(tmp_path, "operations_atom", _OPERATIONS_ATOM)
    session = await _session(tmp_path, spec)
    try:
        assert "operations:bash" in session.services.names()
        assert session.uninstall_extension(spec)
        assert "operations:bash" not in session.services.names()
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_uninstall_removes_a_registered_provider(tmp_path: Path) -> None:
    spec = _file_atom(tmp_path, "provider_atom", _PROVIDER_ATOM)
    session = await _session(tmp_path, spec)
    try:
        assert "provider:probe_provider" in session.services.names()
        assert "probe_provider" in session._providers.configs()
        assert session.uninstall_extension(spec)
        assert "provider:probe_provider" not in session.services.names()
        assert "probe_provider" not in session._providers.configs()
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_uninstall_detaches_a_bus_observer(tmp_path: Path) -> None:
    spec = _file_atom(tmp_path, "observer_atom", _OBSERVER_ATOM)
    session = await _session(tmp_path, spec)
    try:
        sink: list[str] = session.services.require("observer_sink", list)
        session.bus.emit_sync("attribution.probe", object())
        assert "attribution.probe" in sink
        seen_before = len(sink)

        assert session.uninstall_extension(spec)
        session.bus.emit_sync("attribution.probe", object())
        assert len(sink) == seen_before
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_runtime_install_solves_atom_requirements_by_manifest_name(
    tmp_path: Path,
) -> None:
    """A late install spells ``atom:`` the way the manifest that needs it does.

    ``system_prompt`` loads under a dotted module path and calls itself
    ``system_prompt``; a requirement names the latter. Solving the requirement
    against module paths could never match, so every late install of an atom
    with an ``atom:`` dependency failed — the resume path and every atom_watch
    reload included.
    """

    spec = _file_atom(tmp_path, "needs_system_prompt", _REQUIRES_ATOM_BY_NAME)
    session = await _session(tmp_path, ExtensionSpec.from_module(_SYSTEM_PROMPT))
    try:
        session.start()
        await session.install_extension(spec)
        assert spec.module_path in session.installed_extensions
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_uninstalling_the_only_provider_freezes_no_name_for_it(
    tmp_path: Path,
) -> None:
    """A departed provider must not be what a later commit binds the session to.

    Freezing runs from the turn-committed bus hook, not only from
    ``activate()``, so it sees whatever the uninstall left behind. An identity
    naming a provider registered nowhere would be durable: written into the
    trajectory, validated on resume, and enough to make every later
    ``activate()`` raise once any other provider registers.
    """

    spec = _file_atom(tmp_path, "provider_atom", _PROVIDER_ATOM)
    session = await _session(tmp_path, spec)
    try:
        session.start()
        session.trajectory.begin(
            UserInput(content=(TextContent(type="text", text="hi"),)),
            run_id="attribution",
            run_step=0,
        )
        turn = session.trajectory.commit(
            Outcome(cause=ModelEndTurn()),
            TurnMeta(model_id="probe-model"),
        )

        assert session.uninstall_extension(spec)
        session.bus.emit_sync(
            TurnCommittedEvent.CHANNEL,
            TurnCommittedEvent(turn=turn),
        )

        identity = session.provider_session_identity()
        assert identity is not None
        assert identity.name not in session.provider_names()
        assert identity.name == "direct"
        assert identity.model_id == "probe-model"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_the_runtime_solver_counts_capabilities_the_host_provides(
    tmp_path: Path,
) -> None:
    """A late install solves against everything present, host tools included.

    The cold solver below is narrower on purpose. Locking both input sets is
    what keeps that a contract rather than a divergence nobody chose.
    """

    async def probe(args: dict[str, object]) -> ToolResult:
        del args
        return ToolResult(content=(TextContent(type="text", text=""),))

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[ExtensionSpec.from_module(_SYSTEM_PROMPT)],
            extra_tools=[
                FunctionTool(
                    name="host_probe",
                    description="A tool the embedder provides, not an atom.",
                    parameters={"type": "object", "properties": {}},
                    fn=probe,
                )
            ],
            stream_fn=_StubProvider(),
            model=_model(),
        )
    )
    try:
        assert "tool:host_probe" in session._live_capability_keys()
        assert "atom:system_prompt" in session._live_capability_keys()
    finally:
        await session.shutdown()


def test_the_cold_solver_is_offered_services_and_nothing_else() -> None:
    """Composition-time solving decides install order, so it sees services only.

    Anything wider would let a capability outside the plan remove an ordering
    edge between atoms inside it.
    """

    services = ServiceRegistry()
    services.register("attribution_probe", object(), scope="session")
    assert _service_capabilities(services) == {"service:attribution_probe"}


@pytest.mark.asyncio
async def test_a_plain_service_registration_is_not_announced(tmp_path: Path) -> None:
    """Attribution happens on every write; the event fires only for role binds.

    The driver registers a resource transaction through the plain path once per
    turn. If that started emitting, the event stream would carry one register
    event per turn forever and nothing else here would notice. The role bind is
    the control: it proves the subscription this asserts silence on is live.
    """

    session = await _session(tmp_path)
    try:
        events: list[ApiRegisterEvent] = []
        session.bus.on(ApiRegisterEvent.CHANNEL, events.append)

        session.services.register(RESOURCE_TXN_SERVICE, object(), scope="session")
        assert events == []

        session.services.bind(
            PROVIDER_SESSION_IDENTITY,
            ProviderSessionIdentity(name="direct", model_id="stub-model"),
            replace=True,
        )
        assert [event.name for event in events] == [PROVIDER_SESSION_IDENTITY.key]
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_turn_scoped_service_does_not_disturb_the_composition(
    tmp_path: Path,
) -> None:
    """The driver re-registers a resource transaction every turn.

    It lands in the session's own registry, with no atom installing, so it is
    the embedder's. It must stay out of the rebuildable composition and out of
    any atom's registrations.
    """

    spec = _file_atom(tmp_path, "plain_service_atom", _PLAIN_SERVICE_ATOM)
    session = await _session(tmp_path, spec)
    try:
        before = session.composition_snapshot()
        session.services.register(RESOURCE_TXN_SERVICE, object(), scope="session")
        after = session.composition_snapshot()
        assert after.extensions == before.extensions
        assert after.external_tools == before.external_tools
        assert after.external_trigger_renderers == before.external_trigger_renderers
        assert len(after.external_context_policies) == len(
            before.external_context_policies
        )

        assert session.uninstall_extension(spec)
        assert RESOURCE_TXN_SERVICE in session.services.names()
    finally:
        await session.shutdown()


_THREE_REGISTRATIONS_ATOM = """\
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.abi.messages import TextContent
from agentm.core.abi.tool import FunctionTool, ToolResult


MANIFEST = ExtensionManifest(
    name="three_registrations_atom",
    description="Registers one of each kind the embedder also registers.",
)


class AtomRenderer:
    label = "atom"

    def render(self, trigger):
        del trigger
        return []


class AtomPolicy:
    label = "atom"

    async def transform(self, messages, turns):
        del turns
        return messages


async def _probe(args):
    del args
    return ToolResult(content=(TextContent(type="text", text="ok"),))


def install(api, config):
    del config
    api.register_tool(
        FunctionTool(
            name="atom_tool",
            description="a tool the atom provides",
            parameters={"type": "object", "properties": {}},
            fn=_probe,
        )
    )
    api.register_context_policy(AtomPolicy())
    api.register_trigger_renderer("atom_source", AtomRenderer())
    # The source the embedder already bound. The atom's binding is the later
    # write, so it takes the source over -- in the parent and, once replayed,
    # in the child.
    api.register_trigger_renderer("shared_source", AtomRenderer())
"""


class _HostRenderer:
    label = "host"

    def render(self, trigger: object) -> list[object]:
        del trigger
        return []


class _HostPolicy:
    label = "host"

    async def transform(self, messages: object, turns: object) -> object:
        del turns
        return messages


@pytest.mark.asyncio
async def test_a_child_inherits_the_embedders_tables_and_replays_the_atoms(
    tmp_path: Path,
) -> None:
    """What a child is handed directly, and what it is handed by replay.

    A child rebuilds the atoms from their specs, so the only registrations
    carried over are the ones no replay would reproduce: the embedder's. Those
    are the session's own three tables, handed over whole -- an atom cannot
    write into them, so there is nothing to filter out and no second account of
    who registered what to filter by.

    The contested source is the case where "the host's table" and "the sources
    the host currently owns" are not the same set. The embedder binds it first
    and the atom takes it over, so the parent serves the atom's renderer while
    the host's binding sits shadowed in the host's own table. The child gets
    both, in the same relation, and serves the same renderer the parent does.
    """

    spec = _file_atom(tmp_path, "three_registrations_atom", _THREE_REGISTRATIONS_ATOM)

    async def probe(args: dict[str, object]) -> ToolResult:
        del args
        return ToolResult(content=(TextContent(type="text", text=""),))

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extra_tools=[
                FunctionTool(
                    name="host_tool",
                    description="a tool the embedder provides",
                    parameters={"type": "object", "properties": {}},
                    fn=probe,
                )
            ],
            stream_fn=_StubProvider(),
            model=_model(),
        )
    )
    try:
        session.register_context_policy(_HostPolicy())
        session.register_trigger_renderer("host_source", _HostRenderer())
        session.register_trigger_renderer("shared_source", _HostRenderer())
        await session.install_extension(spec)

        assert session.trigger_renderers["shared_source"].label == "atom"

        child = await session.spawn(purpose="probe")
        try:
            assert isinstance(child, SessionRuntime)
            # The embedder's three, and only the embedder's, in the child's own
            # tables.
            assert [tool.name for tool in child._own_tools] == ["host_tool"]
            assert [type(row.policy).__name__ for row in child._own_policies] == [
                "_HostPolicy"
            ]
            assert sorted(child._own_renderers) == ["host_source", "shared_source"]
            assert child._own_renderers["shared_source"].renderer.label == "host"

            # The atom's three are in the child too, replayed into the child's
            # own context for that atom rather than copied into the tables
            # above -- which is what makes them detachable there.
            ownership = child.ownership()
            atom_tool = next(tool for tool in child.tools if tool.name == "atom_tool")
            assert ownership.tool(atom_tool) == spec.module_path
            assert ownership.renderer("atom_source") == spec.module_path
            assert ownership.renderer("shared_source") == spec.module_path
            assert child.trigger_renderers["shared_source"].label == "atom"
            assert [type(policy).__name__ for policy in child.context_policies] == [
                "_HostPolicy",
                "AtomPolicy",
            ]
        finally:
            await child.shutdown()
    finally:
        await session.shutdown()
