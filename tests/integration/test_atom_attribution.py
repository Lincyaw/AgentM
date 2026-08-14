"""Uninstall removes what the atom put there, and nothing else does.

An atom's registrations are recovered from an ownership ledger keyed on the
atom that made them. Anything the ledger never saw survives uninstall forever,
so each test here installs one kind of registration and asserts it is gone
again. The last two guard the other direction: a requirement is spelled in
manifest names, and the driver's per-turn service is nobody's registration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agentm import AgentSession, AgentSessionConfig, ExtensionSpec, Model
from agentm.core.abi.roles import RESOURCE_TXN_SERVICE


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
async def test_turn_scoped_service_does_not_disturb_the_composition(
    tmp_path: Path,
) -> None:
    """The driver re-registers a resource transaction every turn.

    That write now reaches the ownership ledger like any other, with no atom
    installing, so it is recorded as the embedder's. It must stay out of the
    rebuildable composition and out of any atom's registrations.
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
