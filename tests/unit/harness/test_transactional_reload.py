from __future__ import annotations

import importlib
from pathlib import Path
import sys
import uuid

import pytest

from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)
from agentm.core.abi.events import ExtensionReloadEvent
from agentm.core.abi.extension import ExtensionStaleError
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.atom_reloader import LoadedAtom as _LoadedAtom
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


def _tool_result_text(message: UserMessage | AssistantMessage | ToolResultMessage) -> str:
    """Extract the text payload from a ToolResultMessage."""
    assert isinstance(message, ToolResultMessage)
    block = message.content[0]
    assert isinstance(block, ToolResultBlock)
    inner = block.content[0]
    assert isinstance(inner, TextContent)
    return inner.text


_PROVIDER_SOURCE = '''
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.abi import AssistantMessage, MessageEnd, Model, TextContent, ToolCallBlock
from agentm.core.abi.extension import ProviderConfig


class _Stream:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, messages: list[Any], model: Model, tools: list[Any], system: str | None = None, signal: Any = None, thinking: str = "off") -> AsyncIterator[Any]:
        self.calls += 1
        return self._iter(self.calls)

    async def _iter(self, call_no: int) -> AsyncIterator[Any]:
        if call_no % 2 == 1:
            yield MessageEnd(message=AssistantMessage(role="assistant", content=[ToolCallBlock(type="tool_call", id=f"call-{call_no}", name="demo", arguments={})], timestamp=float(call_no), stop_reason="tool_use"))
        else:
            yield MessageEnd(message=AssistantMessage(role="assistant", content=[TextContent(type="text", text="done")], timestamp=float(call_no), stop_reason="end_turn"))


def install(api, config):
    api.register_provider(
        "reload-provider",
        ProviderConfig(
            stream_fn=_Stream(),
            model=Model(id="reload-provider", provider="reload", context_window=4096, max_output_tokens=512),
            name="reload-provider",
        ),
    )
'''


def _tool_source(
    name: str,
    text: str,
    *,
    marker_label: str | None = None,
    install_kind: str = "sync",
    registers: tuple[str, ...] = ("tool:demo",),
) -> str:
    state_import = ""
    handler = ""
    if marker_label is not None:
        state_import = "from reload_state_shared import EVENTS\n"
        handler = f'''
    def _on_marker(event):
        EVENTS.append(({marker_label!r}, event))

    api.on("marker", _on_marker)
'''
    install_prefix = "async " if install_kind == "async" else ""
    return f'''
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI
{state_import}

MANIFEST = ExtensionManifest(
    name={name!r},
    description="reload test atom",
    registers={registers!r},
)
CAPTURED_API = None


{install_prefix}def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    global CAPTURED_API
    CAPTURED_API = api
{handler}
    async def _execute(args: dict[str, object]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text={text!r})])

    api.register_tool(
        FunctionTool(
            name="demo",
            description="demo tool",
            parameters={{"type": "object", "properties": {{}}, "additionalProperties": False}},
            fn=_execute,
        )
    )
'''


def _raising_source(name: str) -> str:
    return f'''
from __future__ import annotations

from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name={name!r}, description="boom", registers=("tool:demo",))


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    raise RuntimeError("boom during install")
'''


def _invalid_signature_source(name: str) -> str:
    return f'''
from __future__ import annotations

from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(name={name!r}, description="bad", registers=("tool:demo",))


def install(api):
    return None
'''


def _observer_source(name: str, label: str) -> str:
    return f'''
from __future__ import annotations

from reload_state_shared import EVENTS

from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name={name!r}, description="observer", registers=())


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    def _observe(channel, event):
        EVENTS.append(({label!r}, channel))

    api.add_observer(_observe)
'''


def _write_package(tmp_path: Path) -> str:
    sys.modules.pop("reload_state_shared", None)
    pkg = f"reloadpkg_{uuid.uuid4().hex[:8]}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "state.py").write_text("EVENTS = []\n", encoding="utf-8")
    (pkg_dir / "provider.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")
    (tmp_path / "reload_state_shared.py").write_text("EVENTS = []\n", encoding="utf-8")
    return pkg


async def _build_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, atom_source: str, extra_extensions: list[tuple[str, dict]] | None = None) -> AgentSession:
    pkg = _write_package(tmp_path)
    (tmp_path / pkg / "tool_demo.py").write_text(atom_source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.operations_local", {}),
                *((extra_extensions or [])),
                (f"{pkg}.tool_demo", {}),
            ],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    session._test_pkg = pkg  # type: ignore[attr-defined]
    return session


@pytest.mark.asyncio
async def test_S1_reload_tool_atom_takes_effect_next_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    try:
        first = await session.prompt("hi")
        assert _tool_result_text(first[2]) == "v1"

        api = session._apis[f"{session._test_pkg}.tool_demo"]  # type: ignore[attr-defined]
        result = api.reload_atom("tool_demo", _tool_source("tool_demo", "v2"))

        assert result.ok is True
        second = await session.prompt("hi again")
        assert _tool_result_text(second[-2]) == "v2"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_S4_syntax_error_rejected_no_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    tool_path = tmp_path / session._test_pkg / "tool_demo.py"  # type: ignore[attr-defined]
    original = tool_path.read_text(encoding="utf-8")
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo", _invalid_signature_source("tool_demo")
        )
        assert result.ok is False
        assert "install" in (result.error or "")
        assert tool_path.read_text(encoding="utf-8") == original
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_reload_rejects_invalid_register_tag_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    tool_path = tmp_path / session._test_pkg / "tool_demo.py"  # type: ignore[attr-defined]
    original = tool_path.read_text(encoding="utf-8")
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _tool_source("tool_demo", "v2", registers=("badtag",)),
        )
        assert result.ok is False
        assert "register" in (result.error or "")
        assert tool_path.read_text(encoding="utf-8") == original
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_S5_install_failure_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "stable"),
    )
    tool_path = tmp_path / session._test_pkg / "tool_demo.py"  # type: ignore[attr-defined]
    original = tool_path.read_text(encoding="utf-8")
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo", _raising_source("tool_demo")
        )
        assert result.ok is False
        assert result.rolled_back is True
        assert tool_path.read_text(encoding="utf-8") == original

        follow_up = await session.prompt("still works")
        assert _tool_result_text(follow_up[-2]) == "stable"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_reload_supports_async_install_atoms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1", install_kind="async"),
    )
    try:
        first = await session.prompt("hi")
        assert _tool_result_text(first[2]) == "v1"

        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _tool_source("tool_demo", "v2", install_kind="async"),
        )
        assert result.ok is True

        second = await session.prompt("hi again")
        assert _tool_result_text(second[-2]) == "v2"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_reload_removes_owner_observer_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_observer_source("tool_demo", "v1"),
    )
    state = importlib.import_module("reload_state_shared")
    try:
        state.EVENTS.clear()
        await session.bus.emit("marker", {"value": 1})
        assert state.EVENTS == [("v1", "marker")]

        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo", _observer_source("tool_demo", "v2")
        )
        assert result.ok is True

        state.EVENTS.clear()
        await session.bus.emit("marker", {"value": 2})
        assert state.EVENTS == [("v2", "marker")]
    finally:
        await session.shutdown()
        state.EVENTS.clear()


@pytest.mark.asyncio
async def test_S6_assert_active_raises_after_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    module = importlib.import_module(f"{session._test_pkg}.tool_demo")  # type: ignore[attr-defined]
    old_api = module.CAPTURED_API
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo", _tool_source("tool_demo", "v2")
        )
        assert result.ok is True
        with pytest.raises(ExtensionStaleError, match=r"Re-acquire via the new install\(\) call"):
            _ = old_api.cwd
        assert old_api.events is session.bus
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_M4_per_atom_api_instances_distinct_and_owner_name_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _write_package(tmp_path)
    (tmp_path / pkg / "atom_a.py").write_text(_tool_source("atom_a", "a"), encoding="utf-8")
    (tmp_path / pkg / "atom_b.py").write_text(_tool_source("atom_b", "b"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.operations_local", {}),
                (f"{pkg}.atom_a", {}),
                (f"{pkg}.atom_b", {}),
            ],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    try:
        mod_a = importlib.import_module(f"{pkg}.atom_a")
        mod_b = importlib.import_module(f"{pkg}.atom_b")
        assert mod_a.CAPTURED_API is not mod_b.CAPTURED_API
        assert mod_a.CAPTURED_API._owner_name == f"{pkg}.atom_a"
        assert mod_b.CAPTURED_API._owner_name == f"{pkg}.atom_b"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_reload_emits_extension_reload_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    seen: list[ExtensionReloadEvent] = []
    session.bus.on("extension_reload", lambda event: seen.append(event))
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo", _tool_source("tool_demo", "v2")
        )
        assert result.ok is True
        assert len(seen) == 1
        assert seen[0].name == "tool_demo"
        assert seen[0].old_hash is not None
        assert seen[0].new_hash == result.new_hash
        assert seen[0].trigger == "agent"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_reload_path_check_rejects_constitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    session._reloader.loaded_by_name["kernel_loop"] = _LoadedAtom(
        name="kernel_loop",
        module_path="agentm.harness.session",
        file_path=Path("src/agentm/harness/session.py"),
        config={},
        manifest=None,
        is_provider=False,
    )
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "kernel_loop", "print('nope')\n"
        )
        assert result.ok is False
        assert "constitution" in (result.error or "")
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_reload_invalidates_old_handlers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1", marker_label="v1"),
    )
    state = importlib.import_module("reload_state_shared")
    try:
        await session.bus.emit("marker", "before")
        assert state.EVENTS == [("v1", "before")]

        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo", _tool_source("tool_demo", "v2", marker_label="v2")
        )
        assert result.ok is True

        await session.bus.emit("marker", "after")
        assert state.EVENTS == [("v1", "before"), ("v2", "after")]
    finally:
        await session.shutdown()

@pytest.mark.asyncio
async def test_reload_double_failure_preserves_loaded_atom_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "stable"),
    )
    module_path = f"{session._test_pkg}.tool_demo"  # type: ignore[attr-defined]
    original_atom = session._reloader.loaded_by_name["tool_demo"]
    original_api = session._apis[module_path]  # type: ignore[attr-defined]
    original_module = sys.modules[module_path]
    original_activate = session._reloader._activate_atom_install  # type: ignore[attr-defined]
    calls = 0

    async def fail_only_rollback(atom: _LoadedAtom) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await original_activate(atom)
            return
        raise RuntimeError("rollback activation exploded")

    monkeypatch.setattr(
        session._reloader,  # type: ignore[attr-defined]
        "_activate_atom_install",
        fail_only_rollback,
    )
    try:
        result = session._apis[module_path].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _raising_source("tool_demo"),
            rationale="exercise double failure",
        )
        assert result.ok is False
        assert "rollback_failure_state_preserved" in (result.error or "")
        assert session._reloader.loaded_by_name["tool_demo"] is original_atom
        assert session._reloader.loaded_by_module[module_path] is original_atom  # type: ignore[attr-defined]
        assert session._apis[module_path] is original_api  # type: ignore[attr-defined]
        assert sys.modules[module_path] is original_module
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_reload_double_failure_preserves_bus_subscriptions_and_registrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot-based rollback must preserve handler/registration tracking,
    not just the loaded-atom registry. Pre-reload the atom registers a
    handler on the ``marker`` channel; both apply and rollback then fail.
    Post-restore, emitting ``marker`` must still hit the original handler
    once and exactly once (proving handler list, owners_by_kind, and bus
    subscriptions all came back from the immutable snapshot)."""
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "stable", marker_label="v1"),
    )
    module_path = f"{session._test_pkg}.tool_demo"  # type: ignore[attr-defined]
    state = importlib.import_module("reload_state_shared")
    original_activate = session._reloader._activate_atom_install  # type: ignore[attr-defined]
    pre_handlers = list(session._reloader._handlers_by_atom.get(module_path, []))  # type: ignore[attr-defined]
    pre_registrations = list(
        session._reloader._registrations_by_atom.get(module_path, [])  # type: ignore[attr-defined]
    )
    pre_marker_subs = session.bus.subscriptions_for("marker")
    calls = 0

    async def fail_only_rollback(atom: _LoadedAtom) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await original_activate(atom)
            return
        raise RuntimeError("rollback activation exploded")

    monkeypatch.setattr(
        session._reloader,  # type: ignore[attr-defined]
        "_activate_atom_install",
        fail_only_rollback,
    )
    try:
        result = session._apis[module_path].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _raising_source("tool_demo"),
            rationale="exercise double failure with subscriptions",
        )
        assert result.ok is False
        assert "rollback_failure_state_preserved" in (result.error or "")

        # Snapshot restoration must rebuild handlers / registrations exactly,
        # not leave the registries half-populated.
        assert (
            session._reloader._handlers_by_atom.get(module_path, []) == pre_handlers  # type: ignore[attr-defined]
        )
        assert (
            session._reloader._registrations_by_atom.get(module_path, [])  # type: ignore[attr-defined]
            == pre_registrations
        )
        assert session._reloader.owners_by_kind["tool"]["demo"] == module_path  # type: ignore[attr-defined]

        # The bus must hold the same subscription objects (same identities,
        # same order) as before the reload attempt — no orphaned post-apply
        # handlers, no missing pre-apply ones.
        post_marker_subs = session.bus.subscriptions_for("marker")
        assert [sub.handler for sub in post_marker_subs] == [
            sub.handler for sub in pre_marker_subs
        ]

        state.EVENTS.clear()
        await session.bus.emit("marker", {"value": "after-double-failure"})
        # Exactly one ``v1`` handler should fire; if rollback had left a
        # second handler subscribed we'd see two events.
        assert state.EVENTS == [("v1", {"value": "after-double-failure"})]
    finally:
        await session.shutdown()
        state.EVENTS.clear()


_OWNER_KIND_SOURCE = '''
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.abi import FunctionTool, MessageEnd, Model, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import CommandSpec, ExtensionAPI, ProviderConfig

MANIFEST = ExtensionManifest(
    name="tool_demo",
    description="owner tracking test atom",
    registers=("tool:demo", "command:demo_cmd", "provider:demo_provider", "renderer:demo_renderer"),
)


async def _stream(**kwargs: Any) -> AsyncIterator[Any]:
    yield MessageEnd(message=None)


def _command(args: str, api: ExtensionAPI) -> None:
    return None


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    async def _execute(args: dict[str, object]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="owner")])

    api.register_tool(
        FunctionTool(
            name="demo",
            description="demo tool",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            fn=_execute,
        )
    )
    api.register_command("demo_cmd", CommandSpec(description="demo", handler=_command))
    api.register_provider(
        "demo_provider",
        ProviderConfig(
            stream_fn=_stream,
            model=Model(id="demo", provider="demo", context_window=1024, max_output_tokens=128),
            name="demo_provider",
        ),
    )
    api.register_message_renderer("demo_renderer", lambda payload: "rendered")
'''


@pytest.mark.asyncio
async def test_registration_owners_are_tracked_for_every_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_OWNER_KIND_SOURCE,
    )
    owner = f"{session._test_pkg}.tool_demo"  # type: ignore[attr-defined]
    try:
        assert session._reloader.owners_by_kind["tool"]["demo"] == owner  # type: ignore[attr-defined]
        assert session._reloader.owners_by_kind["command"]["demo_cmd"] == owner  # type: ignore[attr-defined]
        assert session._reloader.owners_by_kind["provider"]["demo_provider"] == owner  # type: ignore[attr-defined]
        assert session._reloader.owners_by_kind["renderer"]["demo_renderer"] == owner  # type: ignore[attr-defined]
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_session_shutdown_unsubscribes_reloader_registration_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentm.core.abi import EventBus

    class NonClearingEventBus(EventBus):
        def clear(self) -> None:
            return None

    bus = NonClearingEventBus()
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
        extra_extensions=[],
    )
    # Rebuild with an externally visible bus because _build_session intentionally
    # keeps its fixture compact for the other reload tests.
    await session.shutdown()

    pkg = _write_package(tmp_path)
    (tmp_path / pkg / "tool_demo.py").write_text(_tool_source("tool_demo", "v1"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            bus=bus,
            extensions=[
                ("agentm.extensions.builtin.operations_local", {}),
                (f"{pkg}.tool_demo", {}),
            ],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    track_handler = session._reloader._track_registration  # type: ignore[attr-defined]
    assert any(
        sub.handler == track_handler
        for subs in bus._handlers.values()
        for sub in subs
    )

    await session.shutdown()

    assert not any(
        sub.handler == track_handler
        for subs in bus._handlers.values()
        for sub in subs
    )


@pytest.mark.asyncio
async def test_agent_installed_atom_records_synthetic_import_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    api = session._apis[f"{session._test_pkg}.tool_demo"]  # type: ignore[attr-defined]
    try:
        result = api.install_atom(
            name="helper_atom",
            source=_tool_source("helper_atom", "helper", registers=("tool:helper",)).replace('name="demo"', 'name="helper"'),
        )
        assert result.ok is True
        assert session._reloader.loaded_by_name["helper_atom"].import_kind == "synthetic"
    finally:
        await session.shutdown()
