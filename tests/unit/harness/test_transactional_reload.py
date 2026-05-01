from __future__ import annotations

import importlib
from pathlib import Path
import uuid

import pytest

from agentm.harness.events import ExtensionReloadEvent
from agentm.harness.extension import ExtensionStaleError
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig, _LoadedAtom


_PROVIDER_SOURCE = '''
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.kernel import AssistantMessage, MessageEnd, Model, TextContent, ToolCallBlock
from agentm.harness.extension import ProviderConfig


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

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI
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
from agentm.harness.extension import ExtensionAPI

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


def _write_package(tmp_path: Path) -> str:
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
        assert first[2].content[0].content[0].text == "v1"

        api = session._apis[f"{session._test_pkg}.tool_demo"]  # type: ignore[attr-defined]
        result = api.reload_atom("tool_demo", _tool_source("tool_demo", "v2"))

        assert result.ok is True
        second = await session.prompt("hi again")
        assert second[-2].content[0].content[0].text == "v2"
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
        assert follow_up[-2].content[0].content[0].text == "stable"
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
        assert first[2].content[0].content[0].text == "v1"

        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _tool_source("tool_demo", "v2", install_kind="async"),
        )
        assert result.ok is True

        second = await session.prompt("hi again")
        assert second[-2].content[0].content[0].text == "v2"
    finally:
        await session.shutdown()


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
            extensions=[(f"{pkg}.atom_a", {}), (f"{pkg}.atom_b", {})],
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
    session._loaded_atoms_by_name["kernel_loop"] = _LoadedAtom(
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
