from __future__ import annotations

import importlib
from pathlib import Path
import sys
import uuid

import pytest

from agentm.core.abi import (
    AssistantMessage,
    TextContent,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)
from agentm.core.abi import ExtensionStaleError
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
from agentm.core.abi import ProviderConfig


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
from agentm.core.abi import ExtensionAPI
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
from agentm.core.abi import ExtensionAPI

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
from agentm.core.abi import ExtensionAPI

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
                ("agentm.extensions.builtin.operations", {}),
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
async def test_reload_path_check_rejects_constitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    session._reloader.loaded_by_name["kernel_loop"] = _LoadedAtom(
        name="kernel_loop",
        module_path="agentm.core.runtime.session",
        file_path=Path("src/agentm/core/runtime/session.py"),
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




_OWNER_KIND_SOURCE = '''
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.abi import FunctionTool, MessageEnd, Model, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi import CommandSpec, ExtensionAPI, ProviderConfig

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






