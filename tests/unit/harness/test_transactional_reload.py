"""Transactional reload atomicity: success takes effect, failure rolls back."""

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
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.session import AgentSession


def _tool_result_text(message: UserMessage | AssistantMessage | ToolResultMessage) -> str:
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
from agentm.core.abi import AssistantMessage, MessageEnd, Model, TextContent, ToolCallBlock, ProviderConfig

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
    api.register_provider("reload-provider", ProviderConfig(stream_fn=_Stream(), model=Model(id="reload-provider", provider="reload", context_window=4096, max_output_tokens=512), name="reload-provider"))
'''


def _tool_source(name: str, text: str) -> str:
    return f'''
from __future__ import annotations
from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi import ExtensionAPI

MANIFEST = ExtensionManifest(name={name!r}, description="reload test atom", registers=("tool:demo",))

def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    async def _execute(args: dict[str, object]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text={text!r})])
    api.register_tool(FunctionTool(name="demo", description="demo tool", parameters={{"type": "object", "properties": {{}}, "additionalProperties": False}}, fn=_execute))
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


async def _build_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, atom_source: str) -> AgentSession:
    pkg = f"reloadpkg_{uuid.uuid4().hex[:8]}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "provider.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")
    (pkg_dir / "tool_demo.py").write_text(atom_source, encoding="utf-8")
    (tmp_path / "reload_state_shared.py").write_text("EVENTS = []\n", encoding="utf-8")
    sys.modules.pop("reload_state_shared", None)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    session = await AgentSession.create(AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[("agentm.extensions.builtin.operations", {}), (f"{pkg}.tool_demo", {})],
        provider=(f"{pkg}.provider", {}),
        resource_loader=InMemoryResourceLoader(),
    ))
    session._test_pkg = pkg  # type: ignore[attr-defined]
    return session


@pytest.mark.asyncio
async def test_S1_reload_tool_atom_takes_effect_next_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(tmp_path, monkeypatch, atom_source=_tool_source("tool_demo", "v1"))
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
async def test_S5_install_failure_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = await _build_session(tmp_path, monkeypatch, atom_source=_tool_source("tool_demo", "stable"))
    tool_path = tmp_path / session._test_pkg / "tool_demo.py"  # type: ignore[attr-defined]
    original = tool_path.read_text(encoding="utf-8")
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo", _raising_source("tool_demo"),
        )
        assert result.ok is False
        assert result.rolled_back is True
        assert tool_path.read_text(encoding="utf-8") == original

        follow_up = await session.prompt("still works")
        assert _tool_result_text(follow_up[-2]) == "stable"
    finally:
        await session.shutdown()
