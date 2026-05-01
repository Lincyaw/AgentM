from __future__ import annotations

import importlib
import subprocess
import sys
import types
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import AssistantMessage, AssistantStreamEvent, MessageEnd, Model, TextContent
from agentm.core.abi.messages import ToolCallBlock, ToolResultBlock, ToolResultMessage, UserMessage
from agentm.core._internal.catalog.manifest import reload_manifest
from agentm.harness.extension import ProviderConfig
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


_PROVIDER_SOURCE = '''
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.abi import AssistantMessage, MessageEnd, Model, TextContent, ToolCallBlock
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


def _tool_result_text(message: UserMessage | AssistantMessage | ToolResultMessage) -> str:
    assert isinstance(message, ToolResultMessage)
    block = message.content[0]
    assert isinstance(block, ToolResultBlock)
    inner = block.content[0]
    assert isinstance(inner, TextContent)
    return inner.text


def _tool_source(name: str, text: str, *, raises: bool = False) -> str:
    if raises:
        return f'''
from __future__ import annotations

from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name={name!r}, description="boom", registers=("tool:demo",))


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    raise RuntimeError("boom during install")
'''
    return f'''
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name={name!r}, description="reload test atom", registers=("tool:demo",))
CAPTURED_API = None


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    global CAPTURED_API
    CAPTURED_API = api

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


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    (root / "README.md").write_text("baseline\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "initial", "--quiet")


def _write_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentm.core._internal.catalog import manifest as manifest_mod

    manifest_path = tmp_path / "core-manifest.yaml"
    manifest_path.write_text(
        "version: 1\n"
        "constitution:\n"
        "  paths:\n"
        "    - core-manifest.yaml\n"
        "managed:\n"
        "  globs:\n"
        "    - reloadpkg_*/**.py\n"
        "extension_api:\n"
        "  current: 1\n"
        "  semver_rules: {major: x, minor: x, patch: x}\n"
        "  deprecation:\n"
        "    grace: 1\n"
        "reload:\n"
        "  tier_2_atoms: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifest_mod, "_MANIFEST_PATH", manifest_path)
    reload_manifest()


def _write_package(tmp_path: Path, source: str) -> str:
    pkg = f"reloadpkg_{uuid.uuid4().hex[:8]}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "provider.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")
    (pkg_dir / "tool_demo.py").write_text(source, encoding="utf-8")
    return pkg


async def _build_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    atom_source: str,
) -> AgentSession:
    _write_manifest(tmp_path, monkeypatch)
    pkg = _write_package(tmp_path, atom_source)
    _git(tmp_path, "add", pkg)
    _git(tmp_path, "commit", "-m", f"seed {pkg}", "--quiet")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[(f"{pkg}.tool_demo", {})],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    session._test_pkg = pkg  # type: ignore[attr-defined]
    return session


@pytest.mark.asyncio
async def test_G1_reload_atom_commits_via_resource_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repo(tmp_path)
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    tool_path = tmp_path / session._test_pkg / "tool_demo.py"  # type: ignore[attr-defined]
    pre_sha = _git(tmp_path, "rev-parse", "HEAD")
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _tool_source("tool_demo", "v2"),
            rationale="add v2 behavior",
        )
        assert result.ok is True
        assert result.old_hash == pre_sha
        assert result.new_hash == _git(tmp_path, "rev-parse", "HEAD")
        assert result.new_hash is not None and len(result.new_hash) == 40

        follow_up = await session.prompt("still works")
        assert _tool_result_text(follow_up[-2]) == "v2"
    finally:
        await session.shutdown()

    log = _git(tmp_path, "log", "--format=%an|%s", "-n", "1", "--", str(tool_path.relative_to(tmp_path)))
    assert log == "agent|add v2 behavior"


@pytest.mark.asyncio
async def test_G2_reload_install_failure_resets_git_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repo(tmp_path)
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "stable"),
    )
    tool_path = tmp_path / session._test_pkg / "tool_demo.py"  # type: ignore[attr-defined]
    original = tool_path.read_text(encoding="utf-8")
    pre_sha = _git(tmp_path, "rev-parse", "HEAD")
    try:
        result = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _tool_source("tool_demo", "broken", raises=True),
            rationale="break install",
        )
        assert result.ok is False
        assert result.rolled_back is True
        assert _git(tmp_path, "rev-parse", "HEAD") == pre_sha
        assert tool_path.read_text(encoding="utf-8") == original

        follow_up = await session.prompt("still works")
        assert _tool_result_text(follow_up[-2]) == "stable"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_G6_reload_path_supports_rolling_back_to_prior_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repo(tmp_path)
    original_source = _tool_source("tool_demo", "v1")
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=original_source,
    )
    try:
        first = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _tool_source("tool_demo", "v2"),
            rationale="move to v2",
        )
        assert first.ok is True
        second = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            original_source,
            rationale="rollback to v1",
        )
        assert second.ok is True
        assert second.old_hash == first.new_hash
        assert second.new_hash is not None and len(second.new_hash) == 40

        follow_up = await session.prompt("confirm rollback")
        assert _tool_result_text(follow_up[-2]) == "v1"
    finally:
        await session.shutdown()
