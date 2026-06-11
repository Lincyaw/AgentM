from __future__ import annotations

import importlib
import subprocess
import uuid
from pathlib import Path

import pytest

from agentm.core.abi import AssistantMessage, TextContent, ToolResult
from agentm.core.abi import ToolResultBlock, ToolResultMessage, UserMessage
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


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


def _as_tool_result(result: object) -> ToolResult:
    assert isinstance(result, ToolResult)
    return result


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
from agentm.core.abi import ExtensionAPI

MANIFEST = ExtensionManifest(name={name!r}, description="boom", registers=("tool:demo",))


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    raise RuntimeError("boom during install")
'''
    return f'''
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi import ExtensionAPI

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
    _git(root, "init", "-q", "-b", "agent-tests")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    (root / "README.md").write_text("baseline\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "initial", "--quiet")


def _write_manifest(tmp_path: Path) -> None:
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
    manifest_mod.configure_manifest_path(manifest_path)


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
    _write_manifest(tmp_path)
    pkg = _write_package(tmp_path, atom_source)
    _git(tmp_path, "add", pkg)
    _git(tmp_path, "commit", "-m", f"seed {pkg}", "--quiet")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[

                ("agentm.extensions.builtin.operations", {}),

                (f"{pkg}.tool_demo", {})],
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




_AGENT_WRITTEN_ATOM = '''
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="echo_helper",
    description="A tiny helper the agent installed at runtime.",
    registers=("tool:echo_helper",),
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    suffix = str(config.get("suffix", ""))

    async def _execute(args: dict[str, object]) -> ToolResult:
        text = f"{args.get('msg', '')}{suffix}"
        return ToolResult(content=[TextContent(type="text", text=text)])

    api.register_tool(
        FunctionTool(
            name="echo_helper",
            description="echo helper installed at runtime",
            parameters={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "additionalProperties": False,
            },
            fn=_execute,
        )
    )
'''






@pytest.mark.asyncio
async def test_before_install_atom_handler_can_veto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-cutting policy can refuse a self-install via a
    ``before_install_atom`` handler returning ``{"block": True, "reason":
    ...}``. The harness returns ok=False with the reason; nothing is
    written to disk. Symmetric for unload.
    """
    _init_repo(tmp_path)
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    pkg = session._test_pkg  # type: ignore[attr-defined]
    api = session._apis[f"{pkg}.tool_demo"]  # type: ignore[attr-defined]

    def _veto_handler(event):
        if event.name == "blocked_atom":
            return {"block": True, "reason": "policy-rejects-blocked_atom"}
        return None

    session.bus.on("before_install_atom", _veto_handler)

    blocked = api.install_atom(
        name="blocked_atom",
        source=_AGENT_WRITTEN_ATOM.replace("echo_helper", "blocked_atom"),
    )
    assert blocked.ok is False
    assert "policy-rejects-blocked_atom" in (blocked.error or "")
    # Disk wasn't touched: no file under .agentm/atoms/.
    assert not (tmp_path / ".agentm" / "atoms" / "blocked_atom.py").exists()

    # Sanity: a different name passes through to the install pipeline.
    permitted = api.install_atom(
        name="echo_helper",
        source=_AGENT_WRITTEN_ATOM,
    )
    assert permitted.ok is True
    api.unload_atom("echo_helper")

    await session.shutdown()




_INSTALL_THEN_USE_STUB_PROVIDER = '''
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.abi import (
    AssistantMessage,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
    ToolResult,
)
from agentm.core.abi import ProviderConfig


_NEW_ATOM_SOURCE = """\\
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="agent_written_shout",
    description="Atom the agent wrote and installed mid-prompt.",
    registers=("tool:agent_written_shout",),
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    async def _exec(args: dict[str, object]) -> ToolResult:
        text = str(args.get("text", ""))
        return ToolResult(
            content=[TextContent(type="text", text=text.upper())]
        )

    api.register_tool(
        FunctionTool(
            name="agent_written_shout",
            description="Return text uppercased.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            fn=_exec,
        )
    )
"""


class _Stream:
    """Three-turn stub: install_atom -> agent_written_shout -> end_turn.

    Verifies the kernel loop rebuilds its dispatch index per turn so an
    atom registered by ``install_atom`` in turn N becomes callable in
    turn N+1 within the same ``session.prompt`` invocation.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[Any]:
        self.calls += 1
        return self._iter(self.calls, tools)

    async def _iter(self, call_no: int, tools: list[Any]) -> AsyncIterator[Any]:
        if call_no == 1:
            yield MessageEnd(message=AssistantMessage(
                role="assistant",
                content=[ToolCallBlock(
                    type="tool_call",
                    id="call-install",
                    name="install_atom",
                    arguments={
                        "name": "agent_written_shout",
                        "source": _NEW_ATOM_SOURCE,
                        "rationale": "agent self-mod test",
                    },
                )],
                timestamp=1.0,
                stop_reason="tool_use",
            ))
        elif call_no == 2:
            assert any(t.name == "agent_written_shout" for t in tools), (
                "agent_written_shout missing from per-turn tool list — "
                "the kernel snapshotted tools at run start instead of "
                "passing the live registry"
            )
            yield MessageEnd(message=AssistantMessage(
                role="assistant",
                content=[ToolCallBlock(
                    type="tool_call",
                    id="call-shout",
                    name="agent_written_shout",
                    arguments={"text": "hello agentm"},
                )],
                timestamp=2.0,
                stop_reason="tool_use",
            ))
        else:
            yield MessageEnd(message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                timestamp=3.0,
                stop_reason="end_turn",
            ))


def install(api, config):
    api.register_provider(
        "install-then-use",
        ProviderConfig(
            stream_fn=_Stream(),
            model=Model(
                id="install-then-use",
                provider="install-then-use",
                context_window=4096,
                max_output_tokens=512,
            ),
            name="install-then-use",
        ),
    )
'''


@pytest.mark.asyncio
async def test_install_atom_in_turn_n_is_dispatchable_in_turn_n_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-prompt plug-and-play: a stub LLM emits a ``tool_call`` for
    ``install_atom`` in turn 1, then a ``tool_call`` for the brand-new
    atom's tool in turn 2, all inside one ``session.prompt`` invocation.

    Fail-stop protected: if the kernel loop builds its dispatch index
    once at the start of ``run()`` (the pre-fix behavior in
    ``core/abi/loop.py``), turn 2 dispatches "Unknown tool" and the
    final tool_result for ``agent_written_shout`` carries
    ``is_error=True`` with that message — which this test asserts
    against. The bug it pins:  agent writes a self-mod, agent calls it
    next turn, framework returns "Unknown tool" because tool_index was
    snapshot-stale.
    """

    _init_repo(tmp_path)
    _write_manifest(tmp_path)
    pkg = f"reloadpkg_{uuid.uuid4().hex[:8]}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "provider.py").write_text(
        _INSTALL_THEN_USE_STUB_PROVIDER, encoding="utf-8"
    )
    _git(tmp_path, "add", pkg)
    _git(tmp_path, "commit", "-m", f"seed {pkg}", "--quiet")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[

                ("agentm.extensions.builtin.operations", {}),
                ("contrib.extensions.tool_catalog.browse", {}),
                ("contrib.extensions.tool_catalog.mutate", {}),
            ],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    try:
        messages = await session.prompt("install agent_written_shout and call it")

        # Both tool_results must be present and the second must succeed —
        # i.e. the kernel found agent_written_shout in tool_index after
        # the install_atom call in the prior turn.
        tool_results = [
            m for m in messages
            if isinstance(m, ToolResultMessage)
        ]
        assert len(tool_results) >= 2, (
            f"expected install + use tool_results, got {len(tool_results)}"
        )

        install_tr = next(
            m for m in tool_results
            if isinstance(m.content[0], ToolResultBlock)
            and m.content[0].tool_call_id == "call-install"
        )
        shout_tr = next(
            m for m in tool_results
            if isinstance(m.content[0], ToolResultBlock)
            and m.content[0].tool_call_id == "call-shout"
        )
        assert install_tr.content[0].is_error is False, (
            f"install_atom failed: {_tool_result_text(install_tr)!r}"
        )
        assert shout_tr.content[0].is_error is False, (
            "newly-installed tool was not dispatched in the same prompt — "
            f"got: {_tool_result_text(shout_tr)!r}"
        )
        assert _tool_result_text(shout_tr) == "HELLO AGENTM"
    finally:
        await session.shutdown()
