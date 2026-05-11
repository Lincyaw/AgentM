from __future__ import annotations

import importlib
import subprocess
import uuid
from pathlib import Path

import pytest

from agentm.core.abi import AssistantMessage, TextContent, ToolResult
from agentm.core.abi.messages import ToolResultBlock, ToolResultMessage, UserMessage
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


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
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name={name!r}, description="boom", registers=("tool:demo",))


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    raise RuntimeError("boom during install")
'''
    return f'''
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

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

                ("agentm.extensions.builtin.operations_local", {}),

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


_AGENT_WRITTEN_ATOM = '''
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

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
async def test_reload_preserves_handler_position_in_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading an atom must NOT silently rerank its handlers — without
    this guarantee, every reload would append to the tail of each channel,
    flipping last-non-None replacement winners. Two atoms subscribe to the
    same channel; we verify dispatch order is invariant under reload of
    the front-running atom.
    """
    _init_repo(tmp_path)
    _write_manifest(tmp_path)

    pkg = f"reloadpkg_{uuid.uuid4().hex[:8]}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "provider.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")

    front_source = '''
from __future__ import annotations
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name="front", description="front", registers=())
TAG = "front-v1"

def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    def _handler(event):
        event["log"].append(TAG)
    api.on("ordering_probe", _handler)
'''
    back_source = '''
from __future__ import annotations
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name="back", description="back", registers=())

def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    def _handler(event):
        event["log"].append("back")
    api.on("ordering_probe", _handler)
'''
    (pkg_dir / "front.py").write_text(front_source, encoding="utf-8")
    (pkg_dir / "back.py").write_text(back_source, encoding="utf-8")
    _git(tmp_path, "add", pkg)
    _git(tmp_path, "commit", "-m", f"seed {pkg}", "--quiet")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),

                (f"{pkg}.front", {}), (f"{pkg}.back", {})],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    try:
        log: list[str] = []
        await session.bus.emit("ordering_probe", {"log": log})
        assert log == ["front-v1", "back"]

        front_v2 = front_source.replace('TAG = "front-v1"', 'TAG = "front-v2"')
        front_api = session._apis[f"{pkg}.front"]  # type: ignore[attr-defined]
        result = front_api.reload_atom("front", front_v2, rationale="bump tag")
        assert result.ok is True

        log.clear()
        await session.bus.emit("ordering_probe", {"log": log})
        # Without position preservation, reload would have moved front to
        # the tail and we'd see ["back", "front-v2"]. The fix keeps front
        # in slot 0.
        assert log == ["front-v2", "back"], log
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_handler_priority_orders_dispatch_across_install_and_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handlers with declared ``BusPriority`` dispatch in tier order
    regardless of install order, and a reload preserves both the tier and
    the within-tier FIFO position.

    Without this guarantee, atom dispatch order on a shared channel would
    track filename / install order — fragile, and broken once already by
    reload silently moving handlers to the tail of every channel.
    """
    _init_repo(tmp_path)
    _write_manifest(tmp_path)

    pkg = f"reloadpkg_{uuid.uuid4().hex[:8]}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "provider.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")

    pre_source = '''
from __future__ import annotations
from agentm.core.abi import BusPriority
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name="pre", description="pre", registers=())
TAG = "pre"

def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    def _handler(event):
        event["log"].append(TAG)
    api.on("ordering_probe", _handler, priority=BusPriority.PRE)
'''
    normal_source = '''
from __future__ import annotations
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name="normal", description="normal", registers=())

def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    def _handler(event):
        event["log"].append("normal")
    api.on("ordering_probe", _handler)
'''
    post_source = '''
from __future__ import annotations
from agentm.core.abi import BusPriority
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(name="post", description="post", registers=())

def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    def _handler(event):
        event["log"].append("post")
    api.on("ordering_probe", _handler, priority=BusPriority.POST)
'''
    (pkg_dir / "pre.py").write_text(pre_source, encoding="utf-8")
    (pkg_dir / "normal.py").write_text(normal_source, encoding="utf-8")
    (pkg_dir / "post.py").write_text(post_source, encoding="utf-8")
    _git(tmp_path, "add", pkg)
    _git(tmp_path, "commit", "-m", f"seed {pkg}", "--quiet")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    # Install order is intentionally backwards (post first, then normal,
    # then pre) to prove install order does not influence dispatch once
    # priority is declared.
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                (f"{pkg}.post", {}),
                (f"{pkg}.normal", {}),
                (f"{pkg}.pre", {}),
            ],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    try:
        log: list[str] = []
        await session.bus.emit("ordering_probe", {"log": log})
        assert log == ["pre", "normal", "post"], log

        # Reload pre, changing its marker. The new handler must land back
        # in the PRE tier (not the tail of the channel).
        pre_v2 = pre_source.replace('TAG = "pre"', 'TAG = "pre-v2"')
        pre_api = session._apis[f"{pkg}.pre"]  # type: ignore[attr-defined]
        result = pre_api.reload_atom("pre", pre_v2, rationale="bump pre tag")
        assert result.ok is True

        log.clear()
        await session.bus.emit("ordering_probe", {"log": log})
        assert log == ["pre-v2", "normal", "post"], log

        # A fourth subscriber registers at POST after the existing post
        # handler. Within-tier FIFO must place it after, not before.
        def _post_2(event: object) -> None:
            assert isinstance(event, dict)
            event["log"].append("post-2")

        from agentm.core.abi import BusPriority

        session.bus.on("ordering_probe", _post_2, priority=BusPriority.POST)

        log.clear()
        await session.bus.emit("ordering_probe", {"log": log})
        assert log == ["pre-v2", "normal", "post", "post-2"], log
    finally:
        await session.shutdown()


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


@pytest.mark.asyncio
async def test_install_then_unload_roundtrip_is_visible_to_running_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end plug-and-play: a running session installs a brand-new atom
    written from an in-memory source string, exercises the tool that atom
    registers, then unloads the atom and confirms the registration is gone.
    Also asserts the two safety gates that, if broken, brick the loop:
    refusing to unload the provider, and refusing to install at a
    constitution path. Single test by design — protects the
    install/unload fail-stop without padding out fine-grained cases.
    """
    from agentm.core.abi.events import ExtensionInstallEvent, ExtensionUnloadEvent

    _init_repo(tmp_path)
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    pkg = session._test_pkg  # type: ignore[attr-defined]
    api = session._apis[f"{pkg}.tool_demo"]  # type: ignore[attr-defined]

    install_events: list[ExtensionInstallEvent] = []
    unload_events: list[ExtensionUnloadEvent] = []
    session.bus.on("extension_install", install_events.append)
    session.bus.on("extension_unload", unload_events.append)

    try:
        # 1. Agent installs a new atom from a source string.
        install_result = api.install_atom(
            name="echo_helper",
            source=_AGENT_WRITTEN_ATOM,
            config={"suffix": "!"},
            rationale="agent self-install for echo capability",
        )
        assert install_result.ok is True, install_result.error
        assert install_result.target_path is not None
        assert Path(install_result.target_path).is_file()
        assert install_result.new_hash is not None
        assert any(
            e.module_path == install_result.module_path and e.phase == "end"
            and e.trigger == "agent"
            for e in install_events
        )

        # 2. The newly installed tool is live on the session.
        echo = next(t for t in session.tools if t.name == "echo_helper")
        result = _as_tool_result(await echo.execute({"msg": "hello"}))
        assert result.is_error is False
        assert result.content[0].text == "hello!"  # type: ignore[union-attr]

        # 3. list_atoms reflects the new atom; api.install_atom rejects
        #    a duplicate name and refuses constitution paths.
        names = [a.name for a in api.list_atoms()]
        assert "echo_helper" in names
        dup = api.install_atom(name="echo_helper", source=_AGENT_WRITTEN_ATOM)
        assert dup.ok is False and "already loaded" in (dup.error or "")
        # The fixture's manifest puts core-manifest.yaml itself in the
        # constitution; refusal there exercises the same gate as a real
        # core/abi/** target would in production.
        constitution_attempt = api.install_atom(
            name="other_helper",
            source=_AGENT_WRITTEN_ATOM.replace("echo_helper", "other_helper"),
            target_path="core-manifest.yaml",
        )
        assert constitution_attempt.ok is False
        assert "constitution" in (constitution_attempt.error or "").lower()

        # 4. Agent unloads the atom; tool registration disappears, the atom's
        #    captured ExtensionAPI becomes stale, and an unload event fires.
        unload_result = api.unload_atom("echo_helper")
        assert unload_result.ok is True
        assert all(t.name != "echo_helper" for t in session.tools)
        assert any(
            e.name == "echo_helper" and e.trigger == "agent"
            for e in unload_events
        )

        # 5. Provider can never be unloaded — would leave the loop without
        #    a stream_fn. The atom_name for the provider in this fixture
        #    is "rollbackpkg_*.provider" — find it via list_atoms.
        provider_names = [
            a.name for a in api.list_atoms()
            if "provider" in a.name.lower()
        ]
        # Fall back: any module marked is_provider in the reloader registry.
        loaded = session._reloader.loaded_by_name  # type: ignore[attr-defined]
        provider_atom = next(
            (a for a in loaded.values() if a.is_provider), None
        )
        assert provider_atom is not None, (provider_names, list(loaded))
        provider_unload = api.unload_atom(provider_atom.name)
        assert provider_unload.ok is False
        assert "provider" in (provider_unload.error or "").lower()
    finally:
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
from agentm.core.abi.extension import ProviderConfig


_NEW_ATOM_SOURCE = """\\
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

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

                ("agentm.extensions.builtin.operations_local", {}),
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
