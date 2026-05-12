from __future__ import annotations

import importlib
import json
import subprocess
import uuid
from pathlib import Path

import pytest

from agentm.core.abi import AssistantMessage, EventBus, TextContent, ToolResult
from agentm.core.abi.messages import ToolResultBlock, ToolResultMessage, UserMessage
from agentm.core.runtime.catalog import _layout
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.resource_writer import GitBackedResourceWriter
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
        "rollback-provider",
        ProviderConfig(
            stream_fn=_Stream(),
            model=Model(id="rollback-provider", provider="rollback", context_window=4096, max_output_tokens=512),
            name="rollback-provider",
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


def _tool_source(name: str, text: str) -> str:
    return f'''
from __future__ import annotations

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name={name!r},
    description="rollback test atom",
    registers=("tool:demo",),
    api_version=1,
    affects=("demo.success_rate",),
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
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
        "    - rollbackpkg_*/**.py\n"
        "    - skills/**\n"
        "    - prompts/**\n"
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
    pkg = f"rollbackpkg_{uuid.uuid4().hex[:8]}"
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
                (f"{pkg}.tool_demo", {}),
                ("contrib.extensions.tool_catalog.browse", {}),
                ("contrib.extensions.tool_catalog.mutate", {}),
            ],
            provider=(f"{pkg}.provider", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    session._test_pkg = pkg  # type: ignore[attr-defined]
    return session


@pytest.mark.asyncio
async def test_G6_tool_catalog_history_source_and_forceable_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repo(tmp_path)
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    pkg = session._test_pkg  # type: ignore[attr-defined]
    tool_path = f"{pkg}/tool_demo.py"
    try:
        first_sha = _git(tmp_path, "log", "-n", "1", "--format=%H", "--", tool_path)
        upgrade = session._apis[f"{session._test_pkg}.tool_demo"].reload_atom(  # type: ignore[attr-defined]
            "tool_demo",
            _tool_source("tool_demo", "v2"),
            rationale="upgrade to v2",
        )
        assert upgrade.ok is True
        second_sha = _git(tmp_path, "rev-parse", "HEAD")

        history_tool = next(tool for tool in session.tools if tool.name == "list_history")
        history = _as_tool_result(
            await history_tool.execute({"path": "tool_demo", "limit": 2})
        )
        assert history.is_error is False
        assert [entry["sha"] for entry in history.extras] == [second_sha, first_sha]
        assert [entry["author"] for entry in history.extras] == ["agent", "Test User"]
        assert [entry["message"] for entry in history.extras] == [
            "upgrade to v2",
            f"seed {pkg}",
        ]

        source_tool = next(tool for tool in session.tools if tool.name == "get_source_at")
        old_source = _as_tool_result(
            await source_tool.execute({"path": "tool_demo", "sha": first_sha})
        )
        assert old_source.is_error is False
        assert "v1" in old_source.extras
        assert _tool_result_text((await session.prompt("before rollback"))[-2]) == "v2"

        decisions_path = _layout.atom_decisions_path("tool_demo", first_sha, root=tmp_path)
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_path.write_text(
            json.dumps({"kind": "regressed", "reason": "guard tripped"}) + "\n",
            encoding="utf-8",
        )

        rollback_tool = next(tool for tool in session.tools if tool.name == "rollback_resource")
        blocked = _as_tool_result(
            await rollback_tool.execute(
                {
                    "path": "tool_demo",
                    "target_sha": first_sha,
                    "rationale": "undo regression",
                }
            )
        )
        assert blocked.is_error is True
        assert blocked.extras["error"] == "rollback_blocked"
        assert blocked.extras["decision"]["kind"] == "regressed"

        rolled_back = _as_tool_result(
            await rollback_tool.execute(
                {
                    "path": "tool_demo",
                    "target_sha": first_sha,
                    "rationale": "undo regression",
                    "force": True,
                }
            )
        )
        assert rolled_back.is_error is False
        assert rolled_back.extras["ok"] is True
        assert rolled_back.extras["old_hash"] == second_sha
        assert _tool_result_text((await session.prompt("after rollback"))[-2]) == "v1"
        latest_log = _git(tmp_path, "log", "--format=%an|%s", "-n", "1", "--", tool_path)
        assert latest_log == f"agent|rollback to {first_sha[:8]}: undo regression"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_rollback_by_repo_relative_path_still_reloads_atom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rollback_resource(path=<repo-relative .py>) must go through reload_atom
    (transactional), not plain writer.write — the tool's own schema promises
    "atom name, repo-relative path, or absolute path" all work, and the
    self-modifiable-architecture invariant is that atom rollbacks are
    transactional. Regression for the path-form silently degrading to a
    non-reloading writer.write call.
    """
    _init_repo(tmp_path)
    session = await _build_session(
        tmp_path,
        monkeypatch,
        atom_source=_tool_source("tool_demo", "v1"),
    )
    pkg = session._test_pkg  # type: ignore[attr-defined]
    rel_path = f"{pkg}/tool_demo.py"

    def _api():
        return session._apis[f"{pkg}.tool_demo"]  # type: ignore[attr-defined]

    try:
        first_sha = _git(tmp_path, "log", "-n", "1", "--format=%H", "--", rel_path)
        _api().reload_atom(
            "tool_demo",
            _tool_source("tool_demo", "v2"),
            rationale="upgrade to v2",
        )
        assert _tool_result_text((await session.prompt("before"))[-2]) == "v2"

        rollback_tool = next(t for t in session.tools if t.name == "rollback_resource")
        result_rel = _as_tool_result(
            await rollback_tool.execute(
                {
                    "path": rel_path,
                    "target_sha": first_sha,
                    "rationale": "undo via repo-relative path",
                }
            )
        )
        # The success criterion: result is the ReloadResult shape (has 'ok'),
        # not the WriteResult shape — proving rollback dispatched through
        # reload_atom rather than degrading to plain writer.write.
        assert result_rel.is_error is False, result_rel.extras
        assert "ok" in result_rel.extras and result_rel.extras["ok"] is True
        # And the live session uses the reloaded code on the next turn.
        assert _tool_result_text((await session.prompt("after rel"))[-2]) == "v1"

        _api().reload_atom(
            "tool_demo",
            _tool_source("tool_demo", "v2"),
            rationale="re-upgrade",
        )
        assert _tool_result_text((await session.prompt("between"))[-2]) == "v2"
        abs_path = str((tmp_path / rel_path).resolve())
        result_abs = _as_tool_result(
            await rollback_tool.execute(
                {
                    "path": abs_path,
                    "target_sha": first_sha,
                    "rationale": "undo via absolute path",
                }
            )
        )
        assert result_abs.is_error is False, result_abs.extras
        assert "ok" in result_abs.extras and result_abs.extras["ok"] is True
        assert _tool_result_text((await session.prompt("after abs"))[-2]) == "v1"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_G9_batch_write_still_coalesces_into_one_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repo(tmp_path)
    _write_manifest(tmp_path)
    left = tmp_path / "skills" / "foo" / "SKILL.md"
    right = tmp_path / "prompts" / "note.txt"
    left.parent.mkdir(parents=True, exist_ok=True)
    right.parent.mkdir(parents=True, exist_ok=True)
    left.write_text("left before\n", encoding="utf-8")
    right.write_text("right before\n", encoding="utf-8")
    _git(tmp_path, "add", "skills/foo/SKILL.md", "prompts/note.txt")
    _git(tmp_path, "commit", "-m", "seed managed files", "--quiet")
    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="batch-session",
        bus=EventBus(),
    )
    before = _git(tmp_path, "rev-parse", "HEAD")
    async with writer.batch(rationale="batch update") as batch:
        await batch.write("skills/foo/SKILL.md", b"left after\n")
        await batch.write("prompts/note.txt", b"right after\n")

    history = _git(tmp_path, "log", "--format=%an|%s", "-n", "2").splitlines()
    changed = set(
        _git(tmp_path, "show", "--name-only", "--format=", "HEAD").splitlines()
    )
    assert _git(tmp_path, "rev-parse", "HEAD") != before
    assert history[0] == "agent|batch update"
    assert left.read_text(encoding="utf-8") == "left after\n"
    assert right.read_text(encoding="utf-8") == "right after\n"
    assert changed == {"prompts/note.txt", "skills/foo/SKILL.md"}
