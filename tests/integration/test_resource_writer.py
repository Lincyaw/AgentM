"""Acceptance tests for the git-backed ResourceWriter write chokepoint."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    AssistantStreamEvent,
    EventBus,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.harness.events import ResourceWriteEvent
from agentm.harness.extension import ProviderConfig
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.resource_writer import GitBackedResourceWriter
from agentm.harness.session import AgentSession, AgentSessionConfig


class _StaticProvider:
    def __init__(self, text: str = "ok") -> None:
        self._text = text

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        return self._iter(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=self._text)],
                timestamp=1.0,
                stop_reason="end_turn",
            )
        )

    async def _iter(self, msg: AssistantMessage) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=msg)


def _install_provider_module(name: str, provider: _StaticProvider) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-resource-writer",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-resource-writer",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-resource-writer",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _git(
    cwd: Path,
    *args: str,
    git_dir: Path | None = None,
    work_tree: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = ["git"]
    if git_dir is not None:
        cmd.append(f"--git-dir={git_dir}")
    if work_tree is not None:
        cmd.append(f"--work-tree={work_tree}")
    cmd.extend(args)
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    (root / "README.md").write_text("baseline\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "initial", "--quiet")


async def _create_session(tmp_path: Path, *extensions: tuple[str, dict[str, Any]]) -> AgentSession:
    provider_module = _install_provider_module(
        f"tests.integration._resource_writer_provider_{tmp_path.name}",
        _StaticProvider("ready"),
    )
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=list(extensions),
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )


@pytest.mark.asyncio
async def test_G3_constitution_rejects_tool_edit(tmp_path: Path) -> None:
    """G3: constitution edits are rejected before fs or git mutation."""

    _init_repo(tmp_path)
    target = tmp_path / "core-manifest.yaml"
    target.write_text("version: 1\n", encoding="utf-8")
    _git(tmp_path, "add", "core-manifest.yaml")
    _git(tmp_path, "commit", "-m", "add constitution file", "--quiet")
    before = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    session = await _create_session(tmp_path, ("agentm.extensions.builtin.tool_edit", {}))
    try:
        tool = next(tool for tool in session.tools if tool.name == "edit")
        result = await tool.execute(
            {
                "path": "core-manifest.yaml",
                "old_string": "version: 1",
                "new_string": "version: 2",
            }
        )
    finally:
        await session.shutdown()
        os.chdir(original_cwd)

    after = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    assert result.is_error is True
    message = result.content[0]
    assert isinstance(message, TextContent)
    assert "constitution path" in message.text
    assert target.read_text(encoding="utf-8") == "version: 1\n"
    assert after == before


@pytest.mark.asyncio
async def test_G4_skill_markdown_edit_commits_and_emits_event(tmp_path: Path) -> None:
    """G4: editing a managed SKILL.md lands a git commit and emits ResourceWriteEvent."""

    _init_repo(tmp_path)
    skill_path = tmp_path / "skills" / "foo" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text("hello world\n", encoding="utf-8")
    _git(tmp_path, "add", "skills/foo/SKILL.md")
    _git(tmp_path, "commit", "-m", "add skill", "--quiet")

    session = await _create_session(tmp_path, ("agentm.extensions.builtin.tool_edit", {}))
    seen: list[ResourceWriteEvent] = []
    session.bus.on("resource_write", lambda event: seen.append(event))
    try:
        tool = next(tool for tool in session.tools if tool.name == "edit")
        result = await tool.execute(
            {
                "path": str(skill_path),
                "old_string": "world",
                "new_string": "agentm",
                "rationale": "rewrite skill wording",
            }
        )
    finally:
        await session.shutdown()

    log = _git(
        tmp_path,
        "log",
        "--format=%an|%s",
        "-n",
        "1",
        "--",
        "skills/foo/SKILL.md",
    ).stdout.strip()
    assert result.is_error is False
    assert skill_path.read_text(encoding="utf-8") == "hello agentm\n"
    assert log == "agent|rewrite skill wording"
    assert len(seen) == 1
    assert seen[0].path == "skills/foo/SKILL.md"
    assert seen[0].rationale == "rewrite skill wording"
    assert seen[0].author == "agent"


@pytest.mark.asyncio
async def test_G5_unmanaged_tool_write_passthrough(tmp_path: Path) -> None:
    """G5: writes outside the working tree succeed without a commit."""

    _init_repo(tmp_path)
    before = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    outside = tmp_path.parent / f"{tmp_path.name}-scratch.txt"

    session = await _create_session(tmp_path, ("agentm.extensions.builtin.tool_write", {}))
    try:
        tool = next(tool for tool in session.tools if tool.name == "write")
        result = await tool.execute(
            {
                "path": str(outside),
                "content": "scratch data",
                "rationale": "temporary scratch note",
            }
        )
    finally:
        await session.shutdown()

    after = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    assert result.is_error is False
    assert outside.read_text(encoding="utf-8") == "scratch data"
    assert after == before


@pytest.mark.asyncio
async def test_G7_auto_init_shadow_repo_commits_initial_snapshot_and_write(tmp_path: Path) -> None:
    """G7: absent .git falls back to a shadow bare repo under .agentm/repo."""

    skill_path = tmp_path / "skills" / "foo" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text("before\n", encoding="utf-8")
    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="shadow-session",
        bus=EventBus(),
    )
    result = await writer.write(
        "skills/foo/SKILL.md",
        b"after\n",
        rationale="update skill in shadow repo",
    )

    shadow_git_dir = tmp_path / ".agentm" / "repo"
    messages = _git(
        tmp_path,
        "log",
        "--format=%s",
        git_dir=shadow_git_dir,
        work_tree=tmp_path,
    ).stdout.splitlines()
    assert shadow_git_dir.is_dir()
    assert result.committed is True
    assert result.commit_sha_before is not None
    assert result.commit_sha_after is not None
    assert skill_path.read_text(encoding="utf-8") == "after\n"
    assert messages[:2] == ["update skill in shadow repo", "agentm: initial snapshot"]


@pytest.mark.asyncio
async def test_G8_advisory_mode_when_git_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G8: missing git degrades to advisory mode, writes still succeed, warning once."""

    monkeypatch.setattr("agentm.harness.resource_writer.shutil.which", lambda _: None)
    target = tmp_path / "skills" / "foo" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="advisory-session",
        bus=EventBus(),
    )
    first = await writer.write(
        "skills/foo/SKILL.md",
        b"first advisory write\n",
        rationale="advisory one",
    )
    second = await writer.write(
        "skills/foo/SKILL.md",
        b"second advisory write\n",
        rationale="advisory two",
    )

    warnings = [
        record
        for record in caplog.records
        if "resource writer advisory mode enabled" in record.message
    ]
    assert first.committed is False
    assert second.committed is False
    assert first.error is None
    assert second.error is None
    assert target.read_text(encoding="utf-8") == "second advisory write\n"
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_G10_dirty_human_changes_are_snapshot_before_agent_commit(tmp_path: Path) -> None:
    """G10: a dirty tracked file lands a human snapshot commit before the agent commit."""

    _init_repo(tmp_path)
    skill_path = tmp_path / "skills" / "foo" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", "skills/foo/SKILL.md")
    _git(tmp_path, "commit", "-m", "track skill", "--quiet")
    before = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    skill_path.write_text("human draft\n", encoding="utf-8")

    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="dirty-session",
        bus=EventBus(),
    )
    result = await writer.write(
        "skills/foo/SKILL.md",
        b"agent final\n",
        rationale="agent rewrite",
    )

    history = _git(
        tmp_path,
        "log",
        "--format=%an|%s",
        "-n",
        "3",
        "--",
        "skills/foo/SKILL.md",
    ).stdout.splitlines()
    assert result.committed is True
    assert result.commit_sha_before == before
    assert result.commit_sha_after == _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    assert skill_path.read_text(encoding="utf-8") == "agent final\n"
    assert history[0] == "agent|agent rewrite"
    assert history[1] == "human|auto: pre-agent snapshot"
