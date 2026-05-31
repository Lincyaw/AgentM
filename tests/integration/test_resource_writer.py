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
from agentm.core.abi.extension import ProviderConfig
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.resource_writer import GitBackedResourceWriter
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


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
    # Use a non-protected initial branch so the writer's protected-branch
    # guard does not block legacy tests. Tests that exercise the guard
    # explicitly check out `main` / `master` themselves.
    _git(root, "init", "-q", "-b", "agent-tests")
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
            extensions=[
                ("agentm.extensions.builtin.operations_local", {}),
                *extensions,
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )


@pytest.mark.asyncio
async def test_G3_constitution_rejects_tool_edit(tmp_path: Path) -> None:
    """G3: constitution edits are rejected before fs or git mutation."""

    _init_repo(tmp_path)
    target = tmp_path / "core-manifest.yaml"
    target.write_text(
        "version: 1\n"
        "constitution:\n"
        "  paths:\n"
        "    - core-manifest.yaml\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", "core-manifest.yaml")
    _git(tmp_path, "commit", "-m", "add constitution file", "--quiet")
    before = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    session = await _create_session(tmp_path, ("agentm.extensions.builtin.tool_edit", {}))
    try:
        # Simulate a prior read so the read-before-edit gate passes and
        # the constitution guard is the one that rejects the edit.
        from agentm.core.lib.read_state import record_read
        record_read(os.path.normpath("core-manifest.yaml"), total_lines=4, is_partial=False)
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
    assert "version: 2" not in target.read_text(encoding="utf-8")
    assert after == before










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




def _seed_managed_skill(repo: Path) -> Path:
    skill_path = repo / "skills" / "foo" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "skills/foo/SKILL.md")
    _git(repo, "commit", "-m", "seed skill", "--quiet")
    return skill_path


@pytest.mark.asyncio
async def test_protected_branch_refuses_managed_write_on_main(tmp_path: Path) -> None:
    """Auto-commit must not land on `main` in the user's real repo."""

    _init_repo(tmp_path)
    skill_path = _seed_managed_skill(tmp_path)
    _git(tmp_path, "checkout", "-q", "-B", "main")
    before_sha = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    before_bytes = skill_path.read_bytes()

    writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="protected-test",
        bus=EventBus(),
    )
    result = await writer.write(
        "skills/foo/SKILL.md",
        b"hostile rewrite\n",
        rationale="should be refused",
    )

    assert result.committed is False
    assert result.error is not None
    assert "protected branch" in result.error
    assert "main" in result.error
    assert skill_path.read_bytes() == before_bytes
    assert _git(tmp_path, "rev-parse", "HEAD").stdout.strip() == before_sha







