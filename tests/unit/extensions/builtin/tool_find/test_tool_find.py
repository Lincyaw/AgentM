from __future__ import annotations

from pathlib import Path

import pytest

from agentm.extensions.builtin import tool_find
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class FakeFindOps:
    def __init__(self) -> None:
        self.used = False

    async def exists(self, path: str) -> bool:
        return True

    async def glob(self, pattern: str, cwd: str, *, ignore: list[str], limit: int) -> list[str]:
        del pattern, ignore, limit
        self.used = True
        return [f"{cwd}/b/two.py", "a/one.py", "dir/"]


async def _session(tmp_path: Path, config: dict) -> AgentSession:
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_find", config)],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )


@pytest.mark.asyncio
async def test_find_fallback_returns_posix_paths_and_excludes_special_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_find.shutil, "which", lambda name: None)
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "pkg" / "sub" / "b.py").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.py").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "skip.py").write_text("x", encoding="utf-8")
    session = await _session(tmp_path, {})

    result = await session.tools[0].execute({"pattern": "**/*.py", "limit": 20})

    assert result.content[0].text.splitlines() == ["pkg/a.py", "pkg/sub/b.py"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_find_uses_custom_operations(tmp_path: Path) -> None:
    ops = FakeFindOps()
    session = await _session(tmp_path, {"ops": ops})

    result = await session.tools[0].execute({"pattern": "*.py"})

    assert ops.used is True
    assert result.content[0].text.splitlines() == ["a/one.py", "b/two.py", "dir/"]
    await session.shutdown()
