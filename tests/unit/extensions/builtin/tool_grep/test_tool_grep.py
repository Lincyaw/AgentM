from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.extensions.builtin import tool_grep
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class FakeGrepOps:
    def __init__(self, content: str) -> None:
        self.content = content
        self.read_calls: list[str] = []

    async def is_directory(self, path: str) -> bool:
        return False

    async def read_file(self, path: str) -> str:
        self.read_calls.append(path)
        return self.content


class FakeDirectoryGrepOps:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.walk_calls: list[tuple[str, str | None]] = []

    async def is_directory(self, path: str) -> bool:
        return path.endswith("remote-dir")

    async def read_file(self, path: str) -> str:
        return self.files[path]

    async def walk_files(self, root: str, *, glob: str | None = None) -> list[str]:
        self.walk_calls.append((root, glob))
        return sorted(self.files)


class _Stdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class _Stderr:
    async def read(self) -> bytes:
        return b""


class FakeProc:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = _Stdout(lines)
        self.stderr = _Stderr()
        self.returncode = 0
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    async def wait(self) -> int:
        return self.returncode


async def _fake_create_subprocess_exec(*args, **kwargs) -> FakeProc:
    del args, kwargs
    raise AssertionError("test should replace this helper")


async def _session(tmp_path: Path, config: dict) -> AgentSession:
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_grep", config)],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )


@pytest.mark.asyncio
async def test_grep_fallback_respects_gitignore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_grep.shutil, "which", lambda name: None)
    (tmp_path / ".gitignore").write_text("dist/\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "dist").mkdir()
    (tmp_path / "src" / "app.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "dist" / "generated.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    session = await _session(tmp_path, {})

    result = await session.tools[0].execute({"pattern": "def foo", "literal": True})

    assert result.content[0].text.splitlines()[0].startswith("src/app.py:1: def foo()")
    assert "dist/generated.py" not in result.content[0].text
    await session.shutdown()


@pytest.mark.asyncio
async def test_grep_uses_custom_ops_for_file_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_grep.shutil, "which", lambda name: None)
    path = tmp_path / "remote.txt"
    path.write_text("placeholder", encoding="utf-8")
    ops = FakeGrepOps("zero\ndef foo\ntwo\n")
    session = await _session(tmp_path, {"ops": ops})

    result = await session.tools[0].execute({"path": "remote.txt", "pattern": "def foo", "literal": True})

    assert ops.read_calls == [str(path)]
    assert result.content[0].text == "remote.txt:2: def foo"
    await session.shutdown()


@pytest.mark.asyncio
async def test_grep_uses_custom_ops_for_directory_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_grep.shutil, "which", lambda name: "/usr/bin/rg")
    root = tmp_path / "remote-dir"
    alpha = str(root / "pkg" / "alpha.py")
    beta = str(root / "pkg" / "beta.py")
    ops = FakeDirectoryGrepOps(
        {
            alpha: "def foo():\n    pass\n",
            beta: "skip me\n",
        }
    )
    session = await _session(tmp_path, {"ops": ops})

    result = await session.tools[0].execute(
        {"path": "remote-dir", "pattern": "def foo", "literal": True, "glob": "*.py"}
    )

    assert ops.walk_calls == [(str(root), "*.py")]
    assert result.content[0].text == "pkg/alpha.py:1: def foo():"
    await session.shutdown()


@pytest.mark.asyncio
async def test_grep_limit_notice_terminates_rg_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "sample.py"
    target.write_text("\n".join(f"match {index}" for index in range(10)), encoding="utf-8")
    fake_proc = FakeProc(
        [
            json.dumps({"type": "match", "data": {"path": {"text": str(target)}, "line_number": index + 1}}) + "\n"
            for index in range(10)
        ]
    )
    monkeypatch.setattr(tool_grep.shutil, "which", lambda name: "/usr/bin/rg")
    async def _spawn(*args, **kwargs) -> FakeProc:
        del args, kwargs
        return fake_proc

    monkeypatch.setattr(tool_grep.asyncio, "create_subprocess_exec", _spawn)
    session = await _session(tmp_path, {})

    result = await session.tools[0].execute({"pattern": "match", "literal": True, "limit": 5})

    assert fake_proc.terminated is True
    assert "[Truncated: 5 matches limit reached]" in result.content[0].text
    await session.shutdown()


@pytest.mark.asyncio
async def test_grep_missing_path_is_clear(tmp_path: Path) -> None:
    session = await _session(tmp_path, {})

    with pytest.raises(Exception, match="Path not found:"):
        await session.tools[0].execute({"path": "missing.py", "pattern": "x"})

    await session.shutdown()
