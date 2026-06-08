"""Unit tests for the glob tool (merged into file_tools).

Tests exercise the async tool execution path via a minimal fake API with
a local BashOperations backend, matching the patterns used by existing
builtin tool tests.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from agentm.core.abi.operations import ExecResult
from agentm.extensions.builtin import file_tools


# ---------------------------------------------------------------------------
# Minimal fake ExtensionAPI — just enough for install()
# ---------------------------------------------------------------------------

class _LocalBashOps:
    """Runs commands locally via subprocess — suitable for tmp_path tests."""

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_data: Any = None,
        signal: Any = None,
    ) -> ExecResult:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                timeout=timeout,
                cwd=cwd,
            )
            return ExecResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(stdout=b"", stderr=b"", exit_code=-1, timed_out=True)


class _FakeOperations:
    def __init__(self) -> None:
        self.bash = _LocalBashOps()


class _FakeApi:
    """Covers the ExtensionAPI surface that file_tools.install touches."""

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.tools: dict[str, Any] = {}
        self._ops = _FakeOperations()

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t

    def get_operations(self) -> _FakeOperations:
        return self._ops

    def get_resource_writer(self) -> Any:
        raise NotImplementedError("glob tests don't need writer")


def _install(api: _FakeApi) -> None:
    file_tools.install(api, {"require_read": False})  # type: ignore[arg-type]


def _run(api: _FakeApi, **kwargs: Any) -> tuple[str, bool]:
    result = asyncio.run(api.tools["glob"].fn(kwargs))
    return result.content[0].text, bool(result.is_error)


# ---------------------------------------------------------------------------
# End-to-end tool execution via fake API + real filesystem
# ---------------------------------------------------------------------------

def test_tool_install_registers_glob(tmp_path: Path) -> None:
    api = _FakeApi(str(tmp_path))
    _install(api)
    assert "glob" in api.tools


def test_tool_execute_basic(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("x")

    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*.py")
    assert not is_err
    assert "a.py" in text
    assert "b.txt" not in text


def test_tool_execute_no_matches(tmp_path: Path) -> None:
    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*.xyz")
    assert not is_err
    assert text == "No files found"


def test_tool_execute_bad_directory(tmp_path: Path) -> None:
    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*", path="/nonexistent/dir")
    assert is_err
    assert "does not exist" in text


def test_tool_execute_truncation_message(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x")

    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*.py", limit=3)
    assert not is_err
    assert "truncated" in text.lower()


def test_tool_execute_custom_path(tmp_path: Path) -> None:
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "lib.py").write_text("x")
    (tmp_path / "top.py").write_text("x")

    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*.py", path=str(sub))
    assert not is_err
    assert "lib.py" in text
    assert "top.py" not in text


def test_skip_git_directory(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git" / "objects"
    git_dir.mkdir(parents=True)
    (git_dir / "pack.idx").write_text("x")
    (tmp_path / "main.py").write_text("x")

    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*")
    assert not is_err
    assert "main.py" in text
    assert ".git" not in text


def test_skip_node_modules(tmp_path: Path) -> None:
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("x")
    (tmp_path / "app.js").write_text("x")

    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*.js")
    assert not is_err
    assert "app.js" in text
    assert "node_modules" not in text


def test_skip_pycache(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_text("x")
    (tmp_path / "mod.py").write_text("x")

    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*")
    assert not is_err
    assert "mod.py" in text
    assert "__pycache__" not in text


def test_relative_path_output(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "c.txt").write_text("x")

    api = _FakeApi(str(tmp_path))
    _install(api)

    text, is_err = _run(api, pattern="*.txt")
    assert not is_err
    for line in text.splitlines():
        assert not line.startswith("/"), f"Expected relative path, got {line!r}"
