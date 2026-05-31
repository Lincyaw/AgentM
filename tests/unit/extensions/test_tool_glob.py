"""Unit tests for the ``tool_glob`` contrib atom.

Tests exercise the core ``_glob_files`` helper and the async tool
execution path via a minimal fake API, matching the patterns used by
existing builtin tool tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agentm.extensions.contrib import tool_glob
from agentm.extensions.contrib.tool_glob import _glob_files


# ---------------------------------------------------------------------------
# Minimal fake ExtensionAPI — just enough for install()
# ---------------------------------------------------------------------------

class _FakeApi:
    """Covers the ExtensionAPI surface that tool_glob.install touches."""

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.tools: dict[str, Any] = {}

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t


def _install(api: _FakeApi) -> None:
    tool_glob.install(api, {})  # type: ignore[arg-type]


def _run(api: _FakeApi, **kwargs: Any) -> tuple[str, bool]:
    result = asyncio.run(api.tools["glob"].fn(kwargs))
    return result.content[0].text, bool(result.is_error)


# ---------------------------------------------------------------------------
# _glob_files unit tests
# ---------------------------------------------------------------------------

def test_basic_star_pattern(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x")
    (tmp_path / "bar.py").write_text("x")
    (tmp_path / "readme.md").write_text("x")

    files, truncated = _glob_files("*.py", str(tmp_path), 100)
    assert not truncated
    assert files == ["bar.py", "foo.py"]


def test_recursive_double_star(tmp_path: Path) -> None:
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (tmp_path / "top.py").write_text("x")
    (sub / "deep.py").write_text("x")

    files, truncated = _glob_files("**/*.py", str(tmp_path), 100)
    assert not truncated
    assert "top.py" in files
    expected_deep = str(Path("pkg") / "sub" / "deep.py")
    assert expected_deep in files


def test_no_matches_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x")

    files, truncated = _glob_files("*.rs", str(tmp_path), 100)
    assert not truncated
    assert files == []


def test_limit_truncation(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"file_{i:02d}.py").write_text("x")

    files, truncated = _glob_files("*.py", str(tmp_path), 5)
    assert truncated
    assert len(files) == 5
    # Results are sorted alphabetically
    assert files == [f"file_{i:02d}.py" for i in range(5)]


def test_relative_path_output(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "c.txt").write_text("x")

    files, truncated = _glob_files("**/*.txt", str(tmp_path), 100)
    assert not truncated
    for f in files:
        assert not f.startswith("/"), f"Expected relative path, got {f!r}"


def test_skip_git_directory(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git" / "objects"
    git_dir.mkdir(parents=True)
    (git_dir / "pack.idx").write_text("x")
    (tmp_path / "main.py").write_text("x")

    files, _ = _glob_files("**/*", str(tmp_path), 100)
    assert "main.py" in files
    assert all(".git" not in f for f in files)


def test_skip_node_modules(tmp_path: Path) -> None:
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("x")
    (tmp_path / "app.js").write_text("x")

    files, _ = _glob_files("**/*.js", str(tmp_path), 100)
    assert "app.js" in files
    assert all("node_modules" not in f for f in files)


def test_skip_pycache(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_text("x")
    (tmp_path / "mod.py").write_text("x")

    files, _ = _glob_files("**/*", str(tmp_path), 100)
    assert "mod.py" in files
    assert all("__pycache__" not in f for f in files)


def test_directories_excluded(tmp_path: Path) -> None:
    """Only regular files appear in results, not directory names."""
    d = tmp_path / "subdir"
    d.mkdir()
    (d / "inner.py").write_text("x")

    files, _ = _glob_files("*", str(tmp_path), 100)
    assert "subdir" not in files


# ---------------------------------------------------------------------------
# End-to-end tool execution via fake API
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
