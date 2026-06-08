"""Unit tests for the grep tool (merged into file_tools).

Tests cover command building, output parsing, and end-to-end execution
against real files via ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi.operations import ExecResult
from agentm.extensions.builtin.file_tools import (
    build_grep_or_rg_command,
    build_grep_command,
    build_rg_command,
    parse_grep_output,
    relativize_paths,
)
from agentm.extensions.builtin import file_tools


# ---------------------------------------------------------------------------
# Minimal fake API with local BashOperations
# ---------------------------------------------------------------------------

class _LocalBashOps:
    """Runs commands locally via subprocess."""

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
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.tools: dict[str, Any] = {}
        self._ops = _FakeOperations()

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t

    def get_operations(self) -> _FakeOperations:
        return self._ops

    def get_resource_writer(self) -> Any:
        raise NotImplementedError("grep tests don't need writer")


def _install(api: _FakeApi, **config: Any) -> None:
    file_tools.install(api, {"require_read": False, **config})  # type: ignore[arg-type]


def _run(api: _FakeApi, args: dict[str, Any]) -> tuple[str, bool]:
    result = asyncio.run(api.tools["grep"].fn(args))
    return result.content[0].text, bool(result.is_error)


# ===========================================================================
# Command building
# ===========================================================================

class TestBuildRgCommand:
    def test_basic_pattern(self) -> None:
        cmd = build_rg_command("TODO", "/src")
        assert cmd[0] == "rg"
        assert "TODO" in cmd
        assert "/src" == cmd[-1]
        assert "-n" in cmd  # content mode default

    def test_files_with_matches_mode(self) -> None:
        cmd = build_rg_command("TODO", "/src", output_mode="files_with_matches")
        assert "-l" in cmd
        assert "-n" not in cmd

    def test_count_mode(self) -> None:
        cmd = build_rg_command("TODO", "/src", output_mode="count")
        assert "-c" in cmd
        assert "-n" not in cmd

    def test_case_insensitive(self) -> None:
        cmd = build_rg_command("TODO", "/src", case_insensitive=True)
        assert "-i" in cmd

    def test_glob_filter(self) -> None:
        cmd = build_rg_command("TODO", "/src", glob_filter="*.py")
        glob_args = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--glob"]
        assert "*.py" in glob_args

    def test_context_lines(self) -> None:
        cmd = build_rg_command("TODO", "/src", context_lines=3)
        assert "-C" in cmd
        idx = cmd.index("-C")
        assert cmd[idx + 1] == "3"

    def test_context_lines_ignored_in_count_mode(self) -> None:
        cmd = build_rg_command("TODO", "/src", output_mode="count", context_lines=3)
        assert "-C" not in cmd

    def test_dash_pattern_uses_e_flag(self) -> None:
        cmd = build_rg_command("-foo", "/src")
        idx = cmd.index("-e")
        assert cmd[idx + 1] == "-foo"

    def test_excluded_dirs(self) -> None:
        cmd = build_rg_command("TODO", "/src")
        glob_args = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--glob"]
        assert "!.git" in glob_args
        assert "!node_modules" in glob_args
        assert "!__pycache__" in glob_args


class TestBuildGrepCommand:
    def test_basic_pattern(self) -> None:
        cmd = build_grep_command("TODO", "/src")
        assert cmd[0] == "grep"
        assert "-r" in cmd
        assert "TODO" in cmd
        assert "/src" == cmd[-1]

    def test_files_with_matches_mode(self) -> None:
        cmd = build_grep_command("TODO", "/src", output_mode="files_with_matches")
        assert "-l" in cmd

    def test_count_mode(self) -> None:
        cmd = build_grep_command("TODO", "/src", output_mode="count")
        assert "-c" in cmd

    def test_case_insensitive(self) -> None:
        cmd = build_grep_command("TODO", "/src", case_insensitive=True)
        assert "-i" in cmd

    def test_glob_as_include(self) -> None:
        cmd = build_grep_command("TODO", "/src", glob_filter="*.py")
        idx = cmd.index("--include")
        assert cmd[idx + 1] == "*.py"

    def test_excluded_dirs(self) -> None:
        cmd = build_grep_command("TODO", "/src")
        exclude_args = [
            cmd[i + 1] for i, v in enumerate(cmd) if v == "--exclude-dir"
        ]
        assert ".git" in exclude_args
        assert "node_modules" in exclude_args


class TestBuildCommand:
    def test_selects_rg_when_forced(self) -> None:
        cmd = build_grep_or_rg_command("TODO", "/src", use_ripgrep=True)
        assert cmd[0] == "rg"

    def test_selects_grep_when_forced(self) -> None:
        cmd = build_grep_or_rg_command("TODO", "/src", use_ripgrep=False)
        assert cmd[0] == "grep"


# ===========================================================================
# Output parsing
# ===========================================================================

class TestRelativizePaths:
    def test_strips_prefix(self) -> None:
        lines = ["/home/user/project/foo.py:10:hello"]
        result = relativize_paths(lines, "/home/user/project")
        assert result == ["foo.py:10:hello"]

    def test_no_match_leaves_line_intact(self) -> None:
        lines = ["other/path:1:x"]
        result = relativize_paths(lines, "/home/user/project")
        assert result == ["other/path:1:x"]

    def test_trailing_slash_on_base(self) -> None:
        lines = ["/a/b/c.py:1:x"]
        result = relativize_paths(lines, "/a/b/")
        assert result == ["c.py:1:x"]


class TestParseOutput:
    def test_empty_output(self) -> None:
        result = parse_grep_output("", base_path="/x", output_mode="content", limit=250)
        assert result == "No matches found."

    def test_whitespace_only(self) -> None:
        result = parse_grep_output("   \n  ", base_path="/x", output_mode="content", limit=250)
        assert result == "No matches found."

    def test_content_mode(self) -> None:
        raw = "/x/foo.py:10:hello world\n/x/bar.py:20:hello again"
        result = parse_grep_output(raw, base_path="/x", output_mode="content", limit=250)
        assert "foo.py:10:hello world" in result
        assert "bar.py:20:hello again" in result

    def test_limit_truncation(self) -> None:
        raw = "\n".join(f"/x/f.py:{i}:line{i}" for i in range(10))
        result = parse_grep_output(raw, base_path="/x", output_mode="content", limit=3)
        assert "[Results truncated at 3 lines]" in result

    def test_count_mode_drops_zeros(self) -> None:
        raw = "/x/a.py:5\n/x/b.py:0\n/x/c.py:3"
        result = parse_grep_output(raw, base_path="/x", output_mode="count", limit=250)
        assert "a.py:5" in result
        assert "b.py:0" not in result
        assert "c.py:3" in result


# ===========================================================================
# End-to-end (real files via tmp_path)
# ===========================================================================

class TestEndToEnd:
    """Run the installed tool against actual files in a temp directory."""

    @pytest.fixture()
    def project(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text(
            textwrap.dedent("""\
                # TODO: refactor
                def hello():
                    print("hello world")

                # TODO: add tests
                def add(a, b):
                    return a + b
            """)
        )
        (tmp_path / "src" / "util.py").write_text(
            textwrap.dedent("""\
                import os

                def read_env(key):
                    return os.getenv(key)
            """)
        )
        (tmp_path / "src" / "data.json").write_text('{"key": "value"}')
        return tmp_path

    def test_basic_match(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {"pattern": "TODO", "path": str(project / "src")})
        assert not is_err
        assert "TODO" in text

    def test_case_insensitive(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "todo",
            "path": str(project / "src"),
            "case_insensitive": True,
        })
        assert not is_err
        assert "TODO" in text or "todo" in text.lower()

    def test_glob_filter(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "key",
            "path": str(project / "src"),
            "glob": "*.json",
        })
        assert not is_err
        assert "key" in text

    def test_glob_filter_excludes(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "TODO",
            "path": str(project / "src"),
            "glob": "*.json",
        })
        assert not is_err
        assert "No matches found" in text

    def test_files_with_matches_mode(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "TODO",
            "path": str(project / "src"),
            "output_mode": "files_with_matches",
        })
        assert not is_err
        assert "main.py" in text
        assert "util.py" not in text

    def test_count_mode(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "TODO",
            "path": str(project / "src"),
            "output_mode": "count",
        })
        assert not is_err
        assert "main.py" in text

    def test_result_limiting(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": ".",
            "path": str(project / "src"),
            "limit": 2,
        })
        assert not is_err
        assert "[Results truncated at 2 lines]" in text

    def test_no_matches(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "XYZNONEXISTENT",
            "path": str(project / "src"),
        })
        assert not is_err
        assert "No matches found" in text

    def test_relative_paths_in_output(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "TODO",
            "path": str(project / "src"),
        })
        assert not is_err
        assert str(project / "src") not in text

    def test_nonexistent_path(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "TODO",
            "path": str(project / "nonexistent"),
        })
        assert is_err
        assert "does not exist" in text

    def test_relative_path_resolved_against_cwd(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "TODO",
            "path": "src",
        })
        assert not is_err
        assert "TODO" in text

    def test_default_path_uses_cwd(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {"pattern": "TODO"})
        assert not is_err
        assert "TODO" in text

    def test_context_lines(self, project: Path) -> None:
        api = _FakeApi(str(project))
        _install(api)
        text, is_err = _run(api, {
            "pattern": "hello",
            "path": str(project / "src" / "main.py"),
            "context_lines": 1,
        })
        assert not is_err
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) > 1
