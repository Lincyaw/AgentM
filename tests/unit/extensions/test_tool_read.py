"""Tests for ``extensions.builtin.file_tools`` read tool — aligned with Claude Code FileReadTool.

Covers: full-file reads, partial reads (offset+limit), max-size gate,
header format, and record_read integration.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from agentm.core.abi.operations import FileOperations
from agentm.core.lib import read_state
from agentm.extensions.builtin import file_tools
from agentm.extensions.builtin.file_tools import FileToolsConfig


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubFileOps(FileOperations):
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    async def read_file(self, path: str) -> bytes:
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    async def write_file(self, path: str, data: bytes) -> None:
        self._files[path] = data

    async def remove_file(self, path: str) -> None:
        self._files.pop(path, None)

    async def exists(self, path: str) -> bool:
        return path in self._files

    async def stat(self, path: str) -> Any:
        raise NotImplementedError

    async def mkdir(self, path: str, *, parents: bool = False, exist_ok: bool = False) -> None:
        raise NotImplementedError

    async def list_dir(self, path: str) -> list[str]:
        raise NotImplementedError


class _FakeWriter:
    """Stub writer — tests here only exercise the read tool."""

    async def read(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    async def write(self, path: str, content: bytes, *, rationale: str = "", author: str = "agent") -> Any:
        raise NotImplementedError

    async def replace(self, path: str, old: bytes, new: bytes, *, rationale: str = "") -> Any:
        raise NotImplementedError


class _Api:
    def __init__(self, cwd: str, file_ops: FileOperations) -> None:
        self.cwd = cwd
        self._file_ops = file_ops
        self.tools: dict[str, Any] = {}

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t

    def get_operations(self) -> Any:
        class _Ops:
            file = self._file_ops
        return _Ops()

    def get_resource_writer(self) -> _FakeWriter:
        return _FakeWriter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install(api: _Api, **config: Any) -> None:
    file_tools.install(api, FileToolsConfig(**config))  # type: ignore[arg-type]


def _read(api: _Api, path: str, **kwargs: Any) -> tuple[str, bool]:
    args: dict[str, Any] = {"path": path, **kwargs}
    result = asyncio.run(api.tools["read"].fn(args))
    return result.content[0].text, bool(result.is_error)


def _make_lines(n: int) -> bytes:
    """Build a file with *n* numbered lines."""
    return "\n".join(f"line-{i}" for i in range(1, n + 1)).encode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_read_state():
    read_state.clear()
    yield
    read_state.clear()


class TestFullRead:
    """Default read (no offset/limit) returns ALL lines."""

    def test_reads_all_lines(self) -> None:
        content = _make_lines(50)
        api = _Api("/tmp", _StubFileOps({"/tmp/a.py": content}))
        _install(api)

        text, is_err = _read(api, "/tmp/a.py")
        assert not is_err
        lines = text.split("\n")
        assert lines[0] == "(50 lines total)"
        # All 50 lines present
        assert len(lines) == 51  # header + 50 data lines

    def test_no_2000_line_cap(self) -> None:
        """Verify there is no hardcoded 2000-line limit."""
        content = _make_lines(3000)
        api = _Api("/tmp", _StubFileOps({"/tmp/big.py": content}))
        _install(api, max_size_bytes=10_000_000)  # allow large file

        text, is_err = _read(api, "/tmp/big.py")
        assert not is_err
        lines = text.split("\n")
        assert lines[0] == "(3000 lines total)"
        assert len(lines) == 3001  # header + 3000

    def test_header_format_full(self) -> None:
        api = _Api("/tmp", _StubFileOps({"/tmp/x.txt": b"hello\nworld"}))
        _install(api)

        text, _ = _read(api, "/tmp/x.txt")
        assert text.startswith("(2 lines total)")

    def test_line_number_format(self) -> None:
        api = _Api("/tmp", _StubFileOps({"/tmp/x.txt": b"aaa\nbbb\nccc"}))
        _install(api)

        text, _ = _read(api, "/tmp/x.txt")
        lines = text.split("\n")
        assert lines[1] == "1\taaa"
        assert lines[2] == "2\tbbb"
        assert lines[3] == "3\tccc"


class TestPartialRead:
    """Reads with offset and/or limit."""

    def test_offset_and_limit(self) -> None:
        content = _make_lines(10)
        api = _Api("/tmp", _StubFileOps({"/tmp/a.py": content}))
        _install(api)

        text, is_err = _read(api, "/tmp/a.py", offset=3, limit=2)
        assert not is_err
        lines = text.split("\n")
        # offset=3 → 1-based line 3, so 0-based index 2
        assert lines[0] == "(showing lines 3-4 of 10)"
        assert lines[1] == "3\tline-3"
        assert lines[2] == "4\tline-4"
        assert len(lines) == 3  # header + 2 data lines

    def test_offset_only(self) -> None:
        content = _make_lines(5)
        api = _Api("/tmp", _StubFileOps({"/tmp/a.py": content}))
        _install(api)

        text, is_err = _read(api, "/tmp/a.py", offset=4)
        assert not is_err
        lines = text.split("\n")
        # offset=4 → start from line 4, read to end (lines 4,5)
        assert lines[0] == "(showing lines 4-5 of 5)"
        assert lines[1] == "4\tline-4"
        assert lines[2] == "5\tline-5"

    def test_limit_only(self) -> None:
        content = _make_lines(10)
        api = _Api("/tmp", _StubFileOps({"/tmp/a.py": content}))
        _install(api)

        text, is_err = _read(api, "/tmp/a.py", limit=3)
        assert not is_err
        lines = text.split("\n")
        assert lines[0] == "(showing lines 1-3 of 10)"
        assert len(lines) == 4  # header + 3

    def test_header_format_partial(self) -> None:
        content = _make_lines(20)
        api = _Api("/tmp", _StubFileOps({"/tmp/f.py": content}))
        _install(api)

        text, _ = _read(api, "/tmp/f.py", offset=5, limit=3)
        assert text.startswith("(showing lines 5-7 of 20)")


class TestMaxSizeGate:
    """Files exceeding max_size_bytes are rejected unless offset/limit given."""

    def test_large_file_rejected(self) -> None:
        big_data = b"x" * 300_000  # > 262144
        api = _Api("/tmp", _StubFileOps({"/tmp/huge.txt": big_data}))
        _install(api)

        text, is_err = _read(api, "/tmp/huge.txt")
        assert is_err
        assert "exceeds maximum" in text
        assert "offset and limit" in text

    def test_large_file_allowed_with_offset_limit(self) -> None:
        # Even though the file is big, providing offset+limit bypasses the gate.
        lines = _make_lines(100)
        big_data = lines + b"\0" * 300_000  # pad to exceed default
        api = _Api("/tmp", _StubFileOps({"/tmp/huge.txt": big_data}))
        _install(api)

        text, is_err = _read(api, "/tmp/huge.txt", offset=1, limit=5)
        assert not is_err
        assert "showing lines" in text

    def test_custom_max_size(self) -> None:
        data = b"hello\nworld\n"  # 12 bytes
        api = _Api("/tmp", _StubFileOps({"/tmp/small.txt": data}))
        _install(api, max_size_bytes=10)  # set cap to 10 bytes

        text, is_err = _read(api, "/tmp/small.txt")
        assert is_err
        assert "exceeds maximum" in text

    def test_file_within_default_size_ok(self) -> None:
        data = b"hello\nworld\n"  # well under 256KB
        api = _Api("/tmp", _StubFileOps({"/tmp/ok.txt": data}))
        _install(api)

        text, is_err = _read(api, "/tmp/ok.txt")
        assert not is_err


class TestRecordRead:
    """Verify record_read is called with correct arguments."""

    def test_full_read_records_not_partial(self) -> None:
        api = _Api("/tmp", _StubFileOps({"/tmp/a.py": b"one\ntwo\nthree"}))
        _install(api)

        _read(api, "/tmp/a.py")

        state = read_state.get_read_state("/tmp/a.py")
        assert state is not None
        assert state.total_lines == 3
        assert state.is_partial is False

    def test_partial_read_records_partial(self) -> None:
        api = _Api("/tmp", _StubFileOps({"/tmp/a.py": _make_lines(10)}))
        _install(api)

        _read(api, "/tmp/a.py", offset=1, limit=3)

        state = read_state.get_read_state("/tmp/a.py")
        assert state is not None
        assert state.total_lines == 10
        assert state.is_partial is True

    def test_offset_at_end_not_partial(self) -> None:
        """Reading from offset that covers all remaining lines is not partial."""
        api = _Api("/tmp", _StubFileOps({"/tmp/a.py": _make_lines(5)}))
        _install(api)

        # offset=1 with no limit reads everything
        _read(api, "/tmp/a.py")

        state = read_state.get_read_state("/tmp/a.py")
        assert state is not None
        assert state.is_partial is False

    def test_not_called_on_size_gate_error(self) -> None:
        big_data = b"x" * 300_000
        api = _Api("/tmp", _StubFileOps({"/tmp/huge.txt": big_data}))
        _install(api)

        _read(api, "/tmp/huge.txt")

        state = read_state.get_read_state("/tmp/huge.txt")
        assert state is None


class TestConfigSchema:
    """max_size_bytes appears in the config schema."""

    def test_max_size_bytes_in_schema(self) -> None:
        schema = file_tools.MANIFEST.config_schema
        assert schema is not None
        props = schema.model_json_schema()["properties"]
        assert "max_size_bytes" in props
        assert props["max_size_bytes"]["type"] == "integer"
        assert props["max_size_bytes"]["default"] == 262_144


class TestParameterSchema:
    """offset and limit have no defaults in the schema."""

    def test_no_default_on_offset(self) -> None:
        props = file_tools._READ_PARAMETERS["properties"]
        assert "default" not in props["offset"]

    def test_no_default_on_limit(self) -> None:
        props = file_tools._READ_PARAMETERS["properties"]
        assert "default" not in props["limit"]
