"""Unit tests for the ``file_tools`` write tool's safety gates.

Verifies the Claude-Code-aligned read-before-write contract:
- New files can be written without prior read.
- Existing files require a prior *full* read.
- Partial reads are rejected with a clear message.
- ``require_read=False`` disables all gates.
- Post-write updates read_state so downstream edits see fresh state.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import pytest

from agentm.core.abi.resource import WriteResult
from agentm.core.lib import read_state as rs_mod
from agentm.core.lib.read_state import get_read_state, record_read, clear
from agentm.extensions.builtin import file_tools


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class _FakeWriteResult:
    """Mirrors the fields WriteResult exposes."""

    path: str = "/fake"
    path_class: str = "unmanaged"
    committed: bool = False
    commit_sha_before: str | None = None
    commit_sha_after: str | None = None
    error: str | None = None


class _FakeWriter:
    """Minimal ResourceWriter shim for unit tests.

    ``files`` maps path -> bytes.  ``read`` raises when the file is absent
    (matching the real writer), ``write`` always succeeds.
    """

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files: dict[str, bytes] = files or {}

    async def read(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str = "",
        author: str = "agent",
    ) -> _FakeWriteResult:
        self.files[path] = content
        return _FakeWriteResult(path=path)


class _FakeFileOps:
    """Minimal FileOperations shim."""

    async def read_file(self, path: str) -> bytes:
        raise FileNotFoundError(path)


class _FakeOperations:
    """Minimal Operations bundle."""

    def __init__(self) -> None:
        self.file = _FakeFileOps()


class _FakeApi:
    """Minimal ExtensionAPI surface for file_tools."""

    def __init__(self, writer: _FakeWriter) -> None:
        self._writer = writer
        self._ops = _FakeOperations()
        self.tools: dict[str, Any] = {}
        self.cwd = "/tmp"

    def get_resource_writer(self) -> _FakeWriter:
        return self._writer

    def get_operations(self) -> _FakeOperations:
        return self._ops

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install(api: _FakeApi, **config: Any) -> None:
    file_tools.install(api, dict(config))  # type: ignore[arg-type]


def _call(api: _FakeApi, path: str, content: str = "hello") -> tuple[str, bool]:
    result = asyncio.run(api.tools["write"].fn({"path": path, "content": content}))
    return result.content[0].text, bool(result.is_error)


@pytest.fixture(autouse=True)
def _clean_read_state():
    """Reset module-level read state between tests."""
    clear()
    yield
    clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_new_file_succeeds_without_prior_read() -> None:
    """New files (path not in writer.files) should be writable freely."""
    writer = _FakeWriter()
    api = _FakeApi(writer)
    _install(api)

    text, is_err = _call(api, "/tmp/new.txt", "new content")
    assert not is_err
    assert "Created" in text
    assert "bytes" in text


def test_overwrite_existing_without_prior_read_errors() -> None:
    """Existing file + no prior read -> error."""
    writer = _FakeWriter({"/tmp/existing.txt": b"old"})
    api = _FakeApi(writer)
    _install(api)

    text, is_err = _call(api, "/tmp/existing.txt", "overwrite")
    assert is_err
    assert "Read it first" in text


def test_overwrite_existing_with_partial_read_errors() -> None:
    """Existing file + partial read -> error with clear message."""
    writer = _FakeWriter({"/tmp/partial.txt": b"line1\nline2\nline3"})
    api = _FakeApi(writer)
    _install(api)

    # Simulate a partial read (offset/limit).
    record_read(os.path.normpath("/tmp/partial.txt"), total_lines=3, is_partial=True)

    text, is_err = _call(api, "/tmp/partial.txt", "overwrite")
    assert is_err
    assert "partial" in text.lower()


def test_overwrite_existing_with_full_read_succeeds() -> None:
    """Existing file + full read -> write should succeed."""
    writer = _FakeWriter({"/tmp/full.txt": b"original"})
    api = _FakeApi(writer)
    _install(api)

    record_read(os.path.normpath("/tmp/full.txt"), total_lines=1, is_partial=False)

    text, is_err = _call(api, "/tmp/full.txt", "new content")
    assert not is_err
    assert "Updated" in text
    assert "bytes" in text


def test_require_read_false_disables_all_gates() -> None:
    """With require_read=False, existing files can be overwritten freely."""
    writer = _FakeWriter({"/tmp/no-gate.txt": b"existing content"})
    api = _FakeApi(writer)
    _install(api, require_read=False)

    text, is_err = _call(api, "/tmp/no-gate.txt", "overwrite")
    assert not is_err
    assert "Updated" in text


def test_post_write_updates_read_state() -> None:
    """After a successful write, read_state should reflect the new file."""
    writer = _FakeWriter()
    api = _FakeApi(writer)
    _install(api)

    _call(api, "/tmp/post-write.txt", "line1\nline2\n")

    normalized = os.path.normpath("/tmp/post-write.txt")
    rs = get_read_state(normalized)
    assert rs is not None
    assert rs.is_partial is False
    # "line1\nline2\n" has 2 newlines + non-empty -> total_lines = 3
    assert rs.total_lines == 3


def test_output_reports_byte_count() -> None:
    """Output message should include the byte count of the written content."""
    writer = _FakeWriter()
    api = _FakeApi(writer)
    _install(api)

    content = "hello world"
    text, is_err = _call(api, "/tmp/bytes.txt", content)
    assert not is_err
    assert str(len(content.encode("utf-8"))) in text
