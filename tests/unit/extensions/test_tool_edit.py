"""Unit tests for file_tools edit: findActualString, shrinkage guard, mtime gate, post-edit state."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentm.core.lib.read_state import (
    clear as clear_read_state,
    content_hash_for,
    get_read_state,
    record_read,
)

# Import private helpers from file_tools for unit-testing
from agentm.extensions.builtin.file_tools import (
    _check_shrinkage,
    _find_actual_string,
    _strip_line_whitespace,
    _update_read_state_after_edit,
)


# ---------------------------------------------------------------------------
# _find_actual_string tests
# ---------------------------------------------------------------------------


class TestFindActualString:
    def test_exact_match(self) -> None:
        assert _find_actual_string("hello world", "hello") == "hello"

    def test_exact_match_multiline(self) -> None:
        content = "line1\nline2\nline3"
        assert _find_actual_string(content, "line2\nline3") == "line2\nline3"

    def test_not_found(self) -> None:
        assert _find_actual_string("hello world", "missing") is None

    def test_quote_normalized_match(self) -> None:
        # File has curly quotes, search has straight quotes
        content = 'She said “hello”'
        result = _find_actual_string(content, 'She said "hello"')
        assert result is not None
        assert "“" in result  # returns actual string from file

    def test_quote_normalized_single(self) -> None:
        content = "it’s fine"
        result = _find_actual_string(content, "it's fine")
        assert result is not None
        assert "’" in result

    def test_whitespace_trimmed_match(self) -> None:
        # File has trailing whitespace, search does not
        content = "  hello  \n  world  "
        result = _find_actual_string(content, "hello\nworld")
        assert result is not None
        # Result should be the original lines from the file
        assert result == "  hello  \n  world  "

    def test_whitespace_leading_indent_match(self) -> None:
        content = "    def foo():\n        pass"
        result = _find_actual_string(content, "def foo():\npass")
        assert result is not None
        assert result == "    def foo():\n        pass"

    def test_whitespace_no_false_positive(self) -> None:
        """Whitespace trimming should not match across unrelated lines."""
        content = "alpha\nbeta\ngamma"
        assert _find_actual_string(content, "alpha\ngamma") is None

    def test_returns_actual_string_not_search(self) -> None:
        content = 'print(“Hi”)'
        result = _find_actual_string(content, 'print("Hi")')
        assert result is not None
        # The returned value must be the literal characters from the file
        assert result == 'print(“Hi”)'


# ---------------------------------------------------------------------------
# _strip_line_whitespace tests
# ---------------------------------------------------------------------------


class TestStripLineWhitespace:
    def test_strips_both_ends(self) -> None:
        assert _strip_line_whitespace("  hello  ") == "hello"

    def test_multiline(self) -> None:
        assert _strip_line_whitespace("  a  \n  b  ") == "a\nb"

    def test_empty_lines_preserved(self) -> None:
        assert _strip_line_whitespace("a\n\nb") == "a\n\nb"


# ---------------------------------------------------------------------------
# _check_shrinkage tests (existing behavior, keep them)
# ---------------------------------------------------------------------------


class TestCheckShrinkage:
    def test_no_shrinkage(self) -> None:
        original = "line1\nline2\nline3\n"
        updated = "line1\nnew\nline3\n"
        assert _check_shrinkage(original, updated, 5, 3) is None

    def test_acceptable_shrinkage(self) -> None:
        # Deleting 3 lines (within the 5-line threshold)
        original = "a\nb\nc\nd\ne\nf\n"
        # Remove b,c,d => 3 lines lost
        updated = "a\ne\nf\n"
        assert _check_shrinkage(original, updated, 6, 0) is None

    def test_excessive_shrinkage_rejected(self) -> None:
        original = "\n".join(f"line{i}" for i in range(20)) + "\n"
        updated = "line0\n"
        result = _check_shrinkage(original, updated, len("line1"), 0)
        assert result is not None
        assert "Edit rejected" in result


# ---------------------------------------------------------------------------
# _update_read_state_after_edit tests
# ---------------------------------------------------------------------------


class TestUpdateReadStateAfterEdit:
    @pytest.mark.asyncio
    async def test_updates_mtime_and_hash(self) -> None:
        from agentm.extensions.builtin._operations.local import LocalFileOperations

        clear_read_state()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("original content\n")
            path = f.name
        try:
            norm = os.path.normpath(path)
            record_read(norm, total_lines=1, is_partial=False, mtime_ns=1)
            # Simulate an edit by rewriting the file
            with open(path, "w") as f2:
                f2.write("edited content\nline two\n")
            await _update_read_state_after_edit(norm, LocalFileOperations())
            state = get_read_state(norm)
            assert state is not None
            assert state.mtime_ns > 0
            assert state.content_hash != ""
            assert state.total_lines == 3  # "edited content\nline two\n" => 3 (count \n +1)
            with open(path, "rb") as fh:
                expected_hash = content_hash_for(fh.read())
            assert state.content_hash == expected_hash
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_preserves_is_partial(self) -> None:
        from agentm.extensions.builtin._operations.local import LocalFileOperations

        clear_read_state()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("data\n")
            path = f.name
        try:
            norm = os.path.normpath(path)
            record_read(norm, total_lines=1, is_partial=True, mtime_ns=1)
            await _update_read_state_after_edit(norm, LocalFileOperations())
            state = get_read_state(norm)
            assert state is not None
            assert state.is_partial is True
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# file_modified_since_read integration (via file_tools edit flow)
# ---------------------------------------------------------------------------


class TestFileModifiedSinceReadIntegration:
    """Test that file_tools edit's mtime gate works end-to-end with record_read."""

    @pytest.mark.asyncio
    async def test_no_false_positive_after_update(self) -> None:
        """After _update_read_state_after_edit, the file should not appear modified."""
        from agentm.extensions.builtin._operations.local import LocalFileOperations

        clear_read_state()
        from agentm.core.lib.read_state import file_modified_since_read

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("content")
            path = f.name
        try:
            norm = os.path.normpath(path)
            mtime_ns = os.stat(path).st_mtime_ns
            record_read(norm, total_lines=1, is_partial=False, mtime_ns=mtime_ns)
            assert file_modified_since_read(norm) is False

            # Simulate an edit
            time.sleep(0.01)
            with open(path, "w") as f2:
                f2.write("new content")

            # Before update: should detect modification
            assert file_modified_since_read(norm) is True

            # After update: should no longer detect modification
            await _update_read_state_after_edit(norm, LocalFileOperations())
            assert file_modified_since_read(norm) is False
        finally:
            os.unlink(path)
