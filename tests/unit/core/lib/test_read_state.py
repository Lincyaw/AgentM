"""Unit tests for the read-file state module used by read -> edit (file_tools)."""

from __future__ import annotations

import os
import tempfile
import time

from agentm.core.lib.read_state import (
    FileReadState,
    clear,
    content_hash_for,
    file_modified_since_read,
    get_read_state,
    record_read,
)


def test_record_and_get() -> None:
    clear()
    record_read("./src/foo.py", total_lines=100, is_partial=False)
    state = get_read_state("./src/foo.py")
    assert state is not None
    assert state.total_lines == 100
    assert state.is_partial is False


def test_not_read_returns_none() -> None:
    clear()
    assert get_read_state("./unknown.py") is None


def test_path_normalization() -> None:
    clear()
    record_read("./src/../src/foo.py", total_lines=50, is_partial=True)
    state = get_read_state("src/foo.py")
    assert state is not None
    assert state.is_partial is True
    assert state.total_lines == 50


def test_full_read_overwrites_partial() -> None:
    clear()
    record_read("foo.py", total_lines=100, is_partial=True)
    assert get_read_state("foo.py") == FileReadState(total_lines=100, is_partial=True)
    record_read("foo.py", total_lines=100, is_partial=False)
    state = get_read_state("foo.py")
    assert state is not None
    assert state.is_partial is False


def test_clear_removes_all_state() -> None:
    record_read("a.py", total_lines=10, is_partial=False)
    record_read("b.py", total_lines=20, is_partial=True)
    clear()
    assert get_read_state("a.py") is None
    assert get_read_state("b.py") is None


# ---- New tests for mtime / content_hash tracking ----


def test_backward_compat_defaults() -> None:
    """record_read without mtime_ns/content_hash still works (defaults to 0/'')."""
    clear()
    record_read("compat.py", total_lines=5, is_partial=False)
    state = get_read_state("compat.py")
    assert state is not None
    assert state.mtime_ns == 0
    assert state.content_hash == ""


def test_mtime_and_hash_recorded() -> None:
    clear()
    record_read(
        "tracked.py",
        total_lines=42,
        is_partial=False,
        mtime_ns=1_000_000_000,
        content_hash="abc123",
    )
    state = get_read_state("tracked.py")
    assert state is not None
    assert state.mtime_ns == 1_000_000_000
    assert state.content_hash == "abc123"


def test_content_hash_for() -> None:
    import hashlib

    data = b"hello world\n"
    expected = hashlib.sha256(data).hexdigest()
    assert content_hash_for(data) == expected


def test_file_modified_since_read_not_modified() -> None:
    """file_modified_since_read returns False when file hasn't changed."""
    clear()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("original")
        f.flush()
        path = f.name
    try:
        mtime_ns = os.stat(path).st_mtime_ns
        record_read(path, total_lines=1, is_partial=False, mtime_ns=mtime_ns)
        assert file_modified_since_read(path) is False
    finally:
        os.unlink(path)


def test_file_modified_since_read_modified() -> None:
    """file_modified_since_read returns True when file is touched after read."""
    clear()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("original")
        f.flush()
        path = f.name
    try:
        mtime_ns = os.stat(path).st_mtime_ns
        record_read(path, total_lines=1, is_partial=False, mtime_ns=mtime_ns)
        # Bump mtime by writing again
        time.sleep(0.01)
        with open(path, "w") as f2:
            f2.write("changed")
        assert file_modified_since_read(path) is True
    finally:
        os.unlink(path)


def test_file_modified_since_read_no_prior_read() -> None:
    """file_modified_since_read returns False when file was never read."""
    clear()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("data")
        path = f.name
    try:
        assert file_modified_since_read(path) is False
    finally:
        os.unlink(path)


def test_file_modified_since_read_no_mtime() -> None:
    """file_modified_since_read returns False when read was recorded without mtime."""
    clear()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("data")
        path = f.name
    try:
        record_read(path, total_lines=1, is_partial=False)
        assert file_modified_since_read(path) is False
    finally:
        os.unlink(path)


def test_file_modified_since_read_deleted_file() -> None:
    """file_modified_since_read returns False when file no longer exists."""
    clear()
    record_read(
        "/tmp/nonexistent_agentm_test.py",
        total_lines=1,
        is_partial=False,
        mtime_ns=1_000_000,
    )
    assert file_modified_since_read("/tmp/nonexistent_agentm_test.py") is False
