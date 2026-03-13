"""Tests for Knowledge Store path validation.

Bug prevented: Invalid paths (single-segment, empty) cause silent failures
or crash with wrong exception types. These tests verify _path_to_fs
rejects bad paths with clear ValueError messages.
"""

from __future__ import annotations

import pytest

from agentm.tools.knowledge import _path_to_fs


class TestPathToFs:
    """Verify _path_to_fs splits paths into (category, slug)."""

    def test_two_segments(self) -> None:
        assert _path_to_fs("/failure-patterns/pool") == ("failure-patterns", "pool")

    def test_deep_path(self) -> None:
        """Deeper paths use all intermediate segments as category."""
        assert _path_to_fs("/a/b/c") == ("a/b", "c")

    def test_trailing_slash_stripped(self) -> None:
        assert _path_to_fs("/failure-patterns/pool/") == ("failure-patterns", "pool")

    def test_leading_slash_stripped(self) -> None:
        assert _path_to_fs("failure-patterns/pool") == ("failure-patterns", "pool")


class TestPathToFsEdgeCases:
    """Invalid paths should raise ValueError, not IndexError."""

    def test_root_slash_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="at least two segments"):
            _path_to_fs("/")

    def test_empty_string_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="at least two segments"):
            _path_to_fs("")

    def test_single_segment_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="at least two segments"):
            _path_to_fs("/only-one")
