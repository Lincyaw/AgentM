"""Tests for knowledge path utility edge cases.

Bug prevented: IndexError on parts[-1] when path is empty or root-only.
"""

from __future__ import annotations

import pytest

from agentm.tools.knowledge import path_to_namespace_and_key


class TestPathToNamespaceAndKeyEdgeCases:
    """Empty or root-only paths should raise ValueError, not IndexError."""

    def test_root_slash_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one segment"):
            path_to_namespace_and_key("/")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one segment"):
            path_to_namespace_and_key("")
