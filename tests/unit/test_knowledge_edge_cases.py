"""Edge-case tests for knowledge store operations.

Bug prevented: Operations on boundary inputs (special characters, unicode,
very long paths) cause unexpected crashes or data corruption.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.tools import knowledge as knowledge_module
from agentm.tools.knowledge import (
    _path_to_fs,
    knowledge_read,
    knowledge_write,
)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path: Path):
    knowledge_module.init(base_dir=str(tmp_path / "knowledge"))
    yield tmp_path / "knowledge"
    knowledge_module._base_dir = None
    knowledge_module._entries = {}
    knowledge_module._inv_index = {}
    knowledge_module._embeddings = {}


class TestPathEdgeCases:
    """Path parsing edge cases."""

    def test_path_with_hyphens(self) -> None:
        cat, slug = _path_to_fs("/failure-patterns/db-connection-exhaustion")
        assert cat == "failure-patterns"
        assert slug == "db-connection-exhaustion"

    def test_path_with_underscores(self) -> None:
        cat, slug = _path_to_fs("/failure_patterns/db_pool")
        assert cat == "failure_patterns"
        assert slug == "db_pool"


class TestWriteEdgeCases:
    """Write operations with unusual inputs."""

    def test_write_entry_with_unicode(self) -> None:
        """Unicode content round-trips correctly."""
        entry = {"title": "数据库连接池耗尽", "description": "高负载下连接池达到上限"}
        knowledge_write("/failure-patterns/unicode-test", entry)
        result = json.loads(knowledge_read("/failure-patterns/unicode-test"))
        assert result["title"] == "数据库连接池耗尽"

    def test_write_minimal_entry(self) -> None:
        """An entry with no standard fields still writes and reads."""
        knowledge_write("/misc/minimal", {"custom_key": 42})
        result = json.loads(knowledge_read("/misc/minimal"))
        assert result["custom_key"] == 42

    def test_write_single_segment_path_raises(self) -> None:
        """Paths with only one segment are rejected."""
        with pytest.raises(ValueError, match="at least two segments"):
            knowledge_write("/single", {"title": "bad"})
