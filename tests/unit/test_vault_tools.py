"""Tests for vault tool functions -- factory, closures, JSON returns, error handling."""

from __future__ import annotations

import json

import pytest

from agentm.tools.vault.store import MarkdownVault


@pytest.fixture
def vault(tmp_path):
    """Create a MarkdownVault in a temporary directory."""
    return MarkdownVault(tmp_path)


@pytest.fixture
def tools(vault):
    """Create vault tools dict from factory."""
    from agentm.tools.vault.tools import create_vault_tools

    return create_vault_tools(vault)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_should_return_dict_with_all_10_tools(self, tools):
        expected = {
            "vault_write",
            "vault_read",
            "vault_edit",
            "vault_delete",
            "vault_rename",
            "vault_list",
            "vault_search",
            "vault_backlinks",
            "vault_traverse",
            "vault_lint",
        }
        assert set(tools.keys()) == expected

    def test_should_return_callables(self, tools):
        for name, fn in tools.items():
            assert callable(fn), f"{name} is not callable"

    def test_each_tool_has_docstring(self, tools):
        for name, fn in tools.items():
            assert fn.__doc__, f"{name} has no docstring"


# ---------------------------------------------------------------------------
# vault_write
# ---------------------------------------------------------------------------


class TestVaultWrite:
    def test_should_write_single_note(self, tools, vault):
        result = json.loads(tools["vault_write"](
            entries=[{"path": "skill/timeout", "frontmatter": {"type": "skill"}, "body": "# Timeout\n\nBody."}]
        ))
        assert result["status"] == "ok"
        note = vault.read("skill/timeout")
        assert note is not None
        assert note["frontmatter"]["type"] == "skill"

    def test_should_write_batch(self, tools, vault):
        entries = [
            {"path": "a", "frontmatter": {"type": "skill"}, "body": "A"},
            {"path": "b", "frontmatter": {"type": "concept"}, "body": "B"},
        ]
        result = json.loads(tools["vault_write"](entries=entries))
        assert result["status"] == "ok"
        assert result["count"] == 2
        assert vault.read("a") is not None
        assert vault.read("b") is not None


# ---------------------------------------------------------------------------
# vault_read
# ---------------------------------------------------------------------------


class TestVaultRead:
    def test_should_read_existing_note(self, tools, vault):
        vault.write("x", {"type": "skill", "tags": ["a"]}, "# X\n\nBody.")
        result = json.loads(tools["vault_read"](path="x"))
        assert result["path"] == "x"
        assert result["frontmatter"]["type"] == "skill"
        assert "# X" in result["body"]

    def test_should_return_error_for_missing_note(self, tools):
        result = json.loads(tools["vault_read"](path="nonexistent"))
        assert "error" in result


# ---------------------------------------------------------------------------
# vault_edit
# ---------------------------------------------------------------------------


class TestVaultEdit:
    def test_should_apply_replace_string(self, tools, vault):
        vault.write("e", {}, "value is 100ms")
        result = json.loads(tools["vault_edit"](
            path="e", operation="replace_string", params={"old": "100ms", "new": "200ms"}
        ))
        assert result["status"] == "ok"
        note = vault.read("e")
        assert "200ms" in note["body"]

    def test_should_apply_set_frontmatter(self, tools, vault):
        vault.write("e", {"type": "skill"}, "body")
        result = json.loads(tools["vault_edit"](
            path="e", operation="set_frontmatter", params={"confidence": "fact"}
        ))
        assert result["status"] == "ok"

    def test_should_return_error_on_missing_note(self, tools):
        result = json.loads(tools["vault_edit"](
            path="missing", operation="replace_string", params={"old": "a", "new": "b"}
        ))
        assert "error" in result

    def test_should_return_error_on_bad_operation(self, tools, vault):
        vault.write("e", {}, "body")
        result = json.loads(tools["vault_edit"](
            path="e", operation="bad_op", params={}
        ))
        assert "error" in result


# ---------------------------------------------------------------------------
# vault_delete
# ---------------------------------------------------------------------------


class TestVaultDelete:
    def test_should_delete_existing_note(self, tools, vault):
        vault.write("d", {}, "body")
        result = json.loads(tools["vault_delete"](path="d"))
        assert result["status"] == "ok"
        assert vault.read("d") is None

    def test_should_return_error_on_missing_note(self, tools):
        result = json.loads(tools["vault_delete"](path="missing"))
        assert "error" in result


# ---------------------------------------------------------------------------
# vault_rename
# ---------------------------------------------------------------------------


class TestVaultRename:
    def test_should_rename_note(self, tools, vault):
        vault.write("old", {}, "body")
        result = json.loads(tools["vault_rename"](old_path="old", new_path="new"))
        assert result["status"] == "ok"
        assert vault.read("old") is None
        assert vault.read("new") is not None

    def test_should_return_error_on_missing_source(self, tools):
        result = json.loads(tools["vault_rename"](old_path="missing", new_path="new"))
        assert "error" in result


# ---------------------------------------------------------------------------
# vault_list
# ---------------------------------------------------------------------------


class TestVaultList:
    def test_should_list_notes(self, tools, vault):
        vault.write("skill/a", {"type": "skill"}, "body")
        vault.write("skill/b", {"type": "skill"}, "body")
        result = json.loads(tools["vault_list"](path="skill"))
        assert "notes" in result
        assert len(result["notes"]) == 2

    def test_should_respect_type_filter(self, tools, vault):
        vault.write("a", {"type": "skill"}, "body")
        vault.write("b", {"type": "concept"}, "body")
        result = json.loads(tools["vault_list"](path="", depth=10, type_filter="skill"))
        paths = [n["path"] for n in result["notes"]]
        assert "a" in paths
        assert "b" not in paths

    def test_should_return_empty_list_for_no_matches(self, tools):
        result = json.loads(tools["vault_list"](path="nonexistent"))
        assert result["notes"] == []


# ---------------------------------------------------------------------------
# vault_search
# ---------------------------------------------------------------------------


class TestVaultSearch:
    def test_should_search_keyword_mode(self, tools, vault):
        vault.write("s", {"type": "skill", "tags": ["db"]}, "# Timeout\n\nDatabase timeout patterns.")
        result = json.loads(tools["vault_search"](query="timeout", mode="keyword"))
        assert "results" in result
        assert len(result["results"]) >= 1
        assert result["results"][0]["path"] == "s"

    def test_should_return_empty_for_no_match(self, tools, vault):
        result = json.loads(tools["vault_search"](query="nonexistent_term_xyz"))
        assert result["results"] == []

    def test_should_return_error_for_empty_query(self, tools):
        result = json.loads(tools["vault_search"](query=""))
        # Empty query returns empty results, not error
        assert result["results"] == []

    def test_should_pass_filters(self, tools, vault):
        vault.write("a", {"type": "skill"}, "# Test\n\nSome searchable content here.")
        vault.write("b", {"type": "concept"}, "# Test\n\nSome searchable content here.")
        result = json.loads(tools["vault_search"](
            query="searchable", filters={"type": "skill"}, mode="keyword"
        ))
        paths = [r["path"] for r in result["results"]]
        assert "a" in paths
        assert "b" not in paths


# ---------------------------------------------------------------------------
# vault_backlinks
# ---------------------------------------------------------------------------


class TestVaultBacklinks:
    def test_should_return_backlinks(self, tools, vault):
        vault.write("target", {}, "body")
        vault.write("ref1", {}, "See [[target]].")
        vault.write("ref2", {}, "Also [[target]].")
        result = json.loads(tools["vault_backlinks"](path="target"))
        assert set(result["backlinks"]) == {"ref1", "ref2"}

    def test_should_return_empty_for_no_backlinks(self, tools, vault):
        vault.write("lonely", {}, "body")
        result = json.loads(tools["vault_backlinks"](path="lonely"))
        assert result["backlinks"] == []


# ---------------------------------------------------------------------------
# vault_traverse
# ---------------------------------------------------------------------------


class TestVaultTraverse:
    def test_should_return_subgraph(self, tools, vault):
        vault.write("a", {}, "[[b]]")
        vault.write("b", {}, "[[c]]")
        vault.write("c", {}, "body")
        result = json.loads(tools["vault_traverse"](start="a", depth=2))
        node_paths = [n["path"] for n in result["nodes"]]
        assert "a" in node_paths
        assert "b" in node_paths

    def test_should_respect_direction(self, tools, vault):
        vault.write("a", {}, "[[b]]")
        vault.write("b", {}, "body")
        result = json.loads(tools["vault_traverse"](start="b", depth=1, direction="backward"))
        node_paths = [n["path"] for n in result["nodes"]]
        assert "a" in node_paths


# ---------------------------------------------------------------------------
# vault_lint
# ---------------------------------------------------------------------------


class TestVaultLint:
    def test_should_detect_dead_links(self, tools, vault):
        vault.write("a", {}, "[[nonexistent]]")
        result = json.loads(tools["vault_lint"](request="check"))
        assert len(result["dead_links"]) >= 1
        assert result["dead_links"][0] == ["a", "nonexistent"]

    def test_should_detect_orphan_notes(self, tools, vault):
        vault.write("orphan", {}, "no links at all")
        result = json.loads(tools["vault_lint"](request="check"))
        assert "orphan" in result["orphan_notes"]

    def test_should_return_clean_for_empty_vault(self, tools):
        result = json.loads(tools["vault_lint"](request="check"))
        assert result["dead_links"] == []
        assert result["orphan_notes"] == []


# ---------------------------------------------------------------------------
# JSON return contract
# ---------------------------------------------------------------------------


class TestJsonContract:
    def test_all_tools_return_str(self, tools, vault):
        """Every tool must return a str that is valid JSON."""
        vault.write("x", {"type": "skill"}, "# X\n\nBody with [[y]].")
        vault.write("y", {}, "body")

        calls = [
            ("vault_write", {"entries": [{"path": "z", "frontmatter": {}, "body": "new"}]}),
            ("vault_read", {"path": "x"}),
            ("vault_edit", {"path": "x", "operation": "replace_string", "params": {"old": "Body", "new": "Updated"}}),
            ("vault_list", {"path": ""}),
            ("vault_search", {"query": "body", "mode": "keyword"}),
            ("vault_backlinks", {"path": "y"}),
            ("vault_traverse", {"start": "x"}),
            ("vault_lint", {"request": "check"}),
            ("vault_rename", {"old_path": "z", "new_path": "z2"}),
            ("vault_delete", {"path": "z2"}),
        ]
        for name, kwargs in calls:
            result = tools[name](**kwargs)
            assert isinstance(result, str), f"{name} did not return str"
            parsed = json.loads(result)
            assert isinstance(parsed, dict), f"{name} did not return JSON object"
