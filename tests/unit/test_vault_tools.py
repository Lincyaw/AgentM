"""Focused regression tests for vault tool closures."""

from __future__ import annotations

import json

import pytest

from agentm.tools.vault.store import MarkdownVault
from agentm.tools.vault.tools import create_vault_tools


@pytest.fixture
def vault(tmp_path):
    return MarkdownVault(tmp_path)


@pytest.fixture
def tools(vault):
    return create_vault_tools(vault)


def test_factory_returns_expected_tool_set_and_callables(tools) -> None:
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
    assert set(tools) == expected
    assert all(callable(fn) for fn in tools.values())


def test_vault_write_and_read_round_trip(tools) -> None:
    write_result = json.loads(
        tools["vault_write"](
            entries=[
                {
                    "path": "skill/timeout",
                    "frontmatter": {"type": "skill", "confidence": "fact"},
                    "body": "# Timeout\n\nBody",
                }
            ]
        )
    )
    assert write_result["status"] == "ok"

    read_result = json.loads(tools["vault_read"](path="skill/timeout"))
    assert read_result["path"] == "skill/timeout"
    assert read_result["frontmatter"]["type"] == "skill"


def test_missing_read_and_bad_edit_are_wrapped_as_json_errors(tools) -> None:
    missing = json.loads(tools["vault_read"](path="missing/path"))
    assert "error" in missing
    assert "Note not found" in missing["error"]

    tools["vault_write"](
        entries=[{"path": "e", "frontmatter": {}, "body": "# E\nBody"}]
    )
    bad_edit = json.loads(tools["vault_edit"](path="e", operation="bad_op", params={}))
    assert "error" in bad_edit


def test_search_modes_cover_keyword_fallback_and_unknown_mode_error(tools) -> None:
    tools["vault_write"](
        entries=[
            {
                "path": "s",
                "frontmatter": {"type": "skill", "confidence": "fact", "tags": ["db"]},
                "body": "# Search\n\nDatabase timeout troubleshooting",
            }
        ]
    )

    hybrid = json.loads(tools["vault_search"](query="timeout", mode="hybrid"))
    assert hybrid["results"]
    assert hybrid["results"][0]["path"] == "s"

    unknown = json.loads(tools["vault_search"](query="timeout", mode="bogus"))
    assert "error" in unknown


def test_backlinks_traverse_and_lint_capture_graph_state(tools) -> None:
    tools["vault_write"](
        entries=[
            {"path": "a", "frontmatter": {}, "body": "# A\n[[b]] [[missing]]"},
            {"path": "b", "frontmatter": {}, "body": "# B"},
        ]
    )

    backlinks = json.loads(tools["vault_backlinks"](path="b"))
    assert backlinks["backlinks"] == ["a"]

    traverse = json.loads(tools["vault_traverse"](start="a", depth=1, direction="forward"))
    assert {n["path"] for n in traverse["nodes"]} == {"a", "b", "missing"}

    lint = json.loads(tools["vault_lint"](request="check"))
    assert ["a", "missing"] in lint["dead_links"]


def test_all_tools_return_json_strings(tools) -> None:
    tools["vault_write"](
        entries=[
            {"path": "x", "frontmatter": {"type": "skill", "confidence": "fact"}, "body": "# X\n[[y]]"},
            {"path": "y", "frontmatter": {}, "body": "# Y"},
        ]
    )

    calls = [
        ("vault_write", {"entries": [{"path": "z", "frontmatter": {}, "body": "# Z"}]}),
        ("vault_read", {"path": "x"}),
        ("vault_edit", {"path": "x", "operation": "replace_string", "params": {"old": "# X", "new": "# X2"}}),
        ("vault_list", {"path": ""}),
        ("vault_search", {"query": "X2", "mode": "keyword"}),
        ("vault_backlinks", {"path": "y"}),
        ("vault_traverse", {"start": "x"}),
        ("vault_lint", {"request": "check"}),
        ("vault_rename", {"old_path": "z", "new_path": "z2"}),
        ("vault_delete", {"path": "z2"}),
    ]

    for name, kwargs in calls:
        raw = tools[name](**kwargs)
        assert isinstance(raw, str)
        assert isinstance(json.loads(raw), dict)
