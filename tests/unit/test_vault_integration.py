"""High-value integration checks for vault public API wiring."""

from __future__ import annotations

import json
from pathlib import Path


def test_public_api_and_factory_shape_are_stable(tmp_path: Path) -> None:
    import agentm.tools.vault as vault_mod
    from agentm.tools.vault import MarkdownVault, create_vault_tools

    assert set(vault_mod.__all__) == {"MarkdownVault", "create_vault_tools"}

    tools = create_vault_tools(MarkdownVault(tmp_path / "vault"))
    assert set(tools) == {
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
    assert all(callable(fn) for fn in tools.values())


def test_tool_round_trip_write_read_search(tmp_path: Path) -> None:
    from agentm.tools.vault import MarkdownVault, create_vault_tools

    tools = create_vault_tools(MarkdownVault(tmp_path / "vault"))

    write_result = json.loads(
        tools["vault_write"](
            entries=[
                {
                    "path": "services/api",
                    "frontmatter": {"type": "service", "tags": ["api"]},
                    "body": "# API Service\n\nHandles REST requests.",
                }
            ]
        )
    )
    assert write_result["status"] == "ok"

    read_result = json.loads(tools["vault_read"](path="services/api"))
    assert read_result["path"] == "services/api"
    assert "REST requests" in read_result["body"]

    search_result = json.loads(tools["vault_search"](query="REST requests", mode="keyword"))
    assert [r["path"] for r in search_result["results"]] == ["services/api"]
