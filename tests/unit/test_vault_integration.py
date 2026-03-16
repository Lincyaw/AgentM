"""Integration tests for the vault public API and builder wiring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ------------------------------------------------------------------
# Public API surface
# ------------------------------------------------------------------


class TestVaultPublicAPI:
    """Verify the public API is importable and complete."""

    def test_import_markdown_vault(self) -> None:
        from agentm.tools.vault import MarkdownVault

        assert MarkdownVault is not None

    def test_import_create_vault_tools(self) -> None:
        from agentm.tools.vault import create_vault_tools

        assert callable(create_vault_tools)

    def test_all_exports(self) -> None:
        import agentm.tools.vault as vault_mod

        assert set(vault_mod.__all__) == {
            "MarkdownVault",
            "create_vault_tools",
        }

    def test_create_vault_tools_returns_expected_keys(self, tmp_path: Path) -> None:
        from agentm.tools.vault import MarkdownVault, create_vault_tools

        vault = MarkdownVault(tmp_path / "vault")
        tools = create_vault_tools(vault)

        expected_tools = {
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
        assert set(tools.keys()) == expected_tools
        for name, func in tools.items():
            assert callable(func), f"{name} is not callable"


# ------------------------------------------------------------------
# Vault round-trip through tool functions
# ------------------------------------------------------------------


class TestVaultToolRoundTrip:
    """Verify vault tools work end-to-end via the closure API."""

    def test_write_then_read(self, tmp_path: Path) -> None:
        from agentm.tools.vault import MarkdownVault, create_vault_tools

        vault = MarkdownVault(tmp_path / "vault")
        tools = create_vault_tools(vault)

        result = tools["vault_write"](
            entries=[{
                "path": "test/note",
                "frontmatter": {"type": "insight", "tags": ["demo"]},
                "body": "# Hello\n\nThis is a test note.",
            }],
        )
        data = json.loads(result)
        assert data["status"] == "ok"

        result = tools["vault_read"](path="test/note")
        data = json.loads(result)
        assert data["path"] == "test/note"
        assert data["frontmatter"]["type"] == "insight"
        assert "Hello" in data["body"]

    def test_search_after_write(self, tmp_path: Path) -> None:
        from agentm.tools.vault import MarkdownVault, create_vault_tools

        vault = MarkdownVault(tmp_path / "vault")
        tools = create_vault_tools(vault)

        tools["vault_write"](
            entries=[{
                "path": "services/api",
                "frontmatter": {"type": "service", "tags": ["api"]},
                "body": "# API Service\n\nHandles REST requests.",
            }],
        )

        result = tools["vault_search"](query="REST requests", mode="keyword")
        data = json.loads(result)
        assert len(data["results"]) > 0
        assert data["results"][0]["path"] == "services/api"
