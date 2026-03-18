"""Tests for SkillMiddleware (vault-tools-based skill injection)."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agentm.middleware.skill import SkillMiddleware
from agentm.tools.vault.store import MarkdownVault


@pytest.fixture
def vault(tmp_path):
    """Create a MarkdownVault with two skill notes."""
    v = MarkdownVault(tmp_path)
    v.write(
        "skill/duckdb-query",
        {
            "type": "skill",
            "confidence": "fact",
            "name": "DuckDB Query Guide",
            "description": "Best practices for querying DuckDB in RCA workflows.",
        },
        "# DuckDB Query Guide\n\nUse `SELECT * FROM ...` with caution.",
    )
    v.write(
        "skill/log-analysis",
        {
            "type": "skill",
            "confidence": "fact",
            "name": "Log Analysis",
            "description": "How to parse and analyze application logs.",
        },
        "# Log Analysis\n\nAlways filter by timestamp first.",
    )
    return v


# ---------------------------------------------------------------------------
# SkillMiddleware
# ---------------------------------------------------------------------------


class TestSkillMiddlewareNoSkills:
    def test_no_skills_passthrough(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, skill_paths=[])
        state = {"messages": [SystemMessage(content="You are an agent.")]}
        result = mw.before_model(state)
        assert result is None  # no change

    def test_skill_count_zero(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, skill_paths=[])
        assert mw.skill_count == 0


class TestSkillMiddlewareInjection:
    def test_descriptions_injected_into_system_message(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, ["skill/duckdb-query", "skill/log-analysis"])
        state = {"messages": [SystemMessage(content="You are an agent.")]}
        result = mw.before_model(state)

        assert result is not None
        messages = result["messages"]
        assert len(messages) == 1
        sys_msg = messages[0]
        assert isinstance(sys_msg, SystemMessage)
        content = str(sys_msg.content)
        assert "<skills>" in content
        assert "DuckDB Query Guide" in content
        assert "Log Analysis" in content
        assert 'path="skill/duckdb-query"' in content
        assert "vault_read" in content  # usage instruction

    def test_vault_search_mentioned(self, vault: MarkdownVault):
        """Skills section guides agents to use vault_search for discovery."""
        mw = SkillMiddleware(vault, ["skill/duckdb-query"])
        state = {"messages": [SystemMessage(content="System prompt")]}
        result = mw.before_model(state)
        content = str(result["messages"][0].content)
        assert "vault_search" in content

    def test_vault_list_mentioned(self, vault: MarkdownVault):
        """Skills section guides agents to use vault_list for browsing."""
        mw = SkillMiddleware(vault, ["skill/duckdb-query"])
        state = {"messages": [SystemMessage(content="System prompt")]}
        result = mw.before_model(state)
        content = str(result["messages"][0].content)
        assert "vault_list" in content

    def test_original_system_content_preserved(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, ["skill/duckdb-query"])
        original = "You are an orchestrator. Think step by step."
        state = {"messages": [SystemMessage(content=original)]}
        result = mw.before_model(state)

        content = str(result["messages"][0].content)
        assert content.startswith(original)

    def test_non_system_messages_untouched(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, ["skill/duckdb-query"])
        human_msg = HumanMessage(content="Hello")
        state = {"messages": [SystemMessage(content="System"), human_msg]}
        result = mw.before_model(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[1] is human_msg  # same object, not modified

    def test_llm_input_messages_key_used_when_present(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, ["skill/duckdb-query"])
        state = {
            "llm_input_messages": [SystemMessage(content="Compressed")],
            "messages": [SystemMessage(content="Full history")],
        }
        result = mw.before_model(state)
        assert "llm_input_messages" in result
        content = str(result["llm_input_messages"][0].content)
        assert "Compressed" in content
        assert "DuckDB Query Guide" in content


class TestSkillMiddlewareMissing:
    def test_missing_skill_gracefully_skipped(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, ["skill/nonexistent", "skill/duckdb-query"])
        assert mw.skill_count == 1  # only duckdb loaded

        state = {"messages": [SystemMessage(content="Agent prompt")]}
        result = mw.before_model(state)
        content = str(result["messages"][0].content)
        assert "DuckDB Query Guide" in content
        assert "nonexistent" not in content

    def test_all_missing_skills_passthrough(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, ["skill/nope", "skill/also-nope"])
        assert mw.skill_count == 0
        result = mw.before_model({"messages": [SystemMessage(content="Hi")]})
        assert result is None


class TestSkillMiddlewareIndependence:
    def test_per_agent_independence(self, vault: MarkdownVault):
        mw_orch = SkillMiddleware(vault, ["skill/duckdb-query"])
        mw_worker = SkillMiddleware(vault, ["skill/log-analysis"])

        state = {"messages": [SystemMessage(content="Base prompt")]}

        result_orch = mw_orch.before_model(state)
        result_worker = mw_worker.before_model(state)

        orch_content = str(result_orch["messages"][0].content)
        worker_content = str(result_worker["messages"][0].content)

        assert "DuckDB Query Guide" in orch_content
        assert "Log Analysis" not in orch_content

        assert "Log Analysis" in worker_content
        assert "DuckDB Query Guide" not in worker_content


class TestSkillMiddlewareNoSystemMessage:
    def test_prepends_system_message_when_absent(self, vault: MarkdownVault):
        mw = SkillMiddleware(vault, ["skill/duckdb-query"])
        state = {"messages": [HumanMessage(content="No system msg")]}
        result = mw.before_model(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert "DuckDB Query Guide" in str(messages[0].content)
        assert isinstance(messages[1], HumanMessage)
