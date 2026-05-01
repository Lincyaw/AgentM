"""Tests for env-api-keys discovery."""

from __future__ import annotations

import pytest

from agentm.ai import find_env_keys, get_env_api_key


def test_anthropic_oauth_token_takes_precedence_over_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "oauth-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-value")

    assert get_env_api_key("anthropic") == "oauth-value"
    assert find_env_keys("anthropic") == ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]


def test_falls_back_to_api_key_when_oauth_token_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fallback-key")

    assert get_env_api_key("anthropic") == "fallback-key"


def test_unknown_provider_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_env_api_key("not-a-provider") is None
    assert find_env_keys("not-a-provider") is None


def test_returns_none_when_no_env_vars_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY",):
        monkeypatch.delenv(var, raising=False)

    assert get_env_api_key("openai") is None
    assert find_env_keys("openai") is None


def test_github_copilot_checks_three_token_vars_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "gh-fallback")

    assert get_env_api_key("github-copilot") == "gh-fallback"
    assert find_env_keys("github-copilot") == ["GITHUB_TOKEN"]
