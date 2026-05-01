from __future__ import annotations

from pathlib import Path

from agentm.ai.env_api_keys import find_env_keys, get_env_api_key


def test_anthropic_prefers_oauth_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-api")
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "sk-oauth")

    assert find_env_keys("anthropic") == [
        "ANTHROPIC_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ]
    assert get_env_api_key("anthropic") == "sk-oauth"


def test_vertex_adc_fallback_reports_authenticated(tmp_path: Path, monkeypatch) -> None:
    creds = tmp_path / "adc.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("GOOGLE_CLOUD_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")

    assert get_env_api_key("google-vertex") == "<authenticated>"
