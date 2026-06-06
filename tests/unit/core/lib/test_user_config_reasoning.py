"""reasoning_effort / extra_body parse + precedence for user config profiles."""

from __future__ import annotations

from agentm.core.lib.user_config import (
    _parse_profile,
    apply_reasoning_effort,
)


def test_parse_profile_reads_reasoning_effort_and_extra_body() -> None:
    profile = _parse_profile(
        "doubao",
        {
            "provider": "openai",
            "model": "doubao-x",
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    )
    assert profile is not None
    assert profile.reasoning_effort == "high"
    assert profile.extra_body == {"thinking": {"type": "enabled"}}

    build = profile.to_build_config()
    assert build["reasoning_effort"] == "high"
    assert build["extra_body"] == {"thinking": {"type": "enabled"}}


def test_parse_profile_rejects_wrong_types() -> None:
    profile = _parse_profile(
        "x",
        {
            "provider": "openai",
            "model": "m",
            "reasoning_effort": 5,  # not a str
            "extra_body": "nope",  # not a dict
        },
    )
    assert profile is not None
    assert profile.reasoning_effort is None
    assert profile.extra_body is None


def test_apply_reasoning_effort_cli_flag_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENTM_REASONING_EFFORT", "env-value")
    out = apply_reasoning_effort({"reasoning_effort": "profile"}, "cli-value")
    assert out["reasoning_effort"] == "cli-value"


def test_apply_reasoning_effort_env_wins_over_profile(monkeypatch) -> None:
    monkeypatch.setenv("AGENTM_REASONING_EFFORT", "env-value")
    out = apply_reasoning_effort({"reasoning_effort": "profile"}, None)
    assert out["reasoning_effort"] == "env-value"


def test_apply_reasoning_effort_profile_default(monkeypatch) -> None:
    monkeypatch.delenv("AGENTM_REASONING_EFFORT", raising=False)
    out = apply_reasoning_effort({"reasoning_effort": "profile"}, None)
    assert out["reasoning_effort"] == "profile"


def test_apply_reasoning_effort_noop_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("AGENTM_REASONING_EFFORT", raising=False)
    out = apply_reasoning_effort({"model": "m"}, None)
    assert "reasoning_effort" not in out
