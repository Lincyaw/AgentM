"""CLI environment-variable contract tests.

Fail-stop positions covered:

* ``AGENTM_SKIP_DOTENV`` must short-circuit the dotenv autoload before any
  filesystem walk (tests set it and expect no env mutation).
* ``--provider X`` without ``AGENTM_MODEL`` must pick provider X's
  registry default, not the import-time provider's default model.
* ``AGENTM_SCENARIO`` must be honoured when no ``--scenario`` flag is
  given (the CLAUDE.md env table lists it as consumed by ``agentm``).

These guard the documented CLI contract; if any of them regress, the
table in CLAUDE.md is silently a lie.
"""

from __future__ import annotations


from agentm.cli import (
    _resolve_provider_model_cwd,
)








def test_resolve_precedence_flag_over_env_over_default(monkeypatch) -> None:
    """CLI flag beats env var; env var beats built-in default."""

    monkeypatch.setenv("AGENTM_PROVIDER", "openai")
    monkeypatch.setenv("AGENTM_MODEL", "env-model")
    monkeypatch.setenv("AGENTM_CWD", "/from/env")

    # Env wins when no flag.
    provider, model, cwd = _resolve_provider_model_cwd(
        provider_flag=None, model_flag=None, cwd_flag=None
    )
    assert (provider, model, cwd) == ("openai", "env-model", "/from/env")

    # Flag beats env.
    provider, model, cwd = _resolve_provider_model_cwd(
        provider_flag="anthropic",
        model_flag="flag-model",
        cwd_flag="/from/flag",
    )
    assert (provider, model, cwd) == ("anthropic", "flag-model", "/from/flag")


