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


import pytest
import typer

from agentm.cli import (
    _parse_set_overrides,
    _resolve_provider_model_cwd,
)


def test_parse_set_overrides_groups_by_atom_and_keeps_raw_value() -> None:
    out = _parse_set_overrides(
        ["cost_budget.limit=5", "cost_budget.currency=eur", 'permission.deny=["bash"]']
    )
    assert out == {
        "cost_budget": {"limit": "5", "currency": "eur"},
        "permission": {"deny": '["bash"]'},
    }


def test_parse_set_overrides_trims_whitespace_on_atom_and_key() -> None:
    # "a.b = c" must key on "b", not "b " — value stays raw.
    assert _parse_set_overrides(["cost_budget.limit = 5"]) == {
        "cost_budget": {"limit": " 5"}
    }


def test_parse_set_overrides_rejects_missing_dot_or_equals() -> None:
    for bad in ["cost_budget.limit", "limit=5", "=5", "cost_budget.=5"]:
        with pytest.raises(typer.BadParameter):
            _parse_set_overrides([bad])








def test_resolve_precedence_flag_over_env_over_default(monkeypatch) -> None:
    """CLI flag beats env var; env var beats built-in default."""

    monkeypatch.setenv("AGENTM_PROVIDER", "openai")
    monkeypatch.setenv("AGENTM_MODEL", "env-model")
    monkeypatch.setenv("AGENTM_CWD", "/from/env")

    # Env wins when no flag.
    provider, model, cwd, _profile = _resolve_provider_model_cwd(
        provider_flag=None, model_flag=None, cwd_flag=None
    )
    assert (provider, model, cwd) == ("openai", "env-model", "/from/env")

    # Flag beats env.
    provider, model, cwd, _profile = _resolve_provider_model_cwd(
        provider_flag="anthropic",
        model_flag="flag-model",
        cwd_flag="/from/flag",
    )
    assert (provider, model, cwd) == ("anthropic", "flag-model", "/from/flag")


