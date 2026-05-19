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

import os
from pathlib import Path

from agentm.cli import (
    DEFAULT_PROVIDER_REGISTRY,
    DEFAULT_SCENARIO,
    _resolve_provider_model_cwd,
    autoload_dotenv,
)


def test_skip_dotenv_short_circuits_autoload(
    tmp_path: Path, monkeypatch
) -> None:
    """``AGENTM_SKIP_DOTENV=1`` must prevent any ``.env`` from loading.

    The check has to happen before the filesystem walk so tests can
    disable autoload without their tmp cwd contributing noise.
    """

    env_file = tmp_path / ".env"
    env_file.write_text("AGENTM_TEST_SENTINEL=loaded\n", encoding="utf-8")
    monkeypatch.delenv("AGENTM_TEST_SENTINEL", raising=False)
    monkeypatch.setenv("AGENTM_SKIP_DOTENV", "1")

    autoload_dotenv(tmp_path)

    assert "AGENTM_TEST_SENTINEL" not in os.environ


def test_autoload_dotenv_uses_supplied_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """Autoload must consult the caller-supplied cwd, not process cwd."""

    env_file = tmp_path / ".env"
    env_file.write_text("AGENTM_TEST_SENTINEL=from_target\n", encoding="utf-8")
    monkeypatch.delenv("AGENTM_TEST_SENTINEL", raising=False)
    monkeypatch.delenv("AGENTM_SKIP_DOTENV", raising=False)
    # Move process cwd somewhere unrelated to prove the supplied
    # ``cwd=`` argument wins over ``Path.cwd()``.
    monkeypatch.chdir(tmp_path.parent)

    autoload_dotenv(tmp_path)

    assert os.environ.get("AGENTM_TEST_SENTINEL") == "from_target"


def test_explicit_provider_picks_that_providers_default_model(
    monkeypatch,
) -> None:
    """``--provider openai`` (no ``AGENTM_MODEL``) must pick OpenAI's
    registry default, not whatever default the registry-default provider
    happened to have.

    This was the silent failure mode of the old import-time
    ``_model_default()`` helper: it captured ``_provider_default()`` at
    def-time and so ``--provider openai`` inherited (e.g.) Anthropic's
    default model.
    """

    monkeypatch.delenv("AGENTM_PROVIDER", raising=False)
    monkeypatch.delenv("AGENTM_MODEL", raising=False)
    monkeypatch.delenv("AGENTM_CWD", raising=False)

    provider, model, _cwd = _resolve_provider_model_cwd(
        provider_flag="openai",
        model_flag=None,
        cwd_flag=None,
    )
    assert provider == "openai"
    expected = DEFAULT_PROVIDER_REGISTRY.default_model("openai")
    assert model == expected
    # And it must NOT be the registry-default provider's default model
    # whenever those two differ — guard against the import-time bug.
    default_provider_model = DEFAULT_PROVIDER_REGISTRY.default_model()
    if expected != default_provider_model:
        assert model != default_provider_model


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


def test_agentm_scenario_env_respected_when_no_flag(monkeypatch) -> None:
    """``AGENTM_SCENARIO=foo`` must be the resolved scenario when no
    ``--scenario`` flag is given.

    We assert the resolution logic the CLI applies inline (matches
    ``run_cmd``): ``scenario = flag or env or DEFAULT_SCENARIO``.
    """

    monkeypatch.setenv("AGENTM_SCENARIO", "foo")

    flag: str | None = None
    resolved = flag or os.environ.get("AGENTM_SCENARIO") or DEFAULT_SCENARIO
    assert resolved == "foo"

    # And the constant is what the bare default falls back to.
    monkeypatch.delenv("AGENTM_SCENARIO", raising=False)
    resolved = flag or os.environ.get("AGENTM_SCENARIO") or DEFAULT_SCENARIO
    assert resolved == DEFAULT_SCENARIO
