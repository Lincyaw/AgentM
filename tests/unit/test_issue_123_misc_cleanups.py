"""Regression tests for issue #123 misc tail cleanups.

Locks down five small cleanups so that subsequent refactors cannot silently
re-introduce the original drift:

* C13 — ``tool_bash`` default timeout is sourced from a single module
  constant (no parallel literal in schema vs runtime).
* C15 — ``inherit_provider`` exports ``PARENT_PROVIDER_CONFIG_KEY`` and the
  validator flags atoms that read the config via the bare ``"provider"``
  literal alongside the constant import.
* D5 — ``_dynamic_import_target`` recognises one level of ``f"agentm.{...}"``
  template and routes it through the import allow-list.
* E10 — registering two non-canonical OpenAI-compatible providers without an
  explicit ``name`` raises ``DuplicateProviderError``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from agentm.extensions.builtin import inherit_provider, tool_bash
from agentm.extensions.validate import (
    _dynamic_import_target,
    _imports_parent_provider_config_key,
)
from agentm.extensions.builtin.llm_openai import DuplicateProviderError


# --- C13 -------------------------------------------------------------------


def test_tool_bash_default_timeout_single_source_of_truth() -> None:
    """Schema default and runtime fallback must read the same constant."""

    schema_default = (
        tool_bash.MANIFEST.config_schema["properties"]["default_timeout"]["default"]
    )
    assert schema_default == tool_bash._DEFAULT_TIMEOUT_SECONDS

    # The install() runtime fallback reads from the same constant: scan the
    # source for a literal ``120`` / ``120.0`` to ensure it has been removed.
    src = Path(tool_bash.__file__).read_text(encoding="utf-8")
    install_index = src.index("def install(")
    install_body = src[install_index:]
    assert "120.0" not in install_body
    assert "120)" not in install_body


# --- C15 -------------------------------------------------------------------


def test_inherit_provider_exports_constant_and_uses_it_in_schema() -> None:
    assert inherit_provider.PARENT_PROVIDER_CONFIG_KEY == "provider"
    schema = inherit_provider.MANIFEST.config_schema
    # Schema is built via the constant — ensure required[] mentions exactly
    # one canonical key.
    assert schema["required"] == [inherit_provider.PARENT_PROVIDER_CONFIG_KEY]
    assert inherit_provider.PARENT_PROVIDER_CONFIG_KEY in schema["properties"]


def test_validator_flags_bare_provider_literal_in_inherit_provider_consumer() -> None:
    """An atom that imports PARENT_PROVIDER_CONFIG_KEY but also uses the
    bare \"provider\" string literal trips the §11.4.D7 rule."""

    from agentm.extensions.validate import _check_ast_rules

    src = (
        "from agentm.extensions.builtin.inherit_provider import "
        "PARENT_PROVIDER_CONFIG_KEY\n"
        "def install(api, config):\n"
        "    config['provider']  # noqa\n"
    )
    bad_file = Path(__file__).parent / "_issue_123_bad_consumer.py"
    bad_file.write_text(src, encoding="utf-8")
    try:
        issues = _check_ast_rules("test.bad_consumer", bad_file)
    finally:
        bad_file.unlink(missing_ok=True)

    assert any(
        issue.rule == "11.4.D7-inherit-provider-bare-literal"
        for issue in issues
    ), issues


def test_validator_does_not_flag_atoms_that_skip_the_constant() -> None:
    """Atoms that never import PARENT_PROVIDER_CONFIG_KEY are out of scope —
    rule must not fire on incidental ``"provider"`` strings elsewhere in the
    catalog."""

    src = (
        "def install(api, config):\n"
        "    return 'provider'  # unrelated literal\n"
    )
    tree = ast.parse(src)
    assert _imports_parent_provider_config_key(tree) is False


# --- D5 --------------------------------------------------------------------


def test_dynamic_import_target_returns_constant_string() -> None:
    tree = ast.parse('importlib.import_module("agentm.harness.session")')
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert _dynamic_import_target(call) == "agentm.harness.session"


def test_dynamic_import_target_recognises_agentm_fstring_prefix() -> None:
    """f-string with ``agentm.`` constant prefix is treated as a dynamic
    import targeting that namespace, so ``_classify_import`` can apply the
    forbidden-prefix list."""

    tree = ast.parse('importlib.import_module(f"agentm.harness.{name}")')
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    target = _dynamic_import_target(call)
    assert target is not None
    assert target.startswith("agentm.harness")


def test_dynamic_import_target_ignores_non_agentm_fstring() -> None:
    tree = ast.parse('importlib.import_module(f"other.{name}")')
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert _dynamic_import_target(call) is None


# --- E10 -------------------------------------------------------------------


class _StubApi:
    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self.events = None

    def get_service(self, _name: str) -> Any:
        return None

    def register_provider(self, name: str, config: Any) -> None:
        self._providers[name] = config

    def has_provider(self, name: str) -> bool:
        return name in self._providers


def test_openai_install_rejects_default_name_for_non_canonical_base_url() -> None:
    """Two custom endpoints would both default to ``"openai"`` — refuse."""

    from agentm.extensions.builtin import llm_openai as openai_mod

    api = _StubApi()
    with pytest.raises(DuplicateProviderError):
        openai_mod.install(
            api,
            {
                "model": "doubao-seed",
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            },
        )


def test_openai_install_accepts_explicit_name_for_non_canonical_base_url() -> None:
    from agentm.extensions.builtin import llm_openai as openai_mod

    api = _StubApi()
    openai_mod.install(
        api,
        {
            "model": "doubao-seed",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "name": "doubao",
        },
    )
    assert "doubao" in api._providers


def test_openai_install_detects_duplicate_name_in_session() -> None:
    from agentm.extensions.builtin import llm_openai as openai_mod

    api = _StubApi()
    openai_mod.install(
        api,
        {
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
        },
    )
    with pytest.raises(DuplicateProviderError):
        openai_mod.install(
            api,
            {
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
            },
        )


