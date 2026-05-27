"""Atom-config resolution invariants.

Fail-stop position: the precedence and schema-typed coercion of
``resolve_atom_configs`` are the single mechanism by which env vars and CLI
``--set`` overrides reach an atom's ``install(config)``. If precedence flips or
coercion stops reading ``config_schema``, every documented "configure an atom
via env / --set" contract silently produces the wrong-typed or wrong-valued
config, with no other guard between the flag and the atom.
"""

from __future__ import annotations

import pytest

from agentm.core.lib.atom_config import AtomConfigError, resolve_atom_configs

COST_BUDGET = "agentm.extensions.builtin.cost_budget"
PERMISSION = "agentm.extensions.builtin.permission"


def test_manifest_config_passes_through_without_env_or_overrides() -> None:
    out = resolve_atom_configs([(COST_BUDGET, {"limit": 1.0})], env={}, overrides={})
    assert out == [(COST_BUDGET, {"limit": 1.0})]


def test_env_overrides_manifest_and_coerces_to_schema_type() -> None:
    # cost_budget.limit is declared "number" → env string becomes float.
    out = resolve_atom_configs(
        [(COST_BUDGET, {"limit": 1.0})],
        env={"AGENTM_COST_BUDGET_LIMIT": "5"},
        overrides={},
    )
    (_, cfg) = out[0]
    assert cfg["limit"] == 5.0
    assert isinstance(cfg["limit"], float)


def test_override_wins_over_env_and_manifest() -> None:
    out = resolve_atom_configs(
        [(COST_BUDGET, {"limit": 1.0})],
        env={"AGENTM_COST_BUDGET_LIMIT": "5"},
        overrides={"cost_budget": {"limit": "9"}},
    )
    assert out[0][1]["limit"] == 9.0


def test_array_typed_key_coerced_via_json() -> None:
    out = resolve_atom_configs(
        [(PERMISSION, {})],
        env={},
        overrides={"permission": {"deny": '["bash", "write"]'}},
    )
    assert out[0][1]["deny"] == ["bash", "write"]


def test_typed_value_passes_through_uncoerced() -> None:
    # An embedder may hand a typed value directly; it must not be re-parsed.
    out = resolve_atom_configs(
        [(COST_BUDGET, {})],
        env={},
        overrides={"cost_budget": {"limit": 7.5}},
    )
    assert out[0][1]["limit"] == 7.5


def test_undeclared_key_keeps_raw_string() -> None:
    # cost_budget sets additionalProperties: True; an undeclared key still
    # lands, but free text stays text — a numeric-looking value is NOT
    # silently re-typed to int.
    out = resolve_atom_configs(
        [(COST_BUDGET, {"limit": 1.0})],
        env={},
        overrides={"cost_budget": {"currency": "eur", "label": "123"}},
    )
    assert out[0][1]["currency"] == "eur"
    assert out[0][1]["label"] == "123"


def test_malformed_typed_value_raises_with_context() -> None:
    with pytest.raises(AtomConfigError, match="cost_budget.limit"):
        resolve_atom_configs(
            [(COST_BUDGET, {})],
            env={"AGENTM_COST_BUDGET_LIMIT": "not-a-number"},
            overrides={},
        )


def test_non_finite_number_rejected() -> None:
    # A declared `number` must be finite; NaN/inf would silently disable a
    # budget cap (cost < nan is always False) rather than erroring.
    for bad in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(AtomConfigError, match="cost_budget.limit"):
            resolve_atom_configs(
                [(COST_BUDGET, {})],
                env={"AGENTM_COST_BUDGET_LIMIT": bad},
                overrides={},
            )


def test_unimportable_atom_passes_through_untouched() -> None:
    out = resolve_atom_configs(
        [("does.not.exist", {"k": 1})],
        env={"AGENTM_DOES_NOT_EXIST_K": "2"},
        overrides={},
    )
    assert out == [("does.not.exist", {"k": 1})]
