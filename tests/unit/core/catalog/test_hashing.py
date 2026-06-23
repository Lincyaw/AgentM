"""Atom hash determinism and active-set fingerprint pairing."""

from __future__ import annotations

import re

from agentm.core._internal.catalog.hashing import (
    compute_active_set_fingerprint,
    compute_atom_hash,
)


# ---------------------------------------------------------------------------
# compute_atom_hash
# ---------------------------------------------------------------------------


def test_atom_hash_is_idempotent() -> None:
    source = "def install(api, config): ..."
    assert compute_atom_hash(source) == compute_atom_hash(source)


def test_atom_hash_differs_on_single_char_change() -> None:
    assert compute_atom_hash("hello") != compute_atom_hash("hellp")


def test_atom_hash_empty_string() -> None:
    result = compute_atom_hash("")
    assert isinstance(result, str)
    assert len(result) == 12


def test_atom_hash_output_format() -> None:
    for source in ("hello", "", "def f(): pass", "\n\t\0"):
        result = compute_atom_hash(source)
        assert re.fullmatch(r"[0-9a-f]{12}", result), f"bad format for {source!r}: {result}"


def test_atom_hash_content_only() -> None:
    source = "identical content"
    first = compute_atom_hash(source)
    second = compute_atom_hash(source)
    assert first == second


# ---------------------------------------------------------------------------
# compute_active_set_fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic() -> None:
    loaded = {"alpha": "aaa111", "beta": "bbb222"}
    a = compute_active_set_fingerprint(loaded, scenario="sc", core_hash="ch")
    b = compute_active_set_fingerprint(loaded, scenario="sc", core_hash="ch")
    assert a == b


def test_fingerprint_ignores_dict_insertion_order() -> None:
    a = compute_active_set_fingerprint(
        {"a": "h1", "b": "h2"}, scenario="s", core_hash="c"
    )
    b = compute_active_set_fingerprint(
        {"b": "h2", "a": "h1"}, scenario="s", core_hash="c"
    )
    assert a == b


def test_fingerprint_changes_when_atom_added() -> None:
    base = {"x": "hash_x", "y": "hash_y"}
    extended = {**base, "z": "hash_z"}
    fp_base = compute_active_set_fingerprint(base, scenario="s", core_hash="c")
    fp_ext = compute_active_set_fingerprint(extended, scenario="s", core_hash="c")
    assert fp_base != fp_ext


def test_fingerprint_changes_when_scenario_changes() -> None:
    loaded = {"a": "h1"}
    fp_a = compute_active_set_fingerprint(loaded, scenario="alpha", core_hash="c")
    fp_b = compute_active_set_fingerprint(loaded, scenario="beta", core_hash="c")
    assert fp_a != fp_b


def test_fingerprint_none_values() -> None:
    fp = compute_active_set_fingerprint({"a": "h1"}, scenario=None, core_hash=None)
    assert fp["core"] is None
    assert fp["scenario"] is None
    assert "atoms" in fp
    assert isinstance(fp["atoms"], dict)


def test_fingerprint_atom_value_format() -> None:
    fp = compute_active_set_fingerprint(
        {"tool_read": "abc123", "tool_bash": "def456"},
        scenario="local",
        core_hash="corehash",
    )
    for name, value in fp["atoms"].items():
        assert re.fullmatch(rf"{re.escape(name)}@[a-zA-Z0-9_]+", value), (
            f"atom value {value!r} does not match name@hash format"
        )


def test_fingerprint_structure() -> None:
    fp = compute_active_set_fingerprint({"a": "h1"}, scenario="s", core_hash="c")
    assert set(fp.keys()) == {"core", "scenario", "atoms"}
