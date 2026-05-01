"""Tests for catalog hashing helpers."""

from __future__ import annotations

from agentm.core.catalog import (
    compute_active_set_fingerprint,
    compute_atom_hash,
)



def test_hash_is_deterministic() -> None:
    source = "def install(api, config):\n    return None\n"

    assert compute_atom_hash(source) == compute_atom_hash(source)



def test_hash_distinguishes_whitespace_difference() -> None:
    left = "def install(api, config):\n    return None\n"
    right = "def install(api, config):\n     return None\n"

    assert compute_atom_hash(left) != compute_atom_hash(right)



def test_active_set_fingerprint_includes_all_loaded_atoms() -> None:
    fingerprint = compute_active_set_fingerprint(
        {"tool_read": "abc123def456", "tool_bash": "fed654cba321"},
        scenario="general_purpose@aaaabbbbcccc",
        core_hash="111122223333",
    )

    assert fingerprint["core"] == "core@111122223333"
    assert fingerprint["scenario"] == "general_purpose@aaaabbbbcccc"
    assert fingerprint["atoms"] == {
        "tool_bash": "tool_bash@fed654cba321",
        "tool_read": "tool_read@abc123def456",
    }



def test_active_set_fingerprint_optional_core_and_scenario() -> None:
    fingerprint = compute_active_set_fingerprint(
        {"tool_read": "abc123def456"},
        scenario=None,
        core_hash=None,
    )

    assert fingerprint == {
        "core": None,
        "scenario": None,
        "atoms": {"tool_read": "tool_read@abc123def456"},
    }



