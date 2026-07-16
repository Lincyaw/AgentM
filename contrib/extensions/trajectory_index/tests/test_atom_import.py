"""Load-bearing smoke coverage for the trajectory-index atom surface."""

from __future__ import annotations

from trajectory_index.atom import build_extraction_config


def test_atom_imports_and_builds_extraction_config() -> None:
    provider = ("stub", {"model": "stub-model"})
    config = build_extraction_config(
        cwd="/tmp/workspace",
        model="small",
        provider=provider,
        parent_session_id="parent",
    )

    assert config.model == "small"
    assert config.provider == provider
    assert config.purpose == "trajectory_symbol_extractor"
    assert config.lineage["parent_session_id"] == "parent"
