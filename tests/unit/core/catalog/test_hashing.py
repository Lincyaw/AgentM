"""Tests for catalog hashing helpers and public surface shape."""

from __future__ import annotations

import ast
from pathlib import Path

from agentm.core import catalog
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



def test_hash_length_matches_design_examples() -> None:
    content_hash = compute_atom_hash("print('agentm')\n")

    assert len(content_hash) == 12
    assert all(ch in "0123456789abcdef" for ch in content_hash)



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



def test_public_surface_all_matches_contract() -> None:
    # PR #42 froze the catalog-storage surface; PR #43 (tool-catalog-atom)
    # additively extended it with the read-API helpers
    # (CatalogAtom/list_versions/get_manifest_at/runs_for). PR #46
    # (transactional-reload) additively extended it with
    # source_path_for_hash so the harness reload pipeline can locate the
    # frozen source for a known content hash. All are part of the agreed
    # Wave-2 contract.
    assert catalog.__all__ == [
        "CatalogAtom",
        "compute_active_set_fingerprint",
        "compute_atom_hash",
        "freeze_current",
        "get_manifest_at",
        "is_constitution_path",
        "list_atoms",
        "list_versions",
        "runs_for",
        "source_path_for_hash",
    ]



def test_catalog_modules_keep_layer_purity_contract() -> None:
    allowed_import_roots = {
        "os",
        "re",
        "hashlib",
        "yaml",
        "pathlib",
        "functools",
        "dataclasses",
        "datetime",
        "typing",
        # PR #44 indexer additions: stdlib only.
        "argparse",
        "inspect",
        "json",
        "logging",
        "agentm.core.catalog",
        "agentm.extensions",
        "__future__",
    }
    catalog_dir = Path("src/agentm/core/catalog")

    for path in sorted(catalog_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                names = [node.module]
            else:
                continue

            for name in names:
                assert not name.startswith("agentm.harness"), (
                    f"{path} illegally imports {name}"
                )
                assert not name.startswith("agentm.extensions.builtin"), (
                    f"{path} illegally imports {name}"
                )
                root = name if name in allowed_import_roots else name.rsplit(".", 1)[0]
                assert root in allowed_import_roots, (
                    f"{path} imports {name}, which is outside the allowed layer set"
                )
