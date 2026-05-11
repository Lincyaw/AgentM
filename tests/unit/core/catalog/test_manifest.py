"""Tests for the constitution boundary manifest parser and predicate.

Drives the public surface of `agentm.core._internal.catalog.manifest`:

    - load_core_manifest() -> CoreManifest
    - reload_manifest()    -> CoreManifest   (clears cache; for tests)
    - is_constitution_path(path: str) -> bool
    - CoreManifest dataclass

These tests intentionally fail until the module is implemented; the file
itself does not exist yet — that is the RED phase of TDD.

Convention: the loader resolves the manifest path from
:data:`_MANIFEST_PATH_VAR` (a :class:`ContextVar`). Tests redirect the
loader by entering :func:`override_manifest_path`; the cache is busted
automatically on enter and the binding is restored on exit. Production
code uses :func:`configure_manifest_path` once at session startup.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agentm.core._internal.catalog import manifest as manifest_mod
from agentm.core._internal.catalog.manifest import (
    CoreManifest,
    is_constitution_path,
    load_core_manifest,
    reload_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KERNEL_PATH = "src/agentm/core/abi/loop.py"
EXTENSION_ATOM_PATH = "src/agentm/extensions/builtin/tool_read.py"
CATALOG_METRICS_PATH = ".agentm/catalog/atoms/x/y/metrics.jsonl"
MANIFEST_FILENAME = "core-manifest.yaml"


@pytest.fixture(autouse=True)
def _reset_manifest_cache() -> Iterator[None]:
    """Ensure each test starts with a fresh cache.

    The manifest module caches the parsed YAML at module scope; without a
    reset, an earlier test's monkeypatch could leak into the next.
    """
    reload_manifest()
    yield
    reload_manifest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loads_constitution_paths() -> None:
    """Parser must surface every constitution.paths entry verbatim.

    Prevents a regression where the parser silently drops, normalizes,
    or reorders entries; the validator relies on the exact patterns.
    """
    cm = load_core_manifest()

    assert isinstance(cm, CoreManifest)
    assert isinstance(cm.constitution_paths, tuple)
    assert len(cm.constitution_paths) > 0

    # Spot-check several patterns from design §3 that must be present
    # verbatim. If the YAML format ever changes, the design must change
    # in lockstep — this test pins the contract.
    expected_subset = {
        "src/agentm/core/abi/**",
        "src/agentm/core/lib/**",
        "src/agentm/core/_internal/**",
        "src/agentm/llm/**",
        "src/agentm/harness/session.py",
        "src/agentm/extensions/loader.py",
        "src/agentm/extensions/validate.py",
        "src/agentm/cli.py",
        ".agentm/catalog/**",
        "core-manifest.yaml",
    }
    missing = expected_subset - set(cm.constitution_paths)
    assert not missing, f"manifest is missing required paths: {missing}"


def test_kernel_path_is_constitution() -> None:
    """A kernel source file is on the constitution side of the boundary.

    This is the canonical positive case for `is_constitution_path`: the
    kernel mechanism is what the agent must NOT modify directly.
    """
    assert is_constitution_path(KERNEL_PATH) is True


def test_extension_atom_is_not_constitution() -> None:
    """A builtin atom is autonomy-layer, not constitution.

    Self-edit gating depends on this returning False so the validator
    permits `reload_atom` on extension atoms.
    """
    assert is_constitution_path(EXTENSION_ATOM_PATH) is False


def test_catalog_dir_is_constitution() -> None:
    """`.agentm/catalog/**` is constitution-protected (covers acceptance E4).

    Catalog records are evidence; the agent must not be able to rewrite
    them, otherwise self-evaluation becomes spoofable.
    """
    assert is_constitution_path(CATALOG_METRICS_PATH) is True


def test_manifest_yaml_is_self_referential() -> None:
    """The manifest itself is on the constitution list.

    Without this, a self-modifying agent could rewrite `core-manifest.yaml`
    to lift its own constraints — defeating the whole boundary.
    """
    assert is_constitution_path(MANIFEST_FILENAME) is True


def test_S10_manifest_change_moves_constitution_boundary(
    tmp_path: Path,
) -> None:
    """Acceptance scenario S10: removing a path from the manifest moves
    the constitution boundary so paths previously protected become
    autonomy-layer.

    This is the boundary test for the whole feature: the manifest is
    data, not code, and the predicate must reflect runtime data.
    """
    # Build a manifest where the kernel glob is omitted but the
    # manifest itself is still self-referential. extension_api defaults
    # match design §3.
    tmp_manifest = tmp_path / "core-manifest.yaml"
    tmp_manifest.write_text(
        "version: 1\n"
        "constitution:\n"
        "  paths:\n"
        "    - src/agentm/core/lib/**\n"
        "    - core-manifest.yaml\n"
        "extension_api:\n"
        "  current: 1\n"
        "  deprecation:\n"
        "    grace: 1\n"
        "reload:\n"
        "  tier_2_atoms: []\n",
        encoding="utf-8",
    )

    # Redirect the loader at the temp manifest for the scope of this test.
    with manifest_mod.override_manifest_path(tmp_manifest):
        cm = load_core_manifest()

        # Sanity: the new manifest is what the loader sees.
        assert "src/agentm/core/lib/**" in cm.constitution_paths
        assert "core-manifest.yaml" in cm.constitution_paths
        assert "src/agentm/core/abi/**" not in cm.constitution_paths

        # The boundary moved: the kernel (abi) file is now autonomy-layer.
        assert is_constitution_path(KERNEL_PATH) is False
        # ...but lib/ paths and the manifest itself are still protected.
        assert is_constitution_path("src/agentm/core/lib/path_utils.py") is True
        assert is_constitution_path(MANIFEST_FILENAME) is True
