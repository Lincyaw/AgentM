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

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentm.core._internal.catalog import manifest as manifest_mod
from agentm.core._internal.catalog.manifest import (
    CoreManifest,
    configure_manifest_path,
    is_constitution_path,
    load_core_manifest,
    reload_manifest,
    reset_manifest_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KERNEL_PATH = "src/agentm/core/abi/loop.py"
EXTENSION_ATOM_PATH = "src/agentm/extensions/builtin/file_tools.py"
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
        "src/agentm/core/runtime/**",
        "src/agentm/ai/**",
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
        assert is_constitution_path("src/agentm/core/lib/frontmatter.py") is True
        assert is_constitution_path(MANIFEST_FILENAME) is True


# ---------------------------------------------------------------------------
# Cache-correctness fail-stops (constitution write barrier)
# ---------------------------------------------------------------------------


def _write_manifest(target: Path, constitution_paths: list[str]) -> None:
    body = "version: 1\nconstitution:\n  paths:\n"
    for entry in constitution_paths:
        body += f"    - {entry}\n"
    body += (
        "extension_api:\n"
        "  current: 1\n"
        "  deprecation:\n"
        "    grace: 1\n"
        "reload:\n"
        "  tier_2_atoms: []\n"
    )
    target.write_text(body, encoding="utf-8")


def test_cross_cwd_bindings_do_not_thrash(tmp_path: Path) -> None:
    """Switching the ContextVar binding back and forth must answer each
    binding's question correctly.

    Regression: the old implementation paired a ContextVar with a
    process-global ``@functools.cache`` and ``cache_clear`` on every
    bind, so two concurrent sessions in different cwds could observe
    each other's stale data. Dropping the cache means the binding alone
    determines the answer; re-binding the same path must continue to
    work.
    """
    manifest_a = tmp_path / "a" / "core-manifest.yaml"
    manifest_b = tmp_path / "b" / "core-manifest.yaml"
    manifest_a.parent.mkdir()
    manifest_b.parent.mkdir()
    _write_manifest(manifest_a, ["src/agentm/core/abi/**", "core-manifest.yaml"])
    _write_manifest(manifest_b, ["src/agentm/core/lib/**", "core-manifest.yaml"])

    token_a = configure_manifest_path(manifest_a)
    try:
        assert is_constitution_path("src/agentm/core/abi/loop.py") is True
        assert is_constitution_path("src/agentm/core/lib/frontmatter.py") is False

        token_b = configure_manifest_path(manifest_b)
        try:
            # B's binding shadows A's — only B's globs apply now.
            assert is_constitution_path("src/agentm/core/abi/loop.py") is False
            assert is_constitution_path("src/agentm/core/lib/frontmatter.py") is True
        finally:
            reset_manifest_path(token_b)

        # Re-bound to A's view; the answer must flip back without
        # depending on any prior cache_clear having fired.
        assert is_constitution_path("src/agentm/core/abi/loop.py") is True
        assert is_constitution_path("src/agentm/core/lib/frontmatter.py") is False
    finally:
        reset_manifest_path(token_a)


def test_on_disk_edit_visible_without_explicit_reload(tmp_path: Path) -> None:
    """Editing the manifest file on disk must be reflected on the next
    query, without an explicit ``reload_manifest()`` call.

    Regression: the old ``@functools.cache`` was keyed on path alone, so
    a self-modifying agent that rewrote ``core-manifest.yaml`` would
    keep seeing the pre-edit policy. The dropped cache means each query
    reparses, so the new policy applies immediately.
    """
    manifest = tmp_path / "core-manifest.yaml"
    _write_manifest(manifest, ["src/agentm/core/abi/**", "core-manifest.yaml"])

    with manifest_mod.override_manifest_path(manifest):
        assert is_constitution_path("src/agentm/core/abi/loop.py") is True
        assert is_constitution_path("src/agentm/core/lib/frontmatter.py") is False

        # Rewrite the manifest in place — same path, new policy. No
        # cache_clear / reload_manifest call: the next query must see
        # the new globs purely from the on-disk content.
        _write_manifest(manifest, ["src/agentm/core/lib/**", "core-manifest.yaml"])

        assert is_constitution_path("src/agentm/core/abi/loop.py") is False
        assert is_constitution_path("src/agentm/core/lib/frontmatter.py") is True


def test_concurrent_asyncio_bindings_do_not_leak(tmp_path: Path) -> None:
    """Two coroutines with different manifest bindings must each see
    their own answer under interleaved execution.

    Forces interleaving with ``await asyncio.sleep(0)`` between the bind
    and the query so the scheduler gets a chance to run the other
    coroutine before the assertion. ContextVar scoping is the only
    mechanism keeping the two views apart — any process-global state
    (the old cache) would surface here.
    """
    manifest_a = tmp_path / "a" / "core-manifest.yaml"
    manifest_b = tmp_path / "b" / "core-manifest.yaml"
    manifest_a.parent.mkdir()
    manifest_b.parent.mkdir()
    _write_manifest(manifest_a, ["src/agentm/core/abi/**", "core-manifest.yaml"])
    _write_manifest(manifest_b, ["src/agentm/core/lib/**", "core-manifest.yaml"])

    async def session(manifest: Path, probe_true: str, probe_false: str) -> None:
        token = configure_manifest_path(manifest)
        try:
            await asyncio.sleep(0)
            assert is_constitution_path(probe_true) is True
            await asyncio.sleep(0)
            assert is_constitution_path(probe_false) is False
        finally:
            reset_manifest_path(token)

    async def driver() -> None:
        await asyncio.gather(
            session(
                manifest_a,
                probe_true="src/agentm/core/abi/loop.py",
                probe_false="src/agentm/core/lib/frontmatter.py",
            ),
            session(
                manifest_b,
                probe_true="src/agentm/core/lib/frontmatter.py",
                probe_false="src/agentm/core/abi/loop.py",
            ),
        )

    asyncio.run(driver())
