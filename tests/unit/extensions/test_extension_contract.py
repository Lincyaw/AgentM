"""§11 contract gate.

This single test runs ``validate_builtin`` over every module under
``agentm/extensions/builtin/``. **Every PR that adds or modifies a built-in
extension must keep this green.** The test is intentionally minimal — the
validator does the heavy lifting, and the test exists so an agent self-
editing an extension gets a single mechanical "did I break the contract"
signal in CI / pytest.
"""

from __future__ import annotations

import importlib

import pytest

from agentm.core.catalog import manifest as core_manifest_mod
from agentm.core.catalog.manifest import CoreManifest
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import BuiltinEntry, discover_builtin, reset_cache
from agentm.extensions.validate import (
    ValidationIssue,
    validate_builtin,
)


def setup_function() -> None:
    # Each test starts from a clean discovery state so test fixtures do not
    # leak across modules.
    reset_cache()


def test_builtin_catalog_passes_section_11_contract() -> None:
    """Every built-in extension must satisfy the §11 contract.

    If this fails, read the issue list — each entry includes the offending
    module path and the contract rule that was violated.
    """

    issues = validate_builtin()
    assert issues == [], "\n".join(
        f"  - {i.module_path} [{i.rule}]: {i.message}" for i in issues
    )


def test_discover_returns_dict_keyed_by_name() -> None:
    """Discovery contract: maps name → entry, name equals module stem."""

    catalog = discover_builtin()
    for name, entry in catalog.items():
        assert entry.name == name
        assert entry.manifest.name == name
        assert entry.module_path == f"agentm.extensions.builtin.{name}"


# ---------------------------------------------------------------------------
# manifest-schema task: §11.4.9 (versioned API gate) and §11.4.10
# (tier-list mismatch). These tests synthesize a single fake atom on top of
# a real on-disk module (``permission``) so the validator's import allow-list
# / inspect.getsourcefile checks don't add unrelated noise — only the
# manifest is crafted per-test.
#
# Style choice: construct ``BuiltinEntry`` instances inline and monkeypatch
# ``agentm.extensions.validate.discover_builtin`` to return only the
# synthetic catalog. This mirrors the in-test construction pattern used
# elsewhere (no on-disk fixture files) — the task spec permits either; we
# pick this one because:
#   1. No new file system surface to maintain.
#   2. The five existing extension fixture dirs already in the repo
#      (``tests/unit/extensions/builtin/_fixtures``) follow a different
#      "real Python module" pattern that would conflict with the §11.4.1
#      subpackage rule.
# ---------------------------------------------------------------------------


def _stub_entry(manifest: ExtensionManifest) -> BuiltinEntry:
    """Build a ``BuiltinEntry`` whose ``module`` is a real on-disk file
    (``permission.py``) so the validator's source-file rules pass; only the
    manifest is the variable under test."""

    module = importlib.import_module("agentm.extensions.builtin.permission")
    return BuiltinEntry(
        name=manifest.name,
        module_path=f"agentm.extensions.builtin.{manifest.name}",
        module=module,
        manifest=manifest,
    )


def _patch_catalog(
    monkeypatch: pytest.MonkeyPatch, entries: dict[str, BuiltinEntry]
) -> None:
    """Replace ``validate.discover_builtin`` with a stub returning ``entries``."""

    monkeypatch.setattr(
        "agentm.extensions.validate.discover_builtin", lambda: entries
    )


def _patch_core_manifest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current: int = 1,
    grace: int = 1,
    tier_2_atoms: tuple[str, ...] = (),
) -> None:
    """Force ``load_core_manifest`` to return a deterministic fake. The
    validator reads ``extension_api_current`` / ``extension_api_grace`` /
    ``tier_2_atoms`` from this surface, so tests must control them
    independently of the on-disk ``core-manifest.yaml``."""

    fake = CoreManifest(
        version=1,
        constitution_paths=(),
        extension_api_current=current,
        extension_api_grace=grace,
        tier_2_atoms=tier_2_atoms,
    )
    monkeypatch.setattr(core_manifest_mod, "load_core_manifest", lambda: fake)


def _has_rule(issues: list[ValidationIssue], rule: str, name: str) -> bool:
    return any(i.rule == rule and i.module_path.endswith(name) for i in issues)


def test_S7_api_version_too_new_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance scenario S7: an atom declaring an api_version above
    ``extension_api_current`` is rejected by the validator with rule
    ``11.4.9-api-version-too-new``.

    This is the front-side of the version-window gate. Without it, an atom
    written for a future API revision would silently load on an older host
    and crash at first event delivery — the gate moves that crash to load
    time, where it can be fixed mechanically.
    """

    _patch_core_manifest(monkeypatch, current=1, grace=1)
    fake_manifest = ExtensionManifest(
        name="too_new_atom",
        description="atom declaring a future api_version",
        registers=(),
        api_version=99,
    )
    _patch_catalog(monkeypatch, {"too_new_atom": _stub_entry(fake_manifest)})

    issues = validate_builtin()

    assert _has_rule(
        issues, "11.4.9-api-version-too-new", "too_new_atom"
    ), (
        "expected rule '11.4.9-api-version-too-new' to fire on api_version=99 "
        f"with current=1; got: {[(i.rule, i.message) for i in issues]}"
    )


def test_api_version_too_old_rejected_when_outside_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-side of the version-window gate: an atom older than
    ``current - grace`` is rejected with rule ``11.4.9-api-version-too-old``.

    Production constants today are ``current=1, grace=1`` so the rejected
    window is ``api_version <= -1``; we drive that with a synthetic
    manifest. This test is a regression guard for the grace-window
    arithmetic — if a future refactor forgets to subtract ``grace`` it
    will fail.
    """

    current, grace = 1, 1
    too_old = current - grace - 1  # i.e. -1 with the above
    if too_old < 0:
        # api_version=-1 is technically constructible (int field, no bounds)
        # but if a future change adds a non-negativity check at the
        # dataclass level, we lose the rejected window for current=grace=1.
        # Document the dependency rather than silently passing.
        pass

    _patch_core_manifest(monkeypatch, current=current, grace=grace)
    fake_manifest = ExtensionManifest(
        name="too_old_atom",
        description="atom declaring a stale api_version",
        registers=(),
        api_version=too_old,
    )
    _patch_catalog(monkeypatch, {"too_old_atom": _stub_entry(fake_manifest)})

    issues = validate_builtin()

    assert _has_rule(
        issues, "11.4.9-api-version-too-old", "too_old_atom"
    ), (
        "expected rule '11.4.9-api-version-too-old' to fire on "
        f"api_version={too_old} with current={current}, grace={grace}; "
        f"got: {[(i.rule, i.message) for i in issues]}"
    )


def test_tier_2_atom_not_in_manifest_list_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An atom declaring ``tier=2`` but absent from
    ``core_manifest.tier_2_atoms`` produces a warning-severity issue with
    rule ``11.4.10-tier-list-mismatch``.

    Two-layer defense (see task notes): the manifest list is the canonical
    source; the validator emits a warning when the atom and the list
    disagree so the maintainer notices the drift at PR review time. A
    warning (not an error) is intentional — it keeps CI from blocking on
    an in-progress rollout that touches both the atom and the list.
    """

    _patch_core_manifest(
        monkeypatch, current=1, grace=1, tier_2_atoms=()
    )  # empty list — atom is NOT registered there
    fake_manifest = ExtensionManifest(
        name="orphan_tier2",
        description="declares tier=2 but absent from core-manifest.yaml",
        registers=(),
        tier=2,
    )
    _patch_catalog(monkeypatch, {"orphan_tier2": _stub_entry(fake_manifest)})

    issues = validate_builtin()

    assert _has_rule(
        issues, "11.4.10-tier-list-mismatch", "orphan_tier2"
    ), (
        "expected rule '11.4.10-tier-list-mismatch' to fire when an atom's "
        f"tier=2 is not mirrored in core_manifest.tier_2_atoms; "
        f"got: {[(i.rule, i.message) for i in issues]}"
    )
    matching = [i for i in issues if i.rule == "11.4.10-tier-list-mismatch"]
    assert any(i.severity == "warning" for i in matching), (
        "tier-list-mismatch must be a warning, not an error: rolling out a "
        "tier change should not block CI on the transient diff"
    )


def test_tier_1_atom_listed_in_manifest_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror case of the tier-list gate: an atom listed in
    ``core_manifest.tier_2_atoms`` but declaring ``tier=1`` (or relying on
    the default) also emits ``11.4.10-tier-list-mismatch``.

    Without this direction, an agent could silently downgrade a tier-2
    atom to tier-1 via a self-edit and bypass the propose_change gate at
    Phase 2. The validator catches the mismatch at load time.
    """

    _patch_core_manifest(
        monkeypatch,
        current=1,
        grace=1,
        tier_2_atoms=("orphan_tier1",),  # list says tier-2…
    )
    fake_manifest = ExtensionManifest(
        name="orphan_tier1",
        description="declares tier=1 but listed in core-manifest.yaml",
        registers=(),
        tier=1,  # …atom says tier-1
    )
    _patch_catalog(monkeypatch, {"orphan_tier1": _stub_entry(fake_manifest)})

    issues = validate_builtin()

    assert _has_rule(
        issues, "11.4.10-tier-list-mismatch", "orphan_tier1"
    ), (
        "expected rule '11.4.10-tier-list-mismatch' to fire when an atom is "
        f"listed in core_manifest.tier_2_atoms but declares tier!=2; "
        f"got: {[(i.rule, i.message) for i in issues]}"
    )
