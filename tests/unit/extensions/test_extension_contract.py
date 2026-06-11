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
from pathlib import Path

import pytest

from agentm.core._internal.catalog import manifest as core_manifest_mod
from agentm.core._internal.catalog.manifest import CoreManifest
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import BuiltinEntry, reset_cache
from agentm.extensions.validate import (
    ValidationIssue,
    _build_reachability_graph,
    validate_atom_package,
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

    # Known D4 false positives: the atom name "persona" collides with
    # config-schema keys / dict keys in artifact_store and sub_agent. The
    # D4 heuristic flags any string literal matching a peer atom name, but
    # these usages are config fields, not atom-to-atom references.
    _D4_FALSE_POSITIVES = {
        ("agentm.extensions.builtin.artifact_store", "11.4.D4-peer-requires"),
        ("agentm.extensions.builtin.sub_agent", "11.4.D4-peer-requires"),
    }

    issues = validate_builtin()
    real_issues = [
        i for i in issues
        if (i.module_path, i.rule) not in _D4_FALSE_POSITIVES
    ]
    assert real_issues == [], "\n".join(
        f"  - {i.module_path} [{i.rule}]: {i.message}" for i in real_issues
    )


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




def _validate_source(tmp_path: Path, source: str) -> list[ValidationIssue]:
    path = tmp_path / "atom.py"
    path.write_text(source, encoding="utf-8")
    from agentm.extensions.validate import validate_atom_file

    return validate_atom_file(path, module_path="agentm.extensions.builtin.synthetic")


def test_D1_private_api_getattr_rejected(tmp_path: Path) -> None:
    issues = _validate_source(tmp_path, 'def install(api, config):\n    getattr(api, "_observer")\n')

    assert any(issue.rule == "11.4.D1-private-api-reflection" for issue in issues)






def test_D2_api_attribute_assignment_rejected(tmp_path: Path) -> None:
    issues = _validate_source(tmp_path, 'def install(api, config):\n    api.on = something\n')

    assert any(issue.rule == "11.4.D2-api-attribute-overwrite" for issue in issues)




def test_D3_mutable_global_without_final_rejected(tmp_path: Path) -> None:
    issues = _validate_source(tmp_path, '_GLOBAL: dict[str, Any] = {}\n')

    assert any(issue.rule == "11.4.D3-mutable-global" for issue in issues)




def test_D5_agentm_fstring_dynamic_import_rejected(tmp_path: Path) -> None:
    issues = _validate_source(
        tmp_path,
        'import importlib\ndef install(api, config):\n    importlib.import_module(f"agentm.{name}")\n',
    )

    assert any(issue.rule == "11.4.D5-dynamic-agentm-import" for issue in issues)




def test_D6_concrete_harness_service_isinstance_rejected(tmp_path: Path) -> None:
    issues = _validate_source(
        tmp_path,
        "def install(api, config):\n"
        "    isinstance(writer, GitBackedResourceWriter)\n",
    )

    assert any(issue.rule == "11.4.D6-service-downcast" for issue in issues)




def test_D4_peer_literal_requires_manifest_entry(tmp_path: Path) -> None:
    issues = _validate_source(
        tmp_path,
        'from agentm.extensions import ExtensionManifest\n'
        'MANIFEST = ExtensionManifest(name="synthetic", description="", registers=())\n'
        'def install(api, config):\n'
        '    return "system_prompt"\n',
    )

    assert any(issue.rule == "11.4.D4-peer-requires" for issue in issues)




def test_D7_warns_on_undeclared_api_registry_mutation(tmp_path: Path) -> None:
    issues = _validate_source(
        tmp_path,
        'def install(api, config):\n'
        '    api.tools.append(object())\n',
    )

    assert any(
        issue.rule == "11.4.D7-registers-mutation" and issue.severity == "warning"
        for issue in issues
    )


# ---------------------------------------------------------------------------
# Package-aware validation tests: _build_reachability_graph,
# validate_atom_package, and runtime load_extension hard-block.
# ---------------------------------------------------------------------------


def _make_package(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temporary atom package under *tmp_path*.

    *files* maps relative paths (``__init__.py``, ``helper.py``) to source.
    Returns the package root directory.
    """
    pkg_dir = tmp_path / "fake_atom"
    pkg_dir.mkdir()
    for rel_path, content in files.items():
        target = pkg_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return pkg_dir


def test_build_reachability_graph_includes_reachable(tmp_path: Path) -> None:
    """_build_reachability_graph includes files imported from __init__.py
    (both directly and transitively) but excludes unreachable files."""

    pkg_dir = _make_package(tmp_path, {
        "__init__.py": "from .helper import greet\ndef install(api, config): pass\n",
        "helper.py": "from .util import x\ndef greet(): pass\n",
        "util.py": "x = 1\n",
        "unreachable.py": "import os\n",
    })
    init_file = pkg_dir / "__init__.py"
    reachable = _build_reachability_graph(init_file, pkg_dir)

    reachable_names = {p.name for p in reachable}
    assert "__init__.py" in reachable_names
    assert "helper.py" in reachable_names
    assert "util.py" in reachable_names
    assert "unreachable.py" not in reachable_names


def test_build_reachability_graph_handles_cycle(tmp_path: Path) -> None:
    """Cyclic imports between modules do not cause infinite loops."""

    pkg_dir = _make_package(tmp_path, {
        "__init__.py": "from .a import x\ndef install(api, config): pass\n",
        "a.py": "from .b import y\nx = 1\n",
        "b.py": "from .a import x\ny = 2\n",
    })
    init_file = pkg_dir / "__init__.py"
    reachable = _build_reachability_graph(init_file, pkg_dir)

    reachable_names = {p.name for p in reachable}
    assert {"__init__.py", "a.py", "b.py"} == reachable_names


def test_build_reachability_graph_from_dot_import(tmp_path: Path) -> None:
    """``from . import agents, commands`` resolves each name to its module."""

    pkg_dir = _make_package(tmp_path, {
        "__init__.py": "from . import agents, commands\ndef install(api, config): pass\n",
        "agents.py": "x = 1\n",
        "commands.py": "y = 2\n",
        "unreachable.py": "z = 3\n",
    })
    init_file = pkg_dir / "__init__.py"
    reachable = _build_reachability_graph(init_file, pkg_dir)

    reachable_names = {p.name for p in reachable}
    assert "__init__.py" in reachable_names
    assert "agents.py" in reachable_names
    assert "commands.py" in reachable_names
    assert "unreachable.py" not in reachable_names


def test_validate_atom_package_catches_forbidden_import_in_reachable(tmp_path: Path) -> None:
    """A forbidden import in a reachable file is flagged."""

    pkg_dir = _make_package(tmp_path, {
        "__init__.py": "from .worker import do_thing\ndef install(api, config): pass\n",
        "worker.py": "from agentm.core.runtime.session import AgentSession\ndef do_thing(): pass\n",
    })
    issues = validate_atom_package(
        pkg_dir,
        module_path="test.fake_atom",
        known_extension_names=set(),
    )
    error_issues = [i for i in issues if i.severity == "error"]
    assert any(i.rule == "11.4.5-import" for i in error_issues), (
        f"expected forbidden import to be caught; got: {error_issues}"
    )


def test_validate_atom_package_skips_unreachable_file(tmp_path: Path) -> None:
    """A forbidden import in an unreachable file (host-driver) is not flagged."""

    pkg_dir = _make_package(tmp_path, {
        "__init__.py": "from .clean import ok\ndef install(api, config): pass\n",
        "clean.py": "ok = True\n",
        "host_driver.py": "from agentm.core.runtime.session import AgentSession\n",
    })
    issues = validate_atom_package(
        pkg_dir,
        module_path="test.fake_atom",
        known_extension_names=set(),
    )
    error_issues = [i for i in issues if i.severity == "error"]
    assert not error_issues, (
        f"unreachable host-driver violations should not appear; got: {error_issues}"
    )


def test_validate_atom_package_missing_init(tmp_path: Path) -> None:
    """A package directory without __init__.py is rejected."""

    pkg_dir = tmp_path / "no_init_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "helper.py").write_text("x = 1\n", encoding="utf-8")

    issues = validate_atom_package(pkg_dir, module_path="test.no_init")
    assert any(i.rule == "11.4.pkg-entry" for i in issues)


def test_validate_atom_package_clean_package_passes(tmp_path: Path) -> None:
    """A clean package with only allowed imports passes validation."""

    pkg_dir = _make_package(tmp_path, {
        "__init__.py": (
            "from agentm.core.abi.extension import ExtensionAPI\n"
            "from .worker import helper\n"
            "def install(api, config): pass\n"
        ),
        "worker.py": (
            "import json\n"
            "from agentm.core.lib.render import final_summary\n"
            "def helper(): pass\n"
        ),
    })
    issues = validate_atom_package(
        pkg_dir,
        module_path="test.clean_atom",
        known_extension_names=set(),
    )
    error_issues = [i for i in issues if i.severity == "error"]
    assert not error_issues, (
        f"clean package should have no errors; got: {error_issues}"
    )


def test_runtime_load_extension_blocks_bad_atom(tmp_path: Path) -> None:
    """load_extension with validate=True rejects an atom with a forbidden import."""

    import sys
    from unittest.mock import MagicMock

    from agentm.core.abi.extension import ExtensionLoadError
    from agentm.core.runtime.extension import load_extension

    # Write a bad atom file that imports a forbidden module.
    atom_file = tmp_path / "bad_atom.py"
    atom_file.write_text(
        "from agentm.core.runtime.session import AgentSession\n"
        "def install(api, config): pass\n",
        encoding="utf-8",
    )

    # Synthetically register the module so importlib.import_module works.
    import importlib.util

    module_name = f"_agentm_test_bad_atom_{id(atom_file)}"
    spec = importlib.util.spec_from_file_location(module_name, atom_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError:
        # The forbidden import will fail — that's fine, we already seeded
        # sys.modules so import_module returns it. The __file__ is set.
        pass

    api = MagicMock()
    try:
        with pytest.raises(ExtensionLoadError) as exc_info:
            load_extension(module_name, api, {}, validate=True)
        assert "contract violation" in str(exc_info.value).lower() or "11.4.5" in str(exc_info.value)
    finally:
        sys.modules.pop(module_name, None)


def test_runtime_load_extension_skips_validation_when_disabled(tmp_path: Path) -> None:
    """load_extension with validate=False does not run §11 checks."""

    import sys
    from unittest.mock import MagicMock

    from agentm.core.runtime.extension import load_extension

    atom_file = tmp_path / "ok_atom.py"
    atom_file.write_text(
        "def install(api, config): pass\n",
        encoding="utf-8",
    )

    import importlib.util

    module_name = f"_agentm_test_ok_atom_{id(atom_file)}"
    spec = importlib.util.spec_from_file_location(module_name, atom_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    api = MagicMock()
    try:
        result = load_extension(module_name, api, {}, validate=False)
        # Should succeed (install is sync, returns None).
        assert result is None
    finally:
        sys.modules.pop(module_name, None)


def test_existing_contrib_packages_pass_import_validation() -> None:
    """All existing contrib extension packages pass §11 import validation.

    This is a regression gate for the package-aware reachability graph:
    contrib packages must not have forbidden imports in reachable files.
    Pre-existing AST hygiene violations (D3 mutable-global, etc.) in
    contrib code are excluded -- those are tracked separately.
    """

    from agentm.extensions.discover import _agentm_repo_root

    repo_root = _agentm_repo_root()
    if repo_root is None:
        pytest.skip("not running from the agentm repo checkout")

    contrib_dir = repo_root / "contrib" / "extensions"
    if not contrib_dir.is_dir():
        pytest.skip("contrib/extensions/ not found")

    validated = 0
    all_issues: list[ValidationIssue] = []
    for child in sorted(contrib_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name == "tests":
            continue
        init = child / "__init__.py"
        if not init.is_file():
            continue
        module_path = f"contrib.extensions.{child.name}"
        issues = validate_atom_package(
            child, module_path=module_path, known_extension_names=set()
        )
        all_issues.extend(issues)
        validated += 1

    # Only check import-related rules (the package-aware validation's
    # contribution). AST hygiene rules (D3, D7, etc.) have pre-existing
    # violations in contrib code that are tracked separately.
    import_issues = [
        i for i in all_issues
        if i.severity == "error" and i.rule.startswith("11.4.5")
    ]
    assert not import_issues, (
        "contrib packages have forbidden import violations:\n"
        + "\n".join(
            f"  - {i.module_path} [{i.rule}]: {i.message}" for i in import_issues
        )
    )
    assert validated > 0, "expected at least one contrib package to validate"


