"""§11 extension contract gate.

Runs ``validate_builtin`` over every built-in extension and validates
contrib packages.  Every PR that adds or modifies an extension must keep
this green.
"""

from __future__ import annotations

import pytest

from agentm.extensions.discover import reset_cache
from agentm.extensions.validate import (
    ValidationIssue,
    validate_atom_package,
    validate_builtin,
)


def setup_function() -> None:
    reset_cache()


def test_builtin_catalog_passes_section_11_contract() -> None:
    issues = validate_builtin()
    assert issues == [], "\n".join(
        f"  - {i.module_path} [{i.rule}]: {i.message}" for i in issues
    )


def test_existing_contrib_packages_pass_import_validation() -> None:
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

    import_issues = [
        i for i in all_issues if i.severity == "error" and i.rule.startswith("11.4.5")
    ]
    assert not import_issues, (
        "contrib packages have forbidden import violations:\n"
        + "\n".join(
            f"  - {i.module_path} [{i.rule}]: {i.message}" for i in import_issues
        )
    )
    assert validated > 0, "expected at least one contrib package to validate"
