"""§11 contract gate.

This single test runs ``validate_builtin`` over every module under
``agentm/extensions/builtin/``. **Every PR that adds or modifies a built-in
extension must keep this green.** The test is intentionally minimal — the
validator does the heavy lifting, and the test exists so an agent self-
editing an extension gets a single mechanical "did I break the contract"
signal in CI / pytest.
"""

from __future__ import annotations

from agentm.extensions.discover import discover_builtin, reset_cache
from agentm.extensions.validate import validate_builtin


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
