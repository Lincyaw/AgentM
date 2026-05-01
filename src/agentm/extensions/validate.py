"""Validator for the §11 single-file extension contract.

Checks every module under ``agentm/extensions/builtin/`` against the
contract. Returns a list of :class:`ValidationIssue`; an empty list means
the catalog is compliant. The list is the entire reporting surface so an
agent self-editing an extension gets one mechanical channel of feedback.

Checks (numbered to match design §11.4):

1. Module sits at exactly one file path under ``builtin/``; subpackages
   are flagged.
2. ``install`` exists and is callable with two positional args.
3. ``MANIFEST: ExtensionManifest`` exists.
4. ``MANIFEST.name`` matches the module stem.
5. Imports are within the allow-list.
6. Every ``MANIFEST.registers`` tag parses as ``<kind>:<id>``.
7. ``MANIFEST.requires`` / ``MANIFEST.conflicts`` reference known atoms.
8. ``MANIFEST.config_schema`` is a syntactically valid JSON-Schema dict
   (light check: dict-shaped, top-level ``type`` or ``properties`` if set).
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from agentm.extensions import parse_register_tag
from agentm.extensions.discover import discover_builtin

# Modules an extension is allowed to import. See design §11.1 rule 4.
_ALLOWED_PREFIXES: tuple[str, ...] = (
    "agentm.core.edit_diff",
    "agentm.core.kernel",
    "agentm.core.operations",
    "agentm.harness.extension",
    "agentm.harness.events",
    "agentm.harness.session_manager",
    "agentm.harness.resource_loader",
    "agentm.extensions",  # the public surface (ExtensionManifest et al.)
)

# Imports that are explicitly forbidden — listed for clearer error messages
# than "not on the allow-list".
_FORBIDDEN_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "agentm.harness.session",
        "extensions never reach inside the orchestrator",
    ),
    (
        "agentm.extensions.builtin.",
        "atom-to-atom coupling forbidden — depend via events / api only",
    ),
    (
        "agentm.harness.middleware",
        "legacy middleware tree (deleted in Phase 2.5)",
    ),
    (
        "agentm.harness.runtime",
        "legacy runtime tree (deleted in Phase 2.5)",
    ),
    (
        "agentm.harness.scenario",
        "legacy scenario tree (deleted in Phase 2.5)",
    ),
    ("langchain", "langchain is forbidden in the v2 tree"),
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One contract violation. ``rule`` keys to design §11.4 numbering so
    issue messages are mechanically classifiable."""

    module_path: str
    rule: str  # e.g. "11.4.5-import"
    message: str


def _builtin_dir() -> Path:
    pkg = import_module("agentm.extensions.builtin")
    if pkg.__file__ is None:  # pragma: no cover — defensive
        raise RuntimeError("agentm.extensions.builtin has no __file__")
    return Path(pkg.__file__).parent


def validate_builtin() -> list[ValidationIssue]:
    """Validate every module under ``builtin/``.

    Returns a list of issues; empty list ⇒ compliant catalog.
    """

    issues: list[ValidationIssue] = []
    builtin_dir = _builtin_dir()

    discovered = discover_builtin()
    discovered_names = set(discovered)

    # Rule 1: subpackages are forbidden.
    for info in pkgutil.iter_modules([str(builtin_dir)]):
        if info.ispkg and not info.name.startswith("_"):
            issues.append(
                ValidationIssue(
                    module_path=f"agentm.extensions.builtin.{info.name}",
                    rule="11.4.1-subpackage",
                    message=(
                        f"{info.name!r} is a subpackage; "
                        "single-file extensions only"
                    ),
                )
            )

    for name, entry in discovered.items():
        module_path = entry.module_path
        module = entry.module
        manifest = entry.manifest

        # Rule 2: install exists and is a 2-arg callable.
        install = getattr(module, "install", None)
        if install is None or not callable(install):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.2-install",
                    message="missing callable 'install(api, config)'",
                )
            )
        else:
            try:
                sig = inspect.signature(install)
                positional = [
                    p
                    for p in sig.parameters.values()
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                if len(positional) < 2:
                    issues.append(
                        ValidationIssue(
                            module_path=module_path,
                            rule="11.4.2-install",
                            message=(
                                "'install' must accept (api, config); "
                                f"got {sig}"
                            ),
                        )
                    )
            except (TypeError, ValueError):  # pragma: no cover — defensive
                pass

        # Rule 3 + 4: manifest already validated by discover_builtin (it
        # raises if missing or stem-mismatched). Nothing to add here.

        # Rule 5: import allow-list — parse the source AST so we don't
        # rely on accidentally-resolved runtime imports.
        src_file = inspect.getsourcefile(module)
        if src_file is not None:
            issues.extend(
                _check_imports(module_path, Path(src_file))
            )

        # Rule 6: register tag grammar.
        for tag in manifest.registers:
            try:
                parse_register_tag(tag)
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        module_path=module_path,
                        rule="11.4.6-register-tag",
                        message=str(exc),
                    )
                )

        # Rule 7: requires / conflicts reference known atoms.
        for ref in manifest.requires:
            if ref not in discovered_names:
                issues.append(
                    ValidationIssue(
                        module_path=module_path,
                        rule="11.4.7-requires",
                        message=(
                            f"requires {ref!r} which is not a discovered "
                            "built-in extension"
                        ),
                    )
                )
        for ref in manifest.conflicts:
            if ref not in discovered_names:
                issues.append(
                    ValidationIssue(
                        module_path=module_path,
                        rule="11.4.7-conflicts",
                        message=(
                            f"conflicts with {ref!r} which is not a discovered "
                            "built-in extension"
                        ),
                    )
                )
            if ref == name:
                issues.append(
                    ValidationIssue(
                        module_path=module_path,
                        rule="11.4.7-conflicts",
                        message="extension cannot conflict with itself",
                    )
                )

        # Rule 8: config_schema is a dict (or None).
        schema = manifest.config_schema
        if schema is not None and not isinstance(schema, dict):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.8-config-schema",
                    message=(
                        "MANIFEST.config_schema must be a dict (JSON-Schema) "
                        "or None"
                    ),
                )
            )

        # Rule 8 (light shape check): if schema is set and non-empty, it
        # should declare either a top-level ``type`` or ``properties``.
        if isinstance(schema, dict) and schema:
            if "type" not in schema and "properties" not in schema:
                issues.append(
                    ValidationIssue(
                        module_path=module_path,
                        rule="11.4.8-config-schema",
                        message=(
                            "MANIFEST.config_schema should declare 'type' "
                            "or 'properties' at the top level"
                        ),
                    )
                )

    return issues


# --- Import-allow-list AST check -------------------------------------------


def _check_imports(
    module_path: str, src_file: Path
) -> list[ValidationIssue]:
    """Parse the module source and flag any import outside the allow-list."""

    issues: list[ValidationIssue] = []
    try:
        tree = ast.parse(src_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:  # pragma: no cover — defensive
        return [
            ValidationIssue(
                module_path=module_path,
                rule="11.4.5-import-parse",
                message=f"could not parse {src_file}: {exc}",
            )
        ]

    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                names = [node.module]
        for imported in names:
            issues.extend(_classify_import(module_path, imported))

    return issues


def _classify_import(
    module_path: str, imported: str
) -> list[ValidationIssue]:
    # Allow stdlib by default — anything that does not start with "agentm."
    # and is not an explicitly-forbidden third party (langchain et al.) is
    # treated as stdlib / approved third party. This mirrors the design
    # philosophy: extensions can pull in any *neutral* stdlib helper, but
    # AgentM-internal coupling is what we lock down.
    for forbidden, reason in _FORBIDDEN_PREFIXES:
        if imported == forbidden.rstrip(".") or imported.startswith(forbidden):
            return [
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.5-import",
                    message=(
                        f"forbidden import {imported!r}: {reason}"
                    ),
                )
            ]
    if imported.startswith("agentm."):
        if not any(
            imported == prefix or imported.startswith(prefix + ".")
            for prefix in _ALLOWED_PREFIXES
        ):
            return [
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.5-import",
                    message=(
                        f"import {imported!r} is not on the §11.1 allow-list "
                        f"({list(_ALLOWED_PREFIXES)})"
                    ),
                )
            ]
    return []


__all__ = [
    "ValidationIssue",
    "validate_builtin",
]
