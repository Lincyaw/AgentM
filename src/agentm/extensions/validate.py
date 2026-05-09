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
9. ``MANIFEST.api_version`` is within the host's
   ``[current - grace, current]`` window (self-modifiable-architecture §4).
10. ``MANIFEST.tier`` agrees with ``core-manifest.yaml::reload.tier_2_atoms``.
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType

from agentm.core._internal.catalog import manifest as core_manifest_mod
from agentm.extensions import ExtensionManifest, parse_register_tag
from agentm.extensions.discover import discover_builtin

# Modules an extension is allowed to import. See design §11.1 rule 4.
_ALLOWED_PREFIXES: tuple[str, ...] = (
    "agentm.core.abi",
    "agentm.core.lib",
    "agentm.harness.extension",
    "agentm.harness.events",
    "agentm.harness.session_manager",
    "agentm.harness.session_config",
    "agentm.harness.resource_loader",
    "agentm.harness.resource_writer",
    "agentm.extensions",
)

# Imports that are explicitly forbidden — listed for clearer error messages
# than "not on the allow-list".
_FORBIDDEN_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "agentm.core._internal",
        "constitution-private modules — reach via api.skills / "
        "api.prompt_templates / api.catalog / api.compaction / "
        "api.get_operations() instead",
    ),
    (
        "agentm.harness.session",
        "extensions never reach inside the orchestrator",
    ),
    (
        "agentm.extensions.builtin.",
        "atom-to-atom coupling forbidden — depend via events / api only",
    ),
    (
        "_agentm_contrib__",
        "contrib atoms must stay decoupled from each other — depend via "
        "events / api only",
    ),
    (
        "agentm._scenarios.",
        "scenario-local atom-to-atom coupling forbidden — depend via "
        "events / api only",
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
    severity: str = "error"
    """``"error"`` (default, blocks reload) or ``"warning"`` (visible drift,
    does not block CI). Introduced for §11.4.10 tier-list mismatches that
    surface during in-progress rollouts where the atom and the manifest list
    are updated in the same PR."""


def _builtin_dir() -> Path:
    pkg = import_module("agentm.extensions.builtin")
    if pkg.__file__ is None:  # pragma: no cover — defensive
        raise RuntimeError("agentm.extensions.builtin has no __file__")
    return Path(pkg.__file__).parent


def validate_builtin() -> list[ValidationIssue]:
    """Validate every module under ``builtin/``.

    Returns a list of issues; empty list ⇒ compliant catalog.

    §11.4.10 (tier-list mismatch) is the load-time half of a two-layer
    defense: an agent-driven tier downgrade also gets rejected at the
    ``propose_change`` layer in Phase 2, but this check fires at module
    import time so a hand-edited atom drifting from
    ``core-manifest.yaml::reload.tier_2_atoms`` surfaces in PR review
    rather than at runtime.
    """

    issues: list[ValidationIssue] = []
    builtin_dir = _builtin_dir()

    discovered = discover_builtin()
    discovered_names = set(discovered)

    # §11.4.9 / §11.4.10 read constants from the core manifest. Resolve
    # through the module namespace so test fixtures that monkeypatch
    # ``agentm.core._internal.catalog.manifest.load_core_manifest`` take effect.
    core_manifest = core_manifest_mod.load_core_manifest()
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

        src_file = inspect.getsourcefile(module)
        issues.extend(
            validate_extension_contract(
                module_path=module_path,
                module=module,
                src_file=Path(src_file) if src_file is not None else None,
                known_extension_names=discovered_names,
                core_manifest=core_manifest,
                manifest=entry.manifest,
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
        tree = ast.parse(src_file.read_text(encoding="utf-8"), type_comments=True)
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
        elif isinstance(node, ast.Call):
            # Catch ``importlib.import_module("agentm.harness.session")``
            # and ``__import__("agentm.harness.session")`` — dynamic imports
            # bypass the static check above and have been used in the past
            # (sub_agent before A2) to silently slip past the §11.4.5
            # forbidden list. We only flag *constant* arguments; expressions
            # are out of scope for AST-only checking.
            target = _dynamic_import_target(node)
            if target is not None:
                names = [target]
        for imported in names:
            issues.extend(_classify_import(module_path, imported))

    return issues


def _check_ast_rules(module_path: str, src_file: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        tree = ast.parse(src_file.read_text(encoding="utf-8"), type_comments=True)
    except (OSError, SyntaxError) as exc:  # pragma: no cover — defensive
        return [
            ValidationIssue(
                module_path=module_path,
                rule="11.4.ast-parse",
                message=f"could not parse {src_file}: {exc}",
            )
        ]

    ignored_lines = {ignore.lineno for ignore in tree.type_ignores}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_private_api_getattr(node):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.D1-private-api-reflection",
                    message="getattr(api, '_*') and getattr(api.events, '_*') are forbidden",
                )
            )
        if isinstance(node, ast.Call) and _is_agentm_fstring_import(node):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.D5-dynamic-agentm-import",
                    message="dynamic f-string imports under 'agentm.' are forbidden",
                )
            )
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if target.lineno not in ignored_lines and _is_api_attribute_target(target):
                    issues.append(
                        ValidationIssue(
                            module_path=module_path,
                            rule="11.4.D2-api-attribute-overwrite",
                            message="atoms must not overwrite ExtensionAPI attributes",
                        )
                    )
        elif (
            isinstance(node, (ast.AnnAssign, ast.AugAssign))
            and node.target.lineno not in ignored_lines
            and _is_api_attribute_target(node.target)
        ):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.D2-api-attribute-overwrite",
                    message="atoms must not overwrite ExtensionAPI attributes",
                )
            )

    for node in tree.body:
        if _is_unfinalized_mutable_global(node):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.D3-mutable-global",
                    message="module-level dict/list/set globals must be annotated Final",
                )
            )
    return issues


def _is_private_api_getattr(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Name) or node.func.id != "getattr" or len(node.args) < 2:
        return False
    target, attr = node.args[0], node.args[1]
    if not (
        isinstance(attr, ast.Constant)
        and isinstance(attr.value, str)
        and attr.value.startswith("_")
    ):
        return False
    if isinstance(target, ast.Name) and target.id == "api":
        return True
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "events"
        and isinstance(target.value, ast.Name)
        and target.value.id == "api"
    )


def _is_api_attribute_target(target: ast.expr) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "api"
    )


def _is_agentm_fstring_import(node: ast.Call) -> bool:
    func = node.func
    is_dynamic_import = False
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        is_dynamic_import = True
    elif isinstance(func, ast.Name) and func.id in {"import_module", "__import__"}:
        is_dynamic_import = True
    if not is_dynamic_import or not node.args:
        return False
    first = node.args[0]
    if not isinstance(first, ast.JoinedStr) or not first.values:
        return False
    prefix = first.values[0]
    return (
        isinstance(prefix, ast.Constant)
        and isinstance(prefix.value, str)
        and prefix.value.startswith("agentm.")
    )


def _is_unfinalized_mutable_global(node: ast.stmt) -> bool:
    if isinstance(node, ast.AnnAssign):
        return _is_mutable_literal(node.value) and not _annotation_is_final(node.annotation)
    if isinstance(node, ast.Assign):
        return _is_mutable_literal(node.value)
    return False


def _is_mutable_literal(node: ast.expr | None) -> bool:
    return isinstance(node, (ast.Dict, ast.List, ast.Set))


def _annotation_is_final(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Final"
    if isinstance(node, ast.Attribute):
        return node.attr == "Final"
    if isinstance(node, ast.Subscript):
        return _annotation_is_final(node.value)
    return False


def _dynamic_import_target(node: ast.Call) -> str | None:
    """Return the constant module name passed to ``importlib.import_module``
    or ``__import__``, or ``None`` if the call is unrelated or non-constant."""

    func = node.func
    is_dynamic_import = False
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        # importlib.import_module(...)
        is_dynamic_import = True
    elif isinstance(func, ast.Name) and func.id in {"import_module", "__import__"}:
        is_dynamic_import = True
    if not is_dynamic_import or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def validate_atom_file(path: str | Path, *, module_path: str = "<atom>") -> list[ValidationIssue]:
    src_file = Path(path)
    return [*_check_imports(module_path, src_file), *_check_ast_rules(module_path, src_file)]


def _classify_import(
    module_path: str, imported: str
) -> list[ValidationIssue]:
    # Allow stdlib by default — anything that does not start with "agentm."
    # and is not an explicitly-forbidden third party (langchain et al.) is
    # treated as stdlib / approved third party. This mirrors the design
    # philosophy: extensions can pull in any *neutral* stdlib helper, but
    # AgentM-internal coupling is what we lock down.
    for forbidden, reason in _FORBIDDEN_PREFIXES:
        bare = forbidden.rstrip(".")
        # Match the exact module OR a subpackage of it (boundary on '.').
        # Without the dot, ``agentm.harness.session`` would also reject the
        # legitimate ``agentm.harness.session_config`` and ``session_manager``.
        if imported == bare or imported.startswith(bare + "."):
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


def validate_extension_contract(
    *,
    module_path: str,
    module: ModuleType,
    src_file: Path | None,
    known_extension_names: set[str],
    core_manifest: core_manifest_mod.CoreManifest | None = None,
    manifest: ExtensionManifest | None = None,
) -> list[ValidationIssue]:
    """Validate one already-imported extension module against rules 5-10."""

    issues: list[ValidationIssue] = []
    resolved_manifest = (
        manifest if manifest is not None else getattr(module, "MANIFEST")
    )
    name = resolved_manifest.name
    resolved_core_manifest = (
        core_manifest
        if core_manifest is not None
        else core_manifest_mod.load_core_manifest()
    )
    api_current = resolved_core_manifest.extension_api_current
    api_grace = resolved_core_manifest.extension_api_grace
    tier_2_set = set(resolved_core_manifest.tier_2_atoms)

    if src_file is not None:
        issues.extend(_check_imports(module_path, src_file))
        issues.extend(_check_ast_rules(module_path, src_file))

    for tag in resolved_manifest.registers:
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

    for ref in resolved_manifest.requires:
        if ref not in known_extension_names:
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
    for ref in resolved_manifest.conflicts:
        if ref not in known_extension_names:
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

    schema = resolved_manifest.config_schema
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

    api_version = resolved_manifest.api_version
    if api_version > api_current:
        issues.append(
            ValidationIssue(
                module_path=module_path,
                rule="11.4.9-api-version-too-new",
                message=(
                    f"atom requires api_version {api_version}, "
                    f"current is {api_current}"
                ),
            )
        )
    if api_version < api_current - api_grace:
        issues.append(
            ValidationIssue(
                module_path=module_path,
                rule="11.4.9-api-version-too-old",
                message=(
                    f"atom api_version {api_version} is older than "
                    f"the grace window (current={api_current}, "
                    f"grace={api_grace})"
                ),
            )
        )

    tier = resolved_manifest.tier
    if tier == 2 and name not in tier_2_set:
        issues.append(
            ValidationIssue(
                module_path=module_path,
                rule="11.4.10-tier-list-mismatch",
                message=(
                    f"atom {name!r} declares tier=2 but is not listed in "
                    "core-manifest.yaml::reload.tier_2_atoms"
                ),
                severity="warning",
            )
        )
    if tier != 2 and name in tier_2_set:
        issues.append(
            ValidationIssue(
                module_path=module_path,
                rule="11.4.10-tier-list-mismatch",
                message=(
                    f"atom {name!r} is listed in "
                    "core-manifest.yaml::reload.tier_2_atoms "
                    f"but declares tier={tier}"
                ),
                severity="warning",
            )
        )

    return issues


__all__ = [
    "ValidationIssue",
    "validate_atom_file",
    "validate_extension_contract",
    "validate_builtin",
]
