"""Validator for the single-file extension contract.

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
11. AST hygiene rules reject private API reflection, ExtensionAPI mutation,
    mutable globals, dynamic ``agentm.*`` imports, concrete runtime-service
    downcasts, undeclared peer-name string references, and undeclared direct
    mutation of ExtensionAPI-owned registries.
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType

from loguru import logger

from agentm.core._internal.catalog import manifest as core_manifest_mod
from agentm.extensions import ExtensionManifest, parse_register_tag
from agentm.extensions.discover import discover_builtin

# Modules an extension is allowed to import. See design §11.1 rule 4.
# `agentm.ai` holds provider descriptors / env-key registry — pure read-only
# metadata that adapter atoms (e.g. llmharness) consume when wiring child
# sessions. It is conceptually substrate-facing ABI even though it lives in
# its own package for CLI discovery reasons.
_ALLOWED_PREFIXES: tuple[str, ...] = (
    "agentm.core.abi",
    "agentm.core.lib",
    "agentm.core.observability",
    "agentm.ai",
    "agentm.extensions",
)

# Atoms must import from the *package* surface (``from agentm.core.abi import X``),
# not from sub-modules (``from agentm.core.abi.events import X``). This keeps the
# public API auditable via ``__init__.__all__``. Private symbols (``_force_strict``)
# that cannot be on the package surface are the only exception.
_PACKAGE_ONLY_PREFIXES: tuple[str, ...] = (
    "agentm.core.abi",
    "agentm.core.lib",
)

# Imports that are explicitly forbidden — listed for clearer error messages
# than "not on the allow-list".
_FORBIDDEN_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "agentm.core._internal",
        "constitution-private modules — reach via "
        "api.catalog / api.get_operations() instead",
    ),
    (
        "agentm.core.runtime",
        "extensions never reach inside the runtime substrate — "
        "atoms import from agentm.core.abi.* + agentm.core.lib.* only",
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
        "scenario-local atom-to-atom coupling forbidden — depend via events / api only",
    ),
    ("langchain", "langchain is forbidden in the v2 tree"),
)

_FORBIDDEN_SERVICE_ISINSTANCE_NAMES: frozenset[str] = frozenset(
    {
        "BashOperations",
        "FileOperations",
        "GitBackedResourceWriter",
    }
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
                        f"{info.name!r} is a subpackage; single-file extensions only"
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
                            message=(f"'install' must accept (api, config); got {sig}"),
                        )
                    )
            except (TypeError, ValueError) as exc:  # pragma: no cover — defensive
                # Signature introspection failed (e.g. a C-level callable);
                # the arity rule simply can't be checked for this install.
                logger.debug(
                    "extensions.validate: could not introspect install() "
                    "signature for {}: {}",
                    module_path,
                    exc,
                )

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
    module_path: str,
    src_file: Path,
    *,
    package_root: Path | None = None,
) -> list[ValidationIssue]:
    """Parse the module source and flag any import outside the allow-list.

    When *package_root* is set, relative imports (``node.level > 0``) that
    resolve to files inside the package are treated as intra-package wiring
    and are always allowed — the referenced file will be checked separately
    via the reachability graph. Absolute imports targeting the same package
    namespace are also skipped.
    """

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
        imported_names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import — intra-package wiring, always allowed
                # when we're validating inside a package context.
                if package_root is not None:
                    continue
                # Outside package context, skip relative imports (original
                # behaviour — they can't be classified without a package).
                continue
            if node.module is not None:
                names = [node.module]
                imported_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target is not None:
                names = [target]
        for imported in names:
            issues.extend(_classify_import(module_path, imported, imported_names))

    return issues


def _check_ast_rules(
    module_path: str,
    src_file: Path,
    *,
    manifest_registers: set[str] | None = None,
) -> list[ValidationIssue]:
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

    # §11.4.D7 — atoms that opt into the ``inherit_provider`` config protocol
    # must reach for the parent-provider key via ``PARENT_PROVIDER_CONFIG_KEY``,
    # never the bare ``"provider"`` literal. The constant is the only canonical
    # link between the spawn factory and the atom; bare literals silently drift.
    if _imports_parent_provider_config_key(tree):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value == "provider"
            ):
                issues.append(
                    ValidationIssue(
                        module_path=module_path,
                        rule="11.4.D7-inherit-provider-bare-literal",
                        message=(
                            "use PARENT_PROVIDER_CONFIG_KEY rather than the "
                            'bare "provider" string when reading the '
                            "inherit_provider config payload"
                        ),
                    )
                )

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
        if isinstance(node, ast.Call) and _is_forbidden_service_isinstance(node):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.D6-service-downcast",
                    message=(
                        "atoms must use ExtensionAPI/Protocol methods, not "
                        "isinstance checks against concrete runtime services"
                    ),
                )
            )
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if target.lineno not in ignored_lines and _is_api_attribute_target(
                    target
                ):
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
    issues.extend(
        _check_undeclared_api_mutation(
            module_path,
            tree,
            manifest_registers=(
                manifest_registers
                if manifest_registers is not None
                else _manifest_registers_from_tree(tree)
            ),
        )
    )
    return issues


_MUTATING_METHODS: frozenset[str] = frozenset(
    {
        "add",
        "append",
        "clear",
        "discard",
        "extend",
        "insert",
        "pop",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "update",
    }
)


def _check_undeclared_api_mutation(
    module_path: str,
    tree: ast.Module,
    *,
    manifest_registers: set[str],
) -> list[ValidationIssue]:
    if any(tag.startswith("mutates:") for tag in manifest_registers):
        return []
    install = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "install"
        ),
        None,
    )
    if install is None:
        return []
    routed_handlers = _agent_start_handler_names(install)

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            self._check_targets(node.targets)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self._check_targets([node.target])
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self._check_targets([node.target])
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                _is_api_registry_mutating_call(node)
                and not self._inside_routed_handler()
            ):
                self.found = True
            self.generic_visit(node)

        def _check_targets(self, targets: list[ast.expr]) -> None:
            if any(_is_api_registry_mutation_target(target) for target in targets):
                if not self._inside_routed_handler():
                    self.found = True

        def _inside_routed_handler(self) -> bool:
            return bool(
                self.function_stack and self.function_stack[-1] in routed_handlers
            )

    visitor = _Visitor()
    visitor.visit(install)
    return [_undeclared_mutation_issue(module_path)] if visitor.found else []


def _agent_start_handler_names(
    install: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(install):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "on" or not _is_api_registry_mutation_owner(
            node.func.value
        ):
            continue
        if len(node.args) < 2 or not _is_agent_start_channel(node.args[0]):
            continue
        handler = node.args[1]
        if isinstance(handler, ast.Name):
            names.add(handler.id)
    return names


def _undeclared_mutation_issue(module_path: str) -> ValidationIssue:
    return ValidationIssue(
        module_path=module_path,
        rule="11.4.D7-registers-mutation",
        message=(
            "install mutates an ExtensionAPI-owned registry; declare "
            "MANIFEST.registers with a 'mutates:*' tag or route the mutation "
            "through a registered agent_start handler"
        ),
        severity="warning",
    )


def _is_private_api_getattr(node: ast.Call) -> bool:
    if (
        not isinstance(node.func, ast.Name)
        or node.func.id != "getattr"
        or len(node.args) < 2
    ):
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


def _is_api_registry_mutation_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Subscript):
        target = target.value
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "api"
        and target.attr in {"tools", "events"}
    )


def _is_api_registry_mutation_owner(target: ast.expr) -> bool:
    return isinstance(target, ast.Name) and target.id == "api"


def _is_agent_start_channel(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value == "agent_start"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "CHANNEL"
        and isinstance(node.value, ast.Name)
        and node.value.id == "AgentStartEvent"
    )


def _is_api_registry_mutating_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in _MUTATING_METHODS:
        return False
    receiver = node.func.value
    return _is_api_registry_mutation_target(receiver)


def _imports_parent_provider_config_key(tree: ast.Module) -> bool:
    """True if the module imports ``PARENT_PROVIDER_CONFIG_KEY``.

    Used by §11.4.D7 to scope the bare-``"provider"``-literal check to atoms
    that are actually wired into the ``inherit_provider`` protocol; otherwise
    any string literal containing ``"provider"`` would false-positive.
    """

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "PARENT_PROVIDER_CONFIG_KEY":
                    return True
    return False


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


def _is_forbidden_service_isinstance(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Name) or node.func.id != "isinstance":
        return False
    if len(node.args) < 2:
        return False
    return _type_expr_contains_forbidden_service(node.args[1])


def _type_expr_contains_forbidden_service(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _FORBIDDEN_SERVICE_ISINSTANCE_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _FORBIDDEN_SERVICE_ISINSTANCE_NAMES
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_type_expr_contains_forbidden_service(elt) for elt in node.elts)
    return False


def _is_unfinalized_mutable_global(node: ast.stmt) -> bool:
    if isinstance(node, ast.AnnAssign):
        return _is_mutable_literal(node.value) and not _annotation_is_final(
            node.annotation
        )
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
    """Return the (best-effort) module name passed to ``importlib.import_module``
    or ``__import__``, or ``None`` if the call is unrelated or non-recognised.

    Recognises:

    * pure ``ast.Constant`` string arguments — returned verbatim;
    * ``ast.JoinedStr`` (f-string) whose first segment is a constant string
      starting with ``"agentm."`` — returned as the literal prefix so that
      ``_classify_import`` can detect dynamic imports targeting forbidden
      ``agentm.*`` namespaces (e.g. ``f"agentm.core.runtime.{name}"``). The
      remaining ``FormattedValue`` segments are conservatively treated as
      unknown and only the static prefix is returned.
    """

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
    if isinstance(first, ast.JoinedStr) and first.values:
        prefix = first.values[0]
        if (
            isinstance(prefix, ast.Constant)
            and isinstance(prefix.value, str)
            and prefix.value.startswith("agentm.")
        ):
            # Strip a trailing dot so ``_classify_import`` matches the namespace
            # without anchoring on a partial component.
            return prefix.value.rstrip(".")
    return None


def _manifest_requires_from_tree(tree: ast.Module) -> set[str]:
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MANIFEST"
            for target in node.targets
        ):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and (
            isinstance(node.target, ast.Name) and node.target.id == "MANIFEST"
        ):
            value = node.value
        if not isinstance(value, ast.Call):
            continue
        for keyword in value.keywords:
            if keyword.arg == "requires":
                return {
                    item.value
                    for item in ast.walk(keyword.value)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
    return set()


def _manifest_registers_from_tree(tree: ast.Module) -> set[str]:
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MANIFEST"
            for target in node.targets
        ):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and (
            isinstance(node.target, ast.Name) and node.target.id == "MANIFEST"
        ):
            value = node.value
        if not isinstance(value, ast.Call):
            continue
        for keyword in value.keywords:
            if keyword.arg == "registers":
                return {
                    item.value
                    for item in ast.walk(keyword.value)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
    return set()


def _check_peer_literal_requires(
    module_path: str,
    src_file: Path,
    *,
    known_extension_names: set[str],
    manifest_requires: set[str] | None = None,
    own_name: str | None = None,
) -> list[ValidationIssue]:
    try:
        tree = ast.parse(src_file.read_text(encoding="utf-8"), type_comments=True)
    except (OSError, SyntaxError) as exc:  # pragma: no cover — defensive
        return [
            ValidationIssue(
                module_path=module_path,
                rule="11.4.D4-peer-requires-parse",
                message=f"could not parse {src_file}: {exc}",
            )
        ]

    own = own_name or module_path.rpartition(".")[2]
    requires = (
        manifest_requires
        if manifest_requires is not None
        else _manifest_requires_from_tree(tree)
    )
    peers = known_extension_names - {own}
    referenced: set[str] = set()
    descriptive_keywords = frozenset({"description", "help", "message", "title"})

    class _Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            if any(
                isinstance(target, ast.Name) and target.id == "MANIFEST"
                for target in node.targets
            ):
                return
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.target.id == "MANIFEST":
                return
            self.generic_visit(node)

        def visit_Dict(self, node: ast.Dict) -> None:
            # Mapping keys are data-shape labels, not peer atom references.
            for value in node.values:
                if value is not None:
                    self.visit(value)

        def visit_Expr(self, node: ast.Expr) -> None:
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                return
            self.generic_visit(node)

        def visit_keyword(self, node: ast.keyword) -> None:
            if node.arg in descriptive_keywords:
                return
            self.visit(node.value)

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str) and node.value in peers:
                referenced.add(node.value)

    _Visitor().visit(tree)
    return [
        ValidationIssue(
            module_path=module_path,
            rule="11.4.D4-peer-requires",
            message=(
                f"string literal references peer atom {name!r} but "
                "MANIFEST.requires does not declare it"
            ),
        )
        for name in sorted(referenced - requires)
    ]


def _build_reachability_graph(entry_file: Path, package_root: Path) -> set[Path]:
    """Static AST-only import graph traversal starting from *entry_file*.

    Returns all ``.py`` files reachable through intra-package imports
    (both relative and absolute targeting the same package namespace).
    The entry file itself is included in the result.
    """

    visited: set[Path] = set()
    stack: list[Path] = [entry_file.resolve()]
    pkg_name = package_root.name
    resolved_root = package_root.resolve()

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        try:
            tree = ast.parse(current.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            resolved: Path | None = None

            if node.level > 0:
                # Relative import: resolve from current file's directory.
                base = current.parent
                for _ in range(node.level - 1):
                    base = base.parent
                if node.module:
                    # `from .foo import X` — resolve foo as module/package.
                    parts = node.module.split(".")
                    candidate = base / "/".join(parts)
                    if (candidate / "__init__.py").is_file():
                        resolved = (candidate / "__init__.py").resolve()
                    elif candidate.with_suffix(".py").is_file():
                        resolved = candidate.with_suffix(".py").resolve()
                else:
                    # `from . import a, b, c` — each name may be a module.
                    for alias in node.names or []:
                        sub = base / alias.name
                        sub_resolved: Path | None = None
                        if (sub / "__init__.py").is_file():
                            sub_resolved = (sub / "__init__.py").resolve()
                        elif sub.with_suffix(".py").is_file():
                            sub_resolved = sub.with_suffix(".py").resolve()
                        if sub_resolved is not None and sub_resolved not in visited:
                            try:
                                sub_resolved.relative_to(resolved_root)
                            except ValueError:
                                continue
                            stack.append(sub_resolved)
            elif node.module is not None and node.level == 0:
                # Absolute import targeting the same package namespace.
                parts = node.module.split(".")
                if parts[0] == pkg_name:
                    candidate = resolved_root / "/".join(parts[1:])
                    if (candidate / "__init__.py").is_file():
                        resolved = (candidate / "__init__.py").resolve()
                    elif candidate.with_suffix(".py").is_file():
                        resolved = candidate.with_suffix(".py").resolve()

            if resolved is not None and resolved not in visited:
                # Only follow files inside the package root.
                try:
                    resolved.relative_to(resolved_root)
                except ValueError:
                    continue
                stack.append(resolved)

    return visited


def validate_atom_package(
    package_root: Path,
    *,
    module_path: str = "<package>",
    known_extension_names: set[str] | None = None,
) -> list[ValidationIssue]:
    """Validate a multi-file atom package against the S11 contract.

    Finds the entry point (``__init__.py`` containing ``install``), builds
    the reachability graph of intra-package imports, and validates each
    reachable file for forbidden imports and AST rules.
    """

    issues: list[ValidationIssue] = []
    resolved_root = package_root.resolve()

    # Find the entry point.
    init_file = resolved_root / "__init__.py"
    if not init_file.is_file():
        issues.append(
            ValidationIssue(
                module_path=module_path,
                rule="11.4.pkg-entry",
                message=f"package {package_root} has no __init__.py",
            )
        )
        return issues

    # Build the reachability graph from the entry point.
    reachable = _build_reachability_graph(init_file, resolved_root)

    for src_file in sorted(reachable):
        # Build a human-readable module path for each file.
        try:
            rel = src_file.relative_to(resolved_root)
        except ValueError:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        file_module = f"{module_path}.{'.'.join(parts)}" if parts else module_path

        issues.extend(_check_imports(file_module, src_file, package_root=resolved_root))
        issues.extend(_check_ast_rules(file_module, src_file))

    return issues


def validate_atom_file(
    path: str | Path,
    *,
    module_path: str = "<atom>",
    known_extension_names: set[str] | None = None,
) -> list[ValidationIssue]:
    src_file = Path(path)
    known = (
        known_extension_names
        if known_extension_names is not None
        else set(discover_builtin())
    )
    return [
        *_check_imports(module_path, src_file),
        *_check_ast_rules(module_path, src_file),
        *_check_peer_literal_requires(
            module_path,
            src_file,
            known_extension_names=known,
        ),
    ]


def _classify_import(
    module_path: str,
    imported: str,
    all_names: list[str] | None = None,
) -> list[ValidationIssue]:
    # Allow stdlib by default — anything that does not start with "agentm."
    # and is not an explicitly-forbidden third party (langchain et al.) is
    # treated as stdlib / approved third party. This mirrors the design
    # philosophy: extensions can pull in any *neutral* stdlib helper, but
    # AgentM-internal coupling is what we lock down.
    for forbidden, reason in _FORBIDDEN_PREFIXES:
        bare = forbidden.rstrip(".")
        # Match the exact module OR a subpackage of it (boundary on '.').
        # Without the dot, ``agentm.core.runtime`` would also reject the
        # legitimate ``agentm.core.runtime_helpers``-style sibling names.
        if imported == bare or imported.startswith(bare + "."):
            # Underscore-prefixed subpackages within builtin/ are private
            # implementation details (e.g. ``_operations/``), not auto-discovered
            # atoms. An atom importing from its own private subpackage is not
            # atom-to-atom coupling — skip the check.
            if bare == "agentm.extensions.builtin":
                suffix = imported[len(bare) + 1 :]
                if suffix.startswith("_"):
                    continue
            return [
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.5-import",
                    message=(f"forbidden import {imported!r}: {reason}"),
                )
            ]
    for pkg_prefix in _PACKAGE_ONLY_PREFIXES:
        if imported.startswith(pkg_prefix + ".") and imported != pkg_prefix:
            sub = imported[len(pkg_prefix) + 1 :]
            if "." not in sub:
                if all_names and all(n.startswith("_") for n in all_names):
                    continue
                return [
                    ValidationIssue(
                        module_path=module_path,
                        rule="11.4.6-submodule",
                        message=(
                            f"import {imported!r}: use "
                            f"'from {pkg_prefix} import ...' instead of "
                            f"importing from sub-modules directly"
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
        issues.extend(
            _check_ast_rules(
                module_path,
                src_file,
                manifest_registers=set(resolved_manifest.registers),
            )
        )
        issues.extend(
            _check_peer_literal_requires(
                module_path,
                src_file,
                known_extension_names=known_extension_names,
                manifest_requires=set(resolved_manifest.requires),
                own_name=name,
            )
        )

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
                        f"requires {ref!r} which is not a discovered built-in extension"
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
    if schema is not None:
        from pydantic import BaseModel as _PydanticBaseModel

        if not (isinstance(schema, type) and issubclass(schema, _PydanticBaseModel)):
            issues.append(
                ValidationIssue(
                    module_path=module_path,
                    rule="11.4.8-config-schema",
                    message=(
                        "MANIFEST.config_schema must be a BaseModel subclass or None"
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
                    f"atom requires api_version {api_version}, current is {api_current}"
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
    "validate_atom_package",
    "validate_extension_contract",
    "validate_builtin",
]
