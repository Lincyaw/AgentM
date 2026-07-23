# code-health: ignore-file[AM025] -- AST checks require node-class discrimination
"""Project-level code health checks — AST-based static analysis.

Complements ``ruff`` (general Python style) and ``mypy`` (type correctness)
with project-specific architectural rules that neither tool can express.
Pure AST analysis — no imports, no execution, fast on large trees.

Rules:

- **AM001** ``silent-except``: ``except Exception`` without log/raise/warn
  in the handler body. The project convention (``feedback_no_silent_exceptions``)
  requires every catch to surface the error.
- **AM002** ``missing-slots``: ``@dataclass`` without ``slots=True`` on
  Python 3.12+. Zero-cost optimization + prevents accidental attr assignment.
- **AM003** ``private-in-all``: ``_``-prefixed names in ``__all__``. Standard
  module metadata dunders such as ``__version__`` are allowed; other names are
  either public (drop the underscore) or private (drop from ``__all__``).
- **AM004** ``atom-raw-io``: ``open()`` / ``subprocess`` calls inside atom
  files. Atoms must use ``Operations`` (file/bash) so sandbox isolation works.
- **AM005** ``param-explosion``: Functions with >15 parameters. Signals a
  config-object extraction opportunity. Typer/Click command functions are
  skipped because their signatures are the user-facing CLI option schema.
- **AM006** ``mutable-abi-global``: Mutable containers (``dict``, ``list``,
  ``set``) at module level in ``core/abi/`` without ``Final`` annotation.
  ABI should be definitions-only; mutable state belongs in runtime.
- **AM007** ``god-file``: Files exceeding 1500 LOC. Warning-only signal
  for potential split candidates.
- **AM008** ``redundant-local-import``: A function-body import that
  duplicates a module-level import. Inconsistent style with no benefit.
- **AM009** ``god-class``: Concrete classes with >25 concrete methods.
  Protocols, type-only ``@overload`` signatures, and ``@property`` accessors
  are ignored. Swiss-army-knife signal — consider splitting into composable
  facades.
- **AM010** ``cross-layer-import``: Imports that violate the layer
  hierarchy (e.g. core → extension/backend, storage → runtime,
  gateway → builtin atoms, or ABI → runtime). Extends §11 to cover the
  full dependency graph. Constitution-listed exceptions require a precise
  line-level ignore.
- **AM011** ``hand-written-schema``: Tool ``parameters`` defined from a
  dict literal or schema-factory helper instead of a Pydantic model or
  ``pydantic_to_tool_schema(Model)``. Pydantic schemas are the project
  convention — they stay in sync with validation, generate descriptions
  from ``Field()``, and avoid hand-maintained JSON Schema boilerplate.
- **AM012** ``config-dict-splat``: ``**dict`` unpacking into a core typed
  contract — ``AgentSessionConfig(...)``, ``FunctionTool(...)``, or
  ``spawn_child_session(...)``. These are typed contracts — build them
  with explicit fields so the type checker sees every knob. Genuine JSON
  payloads (e.g. ``lineage=``/``experiment=``/``metadata=`` values) stay
  dicts; only the typed *object itself* must not be assembled from an
  untyped dict. Vendor SDK calls (``openai``/``anthropic``/``arl``) and
  pydantic ``super().__init__(**data)`` are out of scope by design.
- **AM013** ``legacy-asyncio-timeout-error``: Catching
  ``asyncio.TimeoutError`` instead of builtin ``TimeoutError``. On Python
  3.12 this is an alias; using the builtin keeps timeout handling uniform
  with modern stdlib guidance and ruff's Python-3.12 pyupgrade rules.
- **AM014** ``resolved-parent-chain``: ``path.resolve().parent`` or
  ``path.resolve().parents[...]``. Resolve-before-parent hides intent and
  silently follows symlinks; split into ``real = path.resolve(); real.parent``
  only when symlink resolution is intentional.
- **AM017** ``core-atom-policy``: Core runtime must not name concrete builtin
  atom implementations. Default composition belongs in scenario manifests.
- **AM018** ``scenario-loader-execution``: Scenario discovery returns durable
  extension data and must not execute local Python or mutate ``sys.modules``.
- **AM019** ``legacy-extension-shape``: Canonical ``ExtensionSpec`` values
  must not be treated as legacy ``(module, config)`` tuples.
- **AM020** ``raw-cli-extension-config``: CLI output must not materialize
  secret-bearing extension config without redaction.
- **AM021** ``dynamic-attribute-access``: ``getattr`` / ``hasattr`` /
  ``setattr`` / ``delattr`` bypass typed contracts. A precise ignore is
  required at genuine reflection or vendor-adapter boundaries.
- **AM022** ``typing-any``: Source annotations must not erase values to
  ``typing.Any``. Genuine vendor/reflection/serialization boundaries require
  a precise ignore.
- **AM023** ``bare-dict``: Source annotations must parameterize ``dict`` or
  use a more precise DTO / Protocol.
- **AM024** ``stdlib-logging``: AgentM uses Loguru consistently; importing
  the standard-library ``logging`` package creates split configuration and
  output behavior.
- **AM025** ``runtime-type-check``: ``isinstance`` branches on runtime shape
  and usually hides a missing typed contract, DTO, Protocol, or dispatch table.
  A precise ignore is required at genuine validation/deserialization boundaries.

AM015 and AM016 intentionally do not exist in v2. They guarded the old mutable
event/HookContract model; v2 events are frozen DTOs and handlers return typed
control values, so those failure modes are structurally impossible.

Invocation::

    agentm lint [--changed] [paths...]
    python -m agentm.code_health src/agentm/

``--changed`` checks only files modified since the merge-base with ``main``.
Exit code 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import typer


@dataclass(frozen=True, slots=True)
class Issue:
    """One code-health violation."""

    path: str
    line: int
    rule: str
    message: str
    severity: str = "error"


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

_SURFACE_CALL_NAMES: Final[frozenset[str]] = frozenset({"print", "warn"})
_SURFACE_METHOD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "critical",
        "debug",
        "echo",
        "emit",
        "error",
        "exception",
        "info",
        "log",
        "print",
        "print_exc",
        "print_exception",
        "put",
        "put_nowait",
        "send",
        "send_nowait",
        "set_exception",
        "warning",
        "warn",
    }
)

_ATOM_BUILTIN_PARTS: Final[tuple[str, str]] = ("extensions", "builtin")
_PUBLIC_MODULE_METADATA_DUNDERS: Final[frozenset[str]] = frozenset(
    {
        "__author__",
        "__copyright__",
        "__license__",
        "__version__",
    }
)
_CLI_DECORATOR_METHODS: Final[frozenset[str]] = frozenset(
    {"callback", "command", "group"}
)
_CLI_DECORATOR_NAMES: Final[frozenset[str]] = frozenset({"command", "group"})


def _is_atom_file(path: Path) -> bool:
    parts = path.parts
    return any(
        parts[index : index + len(_ATOM_BUILTIN_PARTS)] == _ATOM_BUILTIN_PARTS
        for index in range(len(parts) - len(_ATOM_BUILTIN_PARTS) + 1)
    )


def _exception_type_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return {"<bare>"}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for item in node.elts:
            names.update(_exception_type_names(item))
        return names
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    return set()


class _ExceptionSurfaceVisitor(ast.NodeVisitor):
    """Find exception surfacing without being fooled by nested scopes/handlers."""

    def __init__(self, exception_name: str | None) -> None:
        self.surfaced = False
        self._exception_name = exception_name

    def _references_exception(self, node: ast.AST | None) -> bool:
        if node is None or self._exception_name is None:
            return False
        return any(
            isinstance(child, ast.Name) and child.id == self._exception_name
            for child in ast.walk(node)
        )

    def visit_Raise(self, node: ast.Raise) -> None:  # noqa: N802
        self.surfaced = True

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Name) and func.id in _SURFACE_CALL_NAMES:
            self.surfaced = True
            return
        if isinstance(func, ast.Attribute) and func.attr in _SURFACE_METHOD_NAMES:
            self.surfaced = True
            return
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "append"
            and any(self._references_exception(arg) for arg in node.args)
        ):
            self.surfaced = True
            return
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        if self._references_exception(node.value):
            self.surfaced = True

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        # A nested handler surfacing its own exception does not surface the
        # outer exception currently being checked.
        return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return


def _handler_surfaces_exception(node: ast.ExceptHandler) -> bool:
    visitor = _ExceptionSurfaceVisitor(node.name)
    for statement in node.body:
        visitor.visit(statement)
        if visitor.surfaced:
            return True
    return False


def _check_silent_except(tree: ast.Module, path: str) -> list[Issue]:
    """AM001: broad exception handlers must log, report, or re-raise."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = _exception_type_names(node.type)
        if not caught.intersection({"<bare>", "BaseException", "Exception"}):
            continue
        if not _handler_surfaces_exception(node):
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM001",
                    message=(
                        "broad exception handler without reporting — "
                        "log, surface to the caller, or re-raise"
                    ),
                )
            )
    return issues


def _check_missing_slots(tree: ast.Module, path: str) -> list[Issue]:
    """AM002: @dataclass without slots=True."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for deco in node.decorator_list:
            is_dataclass = False
            has_slots = False

            if isinstance(deco, ast.Name) and deco.id == "dataclass":
                is_dataclass = True
            elif isinstance(deco, ast.Attribute) and deco.attr == "dataclass":
                is_dataclass = True
            elif isinstance(deco, ast.Call):
                func = deco.func
                if isinstance(func, ast.Name) and func.id == "dataclass":
                    is_dataclass = True
                elif isinstance(func, ast.Attribute) and func.attr == "dataclass":
                    is_dataclass = True
                if is_dataclass:
                    for kw in deco.keywords:
                        if kw.arg == "slots":
                            has_slots = True
                            break

            if is_dataclass and not has_slots:
                bases = [
                    b.id
                    if isinstance(b, ast.Name)
                    else b.attr
                    if isinstance(b, ast.Attribute)
                    else ""
                    for b in node.bases
                ]
                if any(
                    b
                    in (
                        "Exception",
                        "BaseException",
                        "ValueError",
                        "RuntimeError",
                        "TypeError",
                        "KeyError",
                    )
                    for b in bases
                ):
                    continue
                issues.append(
                    Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM002",
                        message=f"dataclass {node.name!r} missing slots=True",
                        severity="warning",
                    )
                )
    return issues


def _check_private_in_all(tree: ast.Module, path: str) -> list[Issue]:
    """AM003: _-prefixed names in __all__."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            if (
                                elt.value.startswith("_")
                                and elt.value not in _PUBLIC_MODULE_METADATA_DUNDERS
                            ):
                                issues.append(
                                    Issue(
                                        path=path,
                                        line=elt.lineno,
                                        rule="AM003",
                                        message=f"private name {elt.value!r} in __all__",
                                        severity="warning",
                                    )
                                )
    return issues


def _check_atom_raw_io(tree: ast.Module, path: str, file_path: Path) -> list[Issue]:
    """AM004: open()/subprocess in atom files.

    Atom code is builtin extensions plus contrib extension workspace members
    (everything under ``contrib/extensions/`` is atom-reachable, mirroring the
    load-time validator's package rule). Scenario packages stay out: they mix
    atoms with host-level adapters, and the load-time validator covers their
    mounted atom modules.
    """
    parts = file_path.parts
    is_builtin_atom = (
        "extensions" in parts
        and "builtin" in parts
        and not any(p.startswith("_") for p in parts[-2:] if p != file_path.name)
    )
    is_contrib_atom = "contrib" in parts and "extensions" in parts
    if not (is_builtin_atom or is_contrib_atom):
        return []

    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                issues.append(
                    Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM004",
                        message=(
                            "raw open() in atom — use the resource_writer "
                            "service: api.services.get_role(RESOURCE_WRITER)"
                        ),
                        severity="warning",
                    )
                )
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "subprocess":
                issues.append(
                    Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM004",
                        message=(
                            "subprocess usage in atom — use the operations:bash "
                            "service: api.services.require(BASH_OPERATIONS_SERVICE, "
                            "BashOperations)"
                        ),
                        severity="warning",
                    )
                )
    return issues


def _decorator_call(decorator: ast.expr) -> ast.expr:
    if isinstance(decorator, ast.Call):
        return decorator.func
    return decorator


def _is_cli_command_decorator(decorator: ast.expr) -> bool:
    func = _decorator_call(decorator)
    if isinstance(func, ast.Attribute):
        return func.attr in _CLI_DECORATOR_METHODS
    if isinstance(func, ast.Name):
        return func.id in _CLI_DECORATOR_NAMES
    return False


def _is_cli_command_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _is_cli_command_decorator(decorator) for decorator in node.decorator_list
    )


def _check_param_explosion(tree: ast.Module, path: str) -> list[Issue]:
    """AM005: Functions with >15 parameters."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_cli_command_function(node):
            continue
        n = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
        if node.args.vararg:
            n += 1
        if node.args.kwarg:
            n += 1
        if n > 15:
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM005",
                    message=f"{node.name}() has {n} parameters — consider a config object",
                    severity="warning",
                )
            )
    return issues


def _check_mutable_abi_global(
    tree: ast.Module, path: str, file_path: Path
) -> list[Issue]:
    """AM006: Mutable module-level containers in core/abi/."""
    parts = file_path.parts
    if not ("core" in parts and "abi" in parts):
        return []

    issues: list[Issue] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AnnAssign) and node.target and node.value:
            target = node.target
            if not isinstance(target, ast.Name):
                continue
            if target.id == "__all__":
                continue
            ann_str = ast.dump(node.annotation) if node.annotation else ""
            if any(
                t in ann_str for t in ("Dict", "List", "Set", "dict", "list", "set")
            ):
                if "Final" not in ann_str and "ClassVar" not in ann_str:
                    issues.append(
                        Issue(
                            path=path,
                            line=node.lineno,
                            rule="AM006",
                            message=f"mutable module-level {target.id!r} in ABI — wrap in Final or move to runtime",
                            severity="warning",
                        )
                    )
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if not isinstance(tgt, ast.Name):
                    continue
                if tgt.id == "__all__":
                    continue
                if isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    if tgt.id.startswith("_") and tgt.id.isupper():
                        continue
                    issues.append(
                        Issue(
                            path=path,
                            line=node.lineno,
                            rule="AM006",
                            message=f"mutable module-level {tgt.id!r} in ABI — use Final or move to runtime",
                            severity="warning",
                        )
                    )
    return issues


def _check_god_file(source_lines: list[str], path: str) -> list[Issue]:
    """AM007: Files exceeding 1500 LOC."""
    n = len(source_lines)
    if n > 1500:
        return [
            Issue(
                path=path,
                line=1,
                rule="AM007",
                message=f"file has {n} lines (>1500) — consider splitting",
                severity="warning",
            )
        ]
    return []


def _check_redundant_local_import(tree: ast.Module, path: str) -> list[Issue]:
    """AM008: Function-body import duplicating a module-level import."""
    toplevel_names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                toplevel_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                toplevel_names.add(alias.asname or alias.name)

    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if name in toplevel_names:
                        issues.append(
                            Issue(
                                path=path,
                                line=child.lineno,
                                rule="AM008",
                                message=f"redundant local import of {name!r} — already at module level",
                                severity="warning",
                            )
                        )
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    name = alias.asname or alias.name
                    if name in toplevel_names:
                        issues.append(
                            Issue(
                                path=path,
                                line=child.lineno,
                                rule="AM008",
                                message=f"redundant local import of {name!r} — already at module level",
                                severity="warning",
                            )
                        )
    return issues


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Return whether ``node`` has a decorator named ``name``."""

    for decorator in node.decorator_list:
        match decorator:
            case ast.Name(id=decorator_name):
                if decorator_name == name:
                    return True
            case ast.Attribute(attr=decorator_name):
                if decorator_name == name:
                    return True
            case ast.Call(func=ast.Name(id=decorator_name)):
                if decorator_name == name:
                    return True
            case ast.Call(func=ast.Attribute(attr=decorator_name)):
                if decorator_name == name:
                    return True
    return False


def _base_name(expr: ast.expr) -> str | None:
    match expr:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=name):
            return name
        case ast.Subscript(value=value):
            return _base_name(value)
    return None


def _is_protocol_class(node: ast.ClassDef) -> bool:
    return any(_base_name(base) == "Protocol" for base in node.bases)


def _check_god_class(tree: ast.Module, path: str) -> list[Issue]:
    """AM009: Concrete classes with >25 concrete methods."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if _is_protocol_class(node):
            continue
        method_count = sum(
            1
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not _has_decorator(child, "overload")
            and not _has_decorator(child, "property")
        )
        if method_count > 25:
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM009",
                    message=f"class {node.name!r} has {method_count} methods (>25) — consider splitting",
                    severity="warning",
                )
            )
    return issues


_LAYER_RULES: Final[list[tuple[str, str, str]]] = [
    (
        "gateway/",
        "agentm.extensions.builtin",
        "gateway must not import from builtin atoms",
    ),
    ("extensions/", "agentm.gateway", "extensions must not import from gateway"),
    (
        "extensions/",
        "agentm.core.runtime",
        "extensions must depend on core ABI/lib ports, not runtime",
    ),
    ("cli/", "agentm.core.runtime", "CLI presenters must not import runtime internals"),
    ("authoring/", "agentm.presenter", "authoring must not import from presenter"),
    (
        "presenter/",
        "agentm.core.runtime",
        "presenters must depend on core ABI/lib ports, not runtime",
    ),
    ("core/abi/", "agentm.core.runtime", "ABI must not import from runtime"),
    ("core/abi/", "agentm.core._internal", "ABI must not import from _internal"),
    ("core/", "agentm.authoring", "core must not import authoring"),
    ("core/", "agentm.cli", "core must not import CLI presenters"),
    ("core/", "agentm.config", "core must not import host config resolution"),
    ("core/", "agentm.environments", "core must not import environment backends"),
    ("core/", "agentm.execution", "core must not import execution backends"),
    ("core/", "agentm.extensions", "core must not import extension implementations"),
    ("core/", "agentm.gateway", "core must not import gateway hosts"),
    ("core/", "agentm.observability", "core must not import observability backends"),
    ("core/", "agentm.scenarios", "core must not import scenario loaders"),
    ("core/", "agentm.sdk", "core must not import the SDK presenter"),
    ("core/", "agentm.storage", "core must not import storage backends"),
    (
        "environments/",
        "agentm.core.runtime",
        "environment backends must depend on core ABI/lib ports, not runtime",
    ),
    (
        "execution/",
        "agentm.core.runtime",
        "execution backends must depend on core ABI/lib ports, not runtime",
    ),
    (
        "storage/",
        "agentm.core.runtime",
        "storage backends must depend on core ABI/lib ports, not runtime",
    ),
]


def _check_cross_layer_import(
    tree: ast.Module, path: str, file_path: Path
) -> list[Issue]:
    """AM010: Imports that violate the layer hierarchy."""
    issues: list[Issue] = []
    path_str = str(file_path)

    for src_prefix, forbidden_target, msg in _LAYER_RULES:
        if src_prefix not in path_str:
            continue
        for node in ast.walk(tree):
            module_str: str | None = None
            lineno = 0
            if isinstance(node, ast.ImportFrom) and node.module:
                module_str = node.module
                lineno = node.lineno
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not _module_matches(alias.name, forbidden_target):
                        continue
                    if _find_parent_if(tree, node) is not None:
                        continue
                    issues.append(
                        Issue(
                            path=path,
                            line=node.lineno,
                            rule="AM010",
                            message=f"cross-layer import: {msg} ({alias.name})",
                        )
                    )
                continue
            if module_str is None:
                continue
            if _module_matches(module_str, forbidden_target):
                if _find_parent_if(tree, node) is not None:
                    continue
                issues.append(
                    Issue(
                        path=path,
                        line=lineno,
                        rule="AM010",
                        message=f"cross-layer import: {msg} ({module_str})",
                    )
                )
    return issues


def _module_matches(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(f"{forbidden}.")


def _find_parent_if(tree: ast.Module, target: ast.AST) -> ast.If | None:
    """Return the ``if TYPE_CHECKING:`` node wrapping *target*, or None."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            for child in ast.walk(node):
                if child is target:
                    return node
    return None


def _is_dict_with_type_object(node: ast.expr) -> bool:
    """True if *node* is a dict literal containing ``"type": "object"``."""
    if not isinstance(node, ast.Dict):
        return False
    for key, value in zip(node.keys, node.values, strict=True):
        if (
            isinstance(key, ast.Constant)
            and key.value == "type"
            and isinstance(value, ast.Constant)
            and value.value == "object"
        ):
            return True
    return False


def _collect_dict_schema_names(tree: ast.Module) -> set[str]:
    """Return names of module-level constants assigned a ``{"type": "object", ...}`` dict."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_dict_with_type_object(
                    node.value
                ):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None and _is_dict_with_type_object(node.value):
                names.add(node.target.id)
    return names


def _returned_schema_dict_names(node: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            if not _is_dict_with_type_object(statement.value):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
            and _is_dict_with_type_object(statement.value)
        ):
            names.add(statement.target.id)
    return names


class _SchemaReturnVisitor(ast.NodeVisitor):
    """Find schema returns without descending into nested definitions."""

    def __init__(self, local_schema_names: set[str]) -> None:
        self.returns_schema = False
        self._local_schema_names = local_schema_names

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        value = node.value
        if value is None:
            return
        if _is_dict_with_type_object(value):
            self.returns_schema = True
        elif isinstance(value, ast.Name) and value.id in self._local_schema_names:
            self.returns_schema = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return


def _returns_dict_schema(node: ast.FunctionDef) -> bool:
    """Return whether a helper function returns a hand-written tool schema."""
    local_schema_names = _returned_schema_dict_names(node)
    visitor = _SchemaReturnVisitor(local_schema_names)
    for statement in node.body:
        visitor.visit(statement)
        if visitor.returns_schema:
            return True
    return False


def _collect_dict_schema_factory_names(tree: ast.Module) -> set[str]:
    """Return module-level helpers that return a ``{"type": "object", ...}`` dict."""
    return {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef) and _returns_dict_schema(node)
    }


def _is_schema_factory_call(
    node: ast.expr,
    schema_factory_names: set[str],
) -> bool:
    return (
        isinstance(node, ast.Call)
        and (func_name := _call_name(node)) is not None
        and func_name in schema_factory_names
    )


def _target_named_parameters(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "parameters") or (
        isinstance(node, ast.Attribute) and node.attr == "parameters"
    )


def _declares_atom(tree: ast.Module) -> bool:
    has_manifest = False
    has_install = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "MANIFEST"
                for target in targets
            ):
                has_manifest = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == "install"
        ):
            has_install = True
    return has_manifest and has_install


def _check_hand_written_schema(
    tree: ast.Module, path: str, file_path: Path
) -> list[Issue]:
    """AM011: Tool parameters as dict/factory instead of pydantic schema."""
    if not _declares_atom(tree):
        return []

    schema_names = _collect_dict_schema_names(tree)
    schema_factory_names = _collect_dict_schema_factory_names(tree)
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(_target_named_parameters(target) for target in targets):
                continue
            if _is_dict_with_type_object(value):
                issues.append(
                    Issue(
                        path=path,
                        line=value.lineno,
                        rule="AM011",
                        message=(
                            "hand-written tool schema — use a Pydantic model or "
                            "pydantic_to_tool_schema(Model) instead"
                        ),
                        severity="warning",
                    )
                )
            elif isinstance(value, ast.Name) and value.id in schema_names:
                issues.append(
                    Issue(
                        path=path,
                        line=value.lineno,
                        rule="AM011",
                        message=(
                            f"hand-written tool schema ({value.id}) — use a "
                            "Pydantic model or pydantic_to_tool_schema(Model) "
                            "instead"
                        ),
                        severity="warning",
                    )
                )
            elif _is_schema_factory_call(value, schema_factory_names):
                issues.append(
                    Issue(
                        path=path,
                        line=value.lineno,
                        rule="AM011",
                        message=(
                            "hand-written tool schema factory — use a Pydantic "
                            "model or pydantic_to_tool_schema(Model) instead"
                        ),
                        severity="warning",
                    )
                )
            continue

        if not isinstance(node, ast.Call):
            continue
        is_function_tool = _call_name(node) == "FunctionTool"
        for kw in node.keywords:
            if kw.arg != "parameters":
                continue
            if isinstance(kw.value, ast.Dict) and (
                is_function_tool or _is_dict_with_type_object(kw.value)
            ):
                issues.append(
                    Issue(
                        path=path,
                        line=kw.value.lineno,
                        rule="AM011",
                        message=(
                            "hand-written tool schema — use a Pydantic model or "
                            "pydantic_to_tool_schema(Model) instead"
                        ),
                        severity="warning",
                    )
                )
            elif isinstance(kw.value, ast.Name) and kw.value.id in schema_names:
                issues.append(
                    Issue(
                        path=path,
                        line=kw.value.lineno,
                        rule="AM011",
                        message=(
                            f"hand-written tool schema ({kw.value.id}) — use a "
                            "Pydantic model or pydantic_to_tool_schema(Model) "
                            "instead"
                        ),
                        severity="warning",
                    )
                )
            elif _is_schema_factory_call(kw.value, schema_factory_names):
                issues.append(
                    Issue(
                        path=path,
                        line=kw.value.lineno,
                        rule="AM011",
                        message=(
                            "hand-written tool schema factory — use a Pydantic "
                            "model or pydantic_to_tool_schema(Model) instead"
                        ),
                        severity="warning",
                    )
                )
    return issues


# Core typed contracts that must be constructed with explicit fields, never
# assembled from an untyped ``**dict``. Deliberately excludes vendor SDK
# constructors and pydantic ``super().__init__`` — those APIs are kwargs-based
# by design and have no typed-field alternative.
_CONFIG_DICT_SPLAT_CALLEES: Final[frozenset[str]] = frozenset(
    {"AgentSessionConfig", "FunctionTool", "spawn_child_session"}
)


def _check_config_dict_splat(tree: ast.Module, path: str) -> list[Issue]:
    """AM012: ``**dict`` splat into a core typed contract."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name not in _CONFIG_DICT_SPLAT_CALLEES:
            continue
        # ``**x`` in a call is a keyword whose ``arg`` is None.
        if any(kw.arg is None for kw in node.keywords):
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM012",
                    message=(
                        f"dict-splat into {name}() — pass explicit typed fields, "
                        "not **dict"
                    ),
                )
            )
    return issues


def _check_legacy_asyncio_timeout_error(tree: ast.Module, path: str) -> list[Issue]:
    """AM013: ``asyncio.TimeoutError`` catch on Python 3.12+."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        exception_types = (
            list(node.type.elts) if isinstance(node.type, ast.Tuple) else [node.type]
        )
        for exc_type in exception_types:
            if (
                isinstance(exc_type, ast.Attribute)
                and exc_type.attr == "TimeoutError"
                and isinstance(exc_type.value, ast.Name)
                and exc_type.value.id == "asyncio"
            ):
                issues.append(
                    Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM013",
                        message=(
                            "catch builtin TimeoutError instead of "
                            "asyncio.TimeoutError on Python 3.12+"
                        ),
                    )
                )
                break
    return issues


def _is_resolve_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resolve"
    )


def _check_resolved_parent_chain(tree: ast.Module, path: str) -> list[Issue]:
    """AM014: ``path.resolve().parent`` / ``path.resolve().parents[...]``."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            if _is_resolve_call(node.value):
                issues.append(
                    Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM014",
                        message=(
                            "avoid path.resolve().parent — split resolve() from "
                            "parent access when symlink resolution is intentional"
                        ),
                    )
                )
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and _is_resolve_call(node.value.value)
        ):
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM014",
                    message=(
                        "avoid path.resolve().parents[...] — split resolve() from "
                        "parent access when symlink resolution is intentional"
                    ),
                )
            )
    return issues


_CONCRETE_ATOM_PREFIX: Final = "agentm.extensions.builtin."
_SCENARIO_EXECUTION_CALLS: Final[frozenset[str]] = frozenset(
    {
        "exec",
        "import_module",
        "module_from_spec",
        "spec_from_file_location",
    }
)
_DYNAMIC_ATTRIBUTE_CALLS: Final[frozenset[str]] = frozenset(
    {"delattr", "getattr", "hasattr", "setattr"}
)
_IGNORE_DIRECTIVE: Final[re.Pattern[str]] = re.compile(
    r"#\s*code-health:\s*ignore\[(?P<rules>[^\]]+)\]"
)
_IGNORE_FILE_DIRECTIVE: Final[re.Pattern[str]] = re.compile(
    r"#\s*code-health:\s*ignore-file\[(?P<rules>[^\]]+)\]"
)


def _path_parts(path: Path) -> tuple[str, ...]:
    return tuple(part.replace("\\", "/") for part in path.parts)


def _contains_parts(path: Path, expected: tuple[str, ...]) -> bool:
    parts = _path_parts(path)
    width = len(expected)
    return any(
        parts[index : index + width] == expected
        for index in range(len(parts) - width + 1)
    )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _attribute_named(node: ast.AST | None, name: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == name


def _tuple_target(node: ast.AST | None) -> bool:
    return isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) > 1


def _check_core_atom_policy(
    tree: ast.Module,
    path: str,
    file_path: Path,
) -> list[Issue]:
    """AM017: core runtime must not append concrete atom implementations."""
    if not _contains_parts(file_path, ("core", "runtime")):
        return []
    return [
        Issue(
            path=path,
            line=node.lineno,
            rule="AM017",
            message=(
                "core runtime names a concrete builtin atom; put default "
                "composition in a host scenario manifest"
            ),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(_CONCRETE_ATOM_PREFIX)
    ]


def _check_scenario_loader_execution(
    tree: ast.Module,
    path: str,
    file_path: Path,
) -> list[Issue]:
    """AM018: scenario discovery returns data and never executes Python."""
    if not file_path.as_posix().endswith("src/agentm/scenarios.py"):
        return []
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in _SCENARIO_EXECUTION_CALLS:
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM018",
                    message=(
                        "scenario loader executes Python; return a durable "
                        "ExtensionSpec and let the runtime loader validate it"
                    ),
                )
            )
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr == "modules"
        ):
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM018",
                    message=(
                        "scenario loader mutates sys.modules; represent local "
                        "code as a content-addressed file source"
                    ),
                )
            )
    return issues


def _check_legacy_extension_shape(
    tree: ast.Module,
    path: str,
) -> list[Issue]:
    """AM019: canonical ExtensionSpec values are not tuples or indexable pairs."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and _attribute_named(node.value, "provider")
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM019",
                    message=(
                        "ResolvedSessionSpec.provider is an ExtensionSpec; "
                        "read source/config fields instead of tuple indexes"
                    ),
                )
            )
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if _tuple_target(node.target) and _attribute_named(
                node.iter,
                "extensions",
            ):
                issues.append(
                    Issue(
                        path=path,
                        line=node.target.lineno,
                        rule="AM019",
                        message=(
                            "resolved extensions are ExtensionSpec values; "
                            "do not unpack them as (module, config)"
                        ),
                    )
                )
        if isinstance(node, ast.comprehension):
            if _tuple_target(node.target) and _attribute_named(
                node.iter,
                "extensions",
            ):
                issues.append(
                    Issue(
                        path=path,
                        line=node.target.lineno,
                        rule="AM019",
                        message=(
                            "resolved extensions are ExtensionSpec values; "
                            "do not unpack them as (module, config)"
                        ),
                    )
                )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_tuple_target(target) for target in targets) and _attribute_named(
                node.value, "provider"
            ):
                issues.append(
                    Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM019",
                        message=(
                            "ResolvedSessionSpec.provider is an ExtensionSpec; "
                            "do not unpack it as (module, config)"
                        ),
                    )
                )
    return issues


def _check_raw_cli_extension_config(
    tree: ast.Module,
    path: str,
    file_path: Path,
) -> list[Issue]:
    """AM020: presenter output must not directly materialize secret-bearing config."""
    if not _contains_parts(file_path, ("agentm", "cli")):
        return []
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "dict":
            continue
        if not node.args or not _attribute_named(node.args[0], "config"):
            continue
        issues.append(
            Issue(
                path=path,
                line=node.lineno,
                rule="AM020",
                message=(
                    "CLI materializes raw extension config; use redact_config() "
                    "before rendering or serializing it"
                ),
            )
        )
    return issues


def _check_dynamic_attribute_access(
    tree: ast.Module,
    path: str,
) -> list[Issue]:
    """AM021: dynamic attribute lookup obscures typed contracts."""
    return [
        Issue(
            path=path,
            line=node.lineno,
            rule="AM021",
            message=(
                f"{_call_name(node)}() bypasses a typed contract; use an "
                "explicit protocol, field, or dispatch table"
            ),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) in _DYNAMIC_ATTRIBUTE_CALLS
    ]


def _check_typing_any(
    tree: ast.Module,
    path: str,
) -> list[Issue]:
    """AM022: source annotations must not erase values to typing.Any."""
    return [
        Issue(
            path=path,
            line=node.lineno,
            rule="AM022",
            message=(
                "typing.Any erases the typed contract; use object, JsonValue, "
                "a concrete DTO, or a Protocol"
            ),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "Any"
    ]


def _annotation_nodes(tree: ast.Module) -> list[ast.expr]:
    annotations: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            annotations.extend(
                arg.annotation for arg in args if arg.annotation is not None
            )
            if node.args.vararg is not None and node.args.vararg.annotation:
                annotations.append(node.args.vararg.annotation)
            if node.args.kwarg is not None and node.args.kwarg.annotation:
                annotations.append(node.args.kwarg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
    return annotations


def _bare_dict_names(annotation: ast.expr) -> list[ast.Name]:
    parents = {
        id(child): parent
        for parent in ast.walk(annotation)
        for child in ast.iter_child_nodes(parent)
    }
    return [
        node
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name)
        and node.id == "dict"
        and not (
            isinstance((parent := parents.get(id(node))), ast.Subscript)
            and parent.value is node
        )
    ]


def _check_bare_dict(
    tree: ast.Module,
    path: str,
) -> list[Issue]:
    """AM023: annotations must not contain an unparameterized dict."""
    return [
        Issue(
            path=path,
            line=node.lineno,
            rule="AM023",
            message=(
                "bare dict erases key/value contracts; parameterize it or "
                "use Mapping, a dataclass, or a Protocol"
            ),
        )
        for annotation in _annotation_nodes(tree)
        for node in _bare_dict_names(annotation)
    ]


def _check_stdlib_logging(tree: ast.Module, path: str) -> list[Issue]:
    """AM024: source code must use Loguru rather than stdlib logging."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging" or alias.name.startswith("logging."):
                    issues.append(
                        Issue(
                            path=path,
                            line=node.lineno,
                            rule="AM024",
                            message=(
                                "stdlib logging is prohibited; use "
                                "'from loguru import logger'"
                            ),
                        )
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "logging" or node.module.startswith("logging."))
        ):
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM024",
                    message=(
                        "stdlib logging is prohibited; use 'from loguru import logger'"
                    ),
                )
            )
    return issues


def _check_runtime_type_check(tree: ast.Module, path: str) -> list[Issue]:
    """AM025: isinstance() should not replace typed contracts."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM025",
                    message=(
                        "isinstance() branches on runtime shape; prefer a typed "
                        "contract, DTO, Protocol, or dispatch table"
                    ),
                )
            )
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "isinstance"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "builtins"
        ):
            issues.append(
                Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM025",
                    message=(
                        "builtins.isinstance() branches on runtime shape; prefer "
                        "a typed contract, DTO, Protocol, or dispatch table"
                    ),
                )
            )
    return issues


def _directive_rules(line: str, pattern: re.Pattern[str]) -> frozenset[str]:
    match = pattern.search(line)
    if match is None:
        return frozenset()
    return frozenset(
        item.strip() for item in match.group("rules").split(",") if item.strip()
    )


def _suppressed(issue: Issue, source_lines: list[str]) -> bool:
    line_index = issue.line - 1
    if 0 <= line_index < len(source_lines):
        line = source_lines[line_index]
        if "# type: ignore" in line or "# noqa" in line:
            return True
        if issue.rule in _directive_rules(line, _IGNORE_DIRECTIVE):
            return True
    return any(
        issue.rule in _directive_rules(line, _IGNORE_FILE_DIRECTIVE)
        for line in source_lines
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_RULES: Final[tuple[str, ...]] = (
    "AM001",
    "AM002",
    "AM003",
    "AM004",
    "AM005",
    "AM006",
    "AM007",
    "AM008",
    "AM009",
    "AM010",
    "AM011",
    "AM012",
    "AM013",
    "AM014",
    "AM017",
    "AM018",
    "AM019",
    "AM020",
    "AM021",
    "AM022",
    "AM023",
    "AM024",
    "AM025",
)


def check_file(file_path: Path) -> list[Issue]:
    """Run all rules on one Python file. Returns issues found."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    source_lines = source.splitlines()
    rel = str(file_path)

    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return [
            Issue(
                path=rel,
                line=exc.lineno or 1,
                rule="AM000",
                message=f"cannot parse Python source: {exc.msg}",
            )
        ]

    issues: list[Issue] = []
    issues.extend(_check_silent_except(tree, rel))
    issues.extend(_check_missing_slots(tree, rel))
    issues.extend(_check_private_in_all(tree, rel))
    issues.extend(_check_atom_raw_io(tree, rel, file_path))
    issues.extend(_check_param_explosion(tree, rel))
    issues.extend(_check_mutable_abi_global(tree, rel, file_path))
    issues.extend(_check_god_file(source_lines, rel))
    issues.extend(_check_redundant_local_import(tree, rel))
    issues.extend(_check_god_class(tree, rel))
    issues.extend(_check_cross_layer_import(tree, rel, file_path))
    issues.extend(_check_hand_written_schema(tree, rel, file_path))
    issues.extend(_check_config_dict_splat(tree, rel))
    issues.extend(_check_legacy_asyncio_timeout_error(tree, rel))
    issues.extend(_check_resolved_parent_chain(tree, rel))
    issues.extend(_check_core_atom_policy(tree, rel, file_path))
    issues.extend(_check_scenario_loader_execution(tree, rel, file_path))
    issues.extend(_check_legacy_extension_shape(tree, rel))
    issues.extend(_check_raw_cli_extension_config(tree, rel, file_path))
    issues.extend(_check_dynamic_attribute_access(tree, rel))
    issues.extend(_check_typing_any(tree, rel))
    issues.extend(_check_bare_dict(tree, rel))
    issues.extend(_check_stdlib_logging(tree, rel))
    issues.extend(_check_runtime_type_check(tree, rel))
    return [issue for issue in issues if not _suppressed(issue, source_lines)]


def check_paths(paths: list[Path]) -> list[Issue]:
    """Run checks on a list of files or directories."""
    all_issues: list[Issue] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            all_issues.extend(check_file(p))
        elif p.is_dir():
            for py in sorted(p.rglob("*.py")):
                if "__pycache__" in py.parts:
                    continue
                all_issues.extend(check_file(py))
    return all_issues


def changed_files(base: str = "origin/main") -> list[Path]:
    """Return Python files changed since merge-base with *base*."""
    try:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", base],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        merge_base = base

    try:
        diff_output = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", merge_base],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    return [
        Path(f)
        for f in diff_output.splitlines()
        if f.endswith(".py") and Path(f).exists()
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="lint",
    help="Project-level code health checks (AST-based).",
    context_settings={"allow_interspersed_args": True},
)


@app.callback(invoke_without_command=True)
def lint_cmd(
    paths: list[str] | None = typer.Argument(
        None,
        help="Files or directories to check. Default: src/agentm/",
    ),
    changed: bool = typer.Option(
        False,
        "--changed",
        "-c",
        help="Check only files changed since merge-base with main.",
    ),
    rule: list[str] | None = typer.Option(
        None,
        "--rule",
        "-r",
        help="Run only specific rules (e.g. -r AM001 -r AM002).",
    ),
    warnings_as_errors: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Treat warnings as errors.",
    ),
) -> None:
    """Run project-specific code health checks."""
    if changed:
        files = changed_files()
        if not files:
            typer.echo("No changed Python files.", err=True)
            raise typer.Exit(0)
        issues = check_paths(files)
    elif paths:
        issues = check_paths([Path(p) for p in paths])
    else:
        issues = check_paths([Path("src/agentm")])

    if rule:
        rule_set = set(rule)
        unknown = rule_set - set(ALL_RULES)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise typer.BadParameter(f"unknown code-health rule(s): {names}")
        issues = [i for i in issues if i.rule in rule_set]

    if not issues:
        typer.echo("No code health issues found.", err=True)
        raise typer.Exit(0)

    issues.sort(key=lambda i: (i.path, i.line))

    for issue in issues:
        severity_mark = "E" if issue.severity == "error" else "W"
        typer.echo(
            f"{issue.path}:{issue.line}: {issue.rule} [{severity_mark}] {issue.message}"
        )

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity != "error"]
    if warnings_as_errors:
        errors.extend(warnings)
        warnings = []

    summary_parts: list[str] = []
    if errors:
        summary_parts.append(f"{len(errors)} error(s)")
    if warnings:
        summary_parts.append(f"{len(warnings)} warning(s)")
    typer.echo(
        f"Code health: {', '.join(summary_parts)}",
        err=True,
    )

    raise typer.Exit(1 if errors else 0)


if __name__ == "__main__":
    app()
