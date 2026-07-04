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
- **AM003** ``private-in-all``: ``_``-prefixed names in ``__all__``. Either
  the name is public (drop the underscore) or private (drop from ``__all__``).
- **AM004** ``atom-raw-io``: ``open()`` / ``subprocess`` calls inside atom
  files. Atoms must use ``Operations`` (file/bash) so sandbox isolation works.
- **AM005** ``param-explosion``: Functions with >15 parameters. Signals a
  config-object extraction opportunity.
- **AM006** ``mutable-abi-global``: Mutable containers (``dict``, ``list``,
  ``set``) at module level in ``core/abi/`` without ``Final`` annotation.
  ABI should be definitions-only; mutable state belongs in runtime.
- **AM007** ``god-file``: Files exceeding 1500 LOC. Warning-only signal
  for potential split candidates.
- **AM008** ``redundant-local-import``: A function-body import that
  duplicates a module-level import. Inconsistent style with no benefit.
- **AM009** ``god-class``: Classes with >25 methods. Swiss-army-knife
  signal — consider splitting into composable facades.
- **AM010** ``cross-layer-import``: Imports that violate the layer
  hierarchy (e.g. gateway → extensions/builtin, extensions → gateway,
  abi → runtime). Extends §11 to cover the full dependency graph.
- **AM011** ``hand-written-schema``: Tool ``parameters`` defined as a
  dict literal instead of ``pydantic_to_tool_schema(Model)``. Pydantic
  schemas are the project convention — they stay in sync with
  validation, generate descriptions from ``Field()``, and avoid
  hand-maintained JSON Schema boilerplate.
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
from loguru import logger


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

_LOG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\blog(?:ger|ging)?\b|\bwarn(?:ing)?\b|\braise\b|\bprint\b", re.IGNORECASE
)

_ATOM_BUILTIN_PARTS: Final[tuple[str, str]] = ("extensions", "builtin")


def _is_atom_file(path: Path) -> bool:
    parts = path.parts
    return any(
        parts[index : index + len(_ATOM_BUILTIN_PARTS)] == _ATOM_BUILTIN_PARTS
        for index in range(len(parts) - len(_ATOM_BUILTIN_PARTS) + 1)
    )


def _check_silent_except(
    tree: ast.Module, source_lines: list[str], path: str
) -> list[Issue]:
    """AM001: except Exception without logging."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        type_name = ""
        if isinstance(node.type, ast.Name):
            type_name = node.type.id
        elif isinstance(node.type, ast.Attribute):
            type_name = node.type.attr
        if type_name != "Exception":
            continue

        handler_lines = []
        for child in ast.walk(node):
            if hasattr(child, "lineno") and child is not node:
                start = child.lineno - 1
                end = getattr(child, "end_lineno", child.lineno)
                handler_lines.extend(source_lines[start:end])

        if not handler_lines:
            body_start = node.lineno
            body_end = min(node.lineno + 5, len(source_lines))
            handler_lines = source_lines[body_start:body_end]

        handler_text = "\n".join(handler_lines)
        if not _LOG_PATTERN.search(handler_text):
            issues.append(Issue(
                path=path,
                line=node.lineno,
                rule="AM001",
                message="except Exception without logging — add logger.debug/warning or re-raise",
            ))
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
                    b.id if isinstance(b, ast.Name) else
                    b.attr if isinstance(b, ast.Attribute) else ""
                    for b in node.bases
                ]
                if any(b in ("Exception", "BaseException", "ValueError",
                             "RuntimeError", "TypeError", "KeyError")
                       for b in bases):
                    continue
                issues.append(Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM002",
                    message=f"dataclass {node.name!r} missing slots=True",
                    severity="warning",
                ))
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
                            if elt.value.startswith("_"):
                                issues.append(Issue(
                                    path=path,
                                    line=elt.lineno,
                                    rule="AM003",
                                    message=f"private name {elt.value!r} in __all__",
                                    severity="warning",
                                ))
    return issues


def _check_atom_raw_io(
    tree: ast.Module, path: str, file_path: Path
) -> list[Issue]:
    """AM004: open()/subprocess in atom files."""
    parts = file_path.parts
    is_atom = ("extensions" in parts and "builtin" in parts
               and not any(p.startswith("_") for p in parts[-2:] if p != file_path.name))
    if not is_atom:
        return []

    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                issues.append(Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM004",
                    message="raw open() in atom — use api.get_resource_writer() instead",
                    severity="warning",
                ))
        if isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Name)
                    and node.value.id == "subprocess"):
                issues.append(Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM004",
                    message="subprocess usage in atom — use api.get_operations().bash instead",
                    severity="warning",
                ))
    return issues


def _check_param_explosion(tree: ast.Module, path: str) -> list[Issue]:
    """AM005: Functions with >15 parameters."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        n = (len(node.args.args) + len(node.args.posonlyargs)
             + len(node.args.kwonlyargs))
        if node.args.vararg:
            n += 1
        if node.args.kwarg:
            n += 1
        if n > 15:
            issues.append(Issue(
                path=path,
                line=node.lineno,
                rule="AM005",
                message=f"{node.name}() has {n} parameters — consider a config object",
                severity="warning",
            ))
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
            if any(t in ann_str for t in ("Dict", "List", "Set", "dict", "list", "set")):
                if "Final" not in ann_str and "ClassVar" not in ann_str:
                    issues.append(Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM006",
                        message=f"mutable module-level {target.id!r} in ABI — wrap in Final or move to runtime",
                        severity="warning",
                    ))
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if not isinstance(tgt, ast.Name):
                    continue
                if tgt.id == "__all__":
                    continue
                if isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    if tgt.id.startswith("_") and tgt.id.isupper():
                        continue
                    issues.append(Issue(
                        path=path,
                        line=node.lineno,
                        rule="AM006",
                        message=f"mutable module-level {tgt.id!r} in ABI — use Final or move to runtime",
                        severity="warning",
                    ))
    return issues


def _check_god_file(source_lines: list[str], path: str) -> list[Issue]:
    """AM007: Files exceeding 1500 LOC."""
    n = len(source_lines)
    if n > 1500:
        return [Issue(
            path=path,
            line=1,
            rule="AM007",
            message=f"file has {n} lines (>1500) — consider splitting",
            severity="warning",
        )]
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
                        issues.append(Issue(
                            path=path,
                            line=child.lineno,
                            rule="AM008",
                            message=f"redundant local import of {name!r} — already at module level",
                            severity="warning",
                        ))
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    name = alias.asname or alias.name
                    if name in toplevel_names:
                        issues.append(Issue(
                            path=path,
                            line=child.lineno,
                            rule="AM008",
                            message=f"redundant local import of {name!r} — already at module level",
                            severity="warning",
                        ))
    return issues


def _check_god_class(tree: ast.Module, path: str) -> list[Issue]:
    """AM009: Classes with >25 methods."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        method_count = sum(
            1 for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if method_count > 25:
            issues.append(Issue(
                path=path,
                line=node.lineno,
                rule="AM009",
                message=f"class {node.name!r} has {method_count} methods (>25) — consider splitting",
                severity="warning",
            ))
    return issues


_LAYER_RULES: Final[list[tuple[str, str, str]]] = [
    ("gateway/", "extensions/builtin/", "gateway must not import from builtin atoms"),
    ("extensions/", "gateway/", "extensions must not import from gateway"),
    ("core/abi/", "core/runtime/", "ABI must not import from runtime"),
    ("core/abi/", "core/_internal/", "ABI must not import from _internal"),
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
                    module_str = alias.name
                lineno = node.lineno
            if module_str is None:
                continue
            target_dotted = forbidden_target.replace("/", ".")
            if target_dotted.rstrip(".") in module_str:
                if "TYPE_CHECKING" in ast.dump(tree):
                    parent = _find_parent_if(tree, node)
                    if parent is not None:
                        continue
                issues.append(Issue(
                    path=path,
                    line=lineno,
                    rule="AM010",
                    message=f"cross-layer import: {msg} ({module_str})",
                ))
    return issues


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
    for key, value in zip(node.keys, node.values):
        if (isinstance(key, ast.Constant) and key.value == "type"
                and isinstance(value, ast.Constant) and value.value == "object"):
            return True
    return False


def _collect_dict_schema_names(tree: ast.Module) -> set[str]:
    """Return names of module-level constants assigned a ``{"type": "object", ...}`` dict."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_dict_with_type_object(node.value):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None and _is_dict_with_type_object(node.value):
                names.add(node.target.id)
    return names


def _check_hand_written_schema(
    tree: ast.Module, path: str, file_path: Path
) -> list[Issue]:
    """AM011: Tool parameters as dict literal instead of pydantic schema."""
    parts = file_path.parts
    is_atom = ("extensions" in parts and "builtin" in parts
               and not any(p.startswith("_") for p in parts[-2:] if p != file_path.name))
    if not is_atom:
        return []

    schema_names = _collect_dict_schema_names(tree)
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name != "FunctionTool":
            continue
        for kw in node.keywords:
            if kw.arg != "parameters":
                continue
            if isinstance(kw.value, ast.Dict):
                issues.append(Issue(
                    path=path,
                    line=kw.value.lineno,
                    rule="AM011",
                    message="hand-written tool schema — use pydantic_to_tool_schema(Model) instead",
                    severity="warning",
                ))
            elif (isinstance(kw.value, ast.Name)
                  and kw.value.id in schema_names):
                issues.append(Issue(
                    path=path,
                    line=kw.value.lineno,
                    rule="AM011",
                    message=f"hand-written tool schema ({kw.value.id}) — use pydantic_to_tool_schema(Model) instead",
                    severity="warning",
                ))
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
            issues.append(Issue(
                path=path,
                line=node.lineno,
                rule="AM012",
                message=(
                    f"dict-splat into {name}() — pass explicit typed fields, "
                    "not **dict"
                ),
            ))
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
                issues.append(Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM013",
                    message=(
                        "catch builtin TimeoutError instead of "
                        "asyncio.TimeoutError on Python 3.12+"
                    ),
                ))
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
                issues.append(Issue(
                    path=path,
                    line=node.lineno,
                    rule="AM014",
                    message=(
                        "avoid path.resolve().parent — split resolve() from "
                        "parent access when symlink resolution is intentional"
                    ),
                ))
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and _is_resolve_call(node.value.value)
        ):
            issues.append(Issue(
                path=path,
                line=node.lineno,
                rule="AM014",
                message=(
                    "avoid path.resolve().parents[...] — split resolve() from "
                    "parent access when symlink resolution is intentional"
                ),
            ))
    return issues


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_RULES: Final[tuple[str, ...]] = (
    "AM001", "AM002", "AM003", "AM004", "AM005", "AM006", "AM007",
    "AM008", "AM009", "AM010", "AM011", "AM012", "AM013", "AM014",
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
    except SyntaxError:
        return []

    issues: list[Issue] = []
    issues.extend(_check_silent_except(tree, source_lines, rel))
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
    return issues


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


def changed_files(base: str = "main") -> list[Path]:
    """Return Python files changed since merge-base with *base*."""
    try:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", base],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        merge_base = base

    try:
        diff_output = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", merge_base],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    return [
        Path(f) for f in diff_output.splitlines()
        if f.endswith(".py") and Path(f).exists()
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="lint",
    help="Project-level code health checks (AST-based).",
)


@app.callback(invoke_without_command=True)
def lint_cmd(
    paths: list[str] = typer.Argument(
        None,
        help="Files or directories to check. Default: src/agentm/",
    ),
    changed: bool = typer.Option(
        False, "--changed", "-c",
        help="Check only files changed since merge-base with main.",
    ),
    rule: list[str] = typer.Option(
        None, "--rule", "-r",
        help="Run only specific rules (e.g. -r AM001 -r AM002).",
    ),
    warnings_as_errors: bool = typer.Option(
        False, "--strict", "-s",
        help="Treat warnings as errors.",
    ),
) -> None:
    """Run project-specific code health checks."""
    if changed:
        files = changed_files()
        if not files:
            logger.info("No changed Python files.")
            raise typer.Exit(0)
        issues = check_paths(files)
    elif paths:
        issues = check_paths([Path(p) for p in paths])
    else:
        issues = check_paths([Path("src/agentm")])

    if rule:
        rule_set = set(rule)
        issues = [i for i in issues if i.rule in rule_set]

    if not issues:
        logger.info("No code health issues found.")
        raise typer.Exit(0)

    issues.sort(key=lambda i: (i.path, i.line))

    for issue in issues:
        severity_mark = "E" if issue.severity == "error" else "W"
        print(f"{issue.path}:{issue.line}: {issue.rule} [{severity_mark}] {issue.message}")

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity != "error"]
    if warnings_as_errors:
        errors.extend(warnings)
        warnings = []

    summary_parts = []
    if errors:
        summary_parts.append(f"{len(errors)} error(s)")
    if warnings:
        summary_parts.append(f"{len(warnings)} warning(s)")
    logger.info("Code health: {}", ", ".join(summary_parts))

    raise typer.Exit(1 if errors else 0)


if __name__ == "__main__":
    app()
