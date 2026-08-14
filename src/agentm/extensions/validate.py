# code-health: ignore-file[AM025] -- extension validator checks dynamically loaded atom objects
"""Minimal load-time atom contract validator."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Severity = Literal["error", "warning"]

_FORBIDDEN_IMPORTS: dict[str, str] = {
    "agentm.core.runtime": (
        "atoms must use agentm.core.abi/agentm.core.lib surfaces instead of "
        "runtime internals"
    ),
    "agentm.core._internal": (
        "atoms must not depend on constitution-private internals"
    ),
    "agentm.sdk": (
        "atoms must not depend on the presenter layer; spawn sessions through "
        "api.spawn_child_session and expose capabilities as services"
    ),
    "agentm.cli": ("atoms must not depend on the CLI presenter"),
    "agentm.gateway": ("atoms must not depend on the gateway presenter"),
    "agentm.presenter": (
        "atoms must not depend on the presenter layer; shared view libraries "
        "live outside it (e.g. agentm.trajectory_view)"
    ),
    "agentm.extensions.builtin": (
        "atom-to-atom coupling is forbidden; communicate through AtomAPI "
        "services, events, or explicit extension dependencies"
    ),
    "_agentm_contrib__": ("contrib atoms must stay decoupled from each other"),
    "agentm._scenarios": ("scenario-local atom-to-atom coupling is forbidden"),
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    rule: str
    message: str
    path: str
    line: int = 0


def validate_atom_file(
    path: str | Path,
    *,
    atom_package: str | None = None,
    module_name: str | None = None,
) -> list[ValidationIssue]:
    """Validate one atom source file.

    ``atom_package`` is the dotted name of the package this file belongs to
    when the atom is a package. Imports within that package are the atom's own
    parts, so they are allowed where an import of a different atom is not.

    ``module_name`` is this file's own dotted name, used to resolve relative
    imports before checking them. Without it a relative import that escapes the
    package is unreadable and passes unchecked.
    """

    src_path = Path(path)
    try:
        tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
    except SyntaxError as exc:
        return [
            ValidationIssue(
                severity="error",
                rule="syntax",
                message=str(exc),
                path=str(src_path),
                line=exc.lineno or 0,
            )
        ]
    except OSError as exc:
        return [
            ValidationIssue(
                severity="error",
                rule="read",
                message=str(exc),
                path=str(src_path),
            )
        ]

    issues: list[ValidationIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _record_forbidden_import(
                    issues, alias.name, src_path, node.lineno, atom_package
                )
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(
                node.module or "",
                node.level,
                module_name=module_name,
                is_init=src_path.name == "__init__.py",
            )
            _record_forbidden_import(
                issues, module, src_path, node.lineno, atom_package
            )
            if module == "agentm.core":
                for alias in node.names:
                    _record_forbidden_import(
                        issues,
                        f"agentm.core.{alias.name}",
                        src_path,
                        node.lineno,
                        atom_package,
                    )
    return issues


def validate_atom_package(
    package_dir: str | Path,
    *,
    atom_package: str | None = None,
) -> list[ValidationIssue]:
    """Validate every Python file in a package atom."""

    root = Path(package_dir)
    issues: list[ValidationIssue] = []
    for path in sorted(root.rglob("*.py")):
        issues.extend(
            validate_atom_file(
                path,
                atom_package=atom_package,
                module_name=_module_name_of(path, root, atom_package),
            )
        )
    return issues


def _module_name_of(path: Path, root: Path, atom_package: str | None) -> str | None:
    """Dotted name of one file inside a package rooted at ``atom_package``."""

    if atom_package is None:
        return None
    parts = list(path.relative_to(root).parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join([atom_package, *parts])


def _absolute_module(
    module: str,
    level: int,
    *,
    module_name: str | None,
    is_init: bool,
) -> str:
    """Resolve a relative import to the dotted name it actually names."""

    if level == 0 or module_name is None:
        return module
    package = module_name if is_init else module_name.rpartition(".")[0]
    parts = package.split(".") if package else []
    climb = level - 1
    parts = parts[: len(parts) - climb] if climb <= len(parts) else []
    base = ".".join(parts)
    if not base:
        return module
    return f"{base}.{module}" if module else base


def extension_helper_imports(path: str | Path) -> list[str]:
    """Return extension helper modules imported by one atom source file."""

    src_path = Path(path)
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_extension_helper_module(alias.name):
                    modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_extension_helper_module(module):
                modules.add(module)
    return sorted(modules)


def _is_extension_helper_module(module: str) -> bool:
    if module == "agentm.extensions":
        return False
    if module.startswith("agentm.extensions.builtin"):
        return False
    return module.startswith("agentm.extensions.")


def _record_forbidden_import(
    issues: list[ValidationIssue],
    module: str,
    path: Path,
    line: int,
    atom_package: str | None = None,
) -> None:
    reason = _forbidden_import_reason(module, atom_package)
    if reason is None:
        return
    issues.append(
        ValidationIssue(
            severity="error",
            rule="forbidden-import",
            message=f"atom imports {module!r}; {reason}",
            path=str(path),
            line=line,
        )
    )


def _forbidden_import_reason(module: str, atom_package: str | None) -> str | None:
    if atom_package is not None and (
        module == atom_package or module.startswith(f"{atom_package}.")
    ):
        return None
    for forbidden, reason in _FORBIDDEN_IMPORTS.items():
        if module == forbidden or module.startswith(f"{forbidden}."):
            return reason
    return None


__all__ = [
    "ValidationIssue",
    "extension_helper_imports",
    "validate_atom_file",
    "validate_atom_package",
]
