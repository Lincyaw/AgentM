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
) -> list[ValidationIssue]:
    """Validate one atom source file."""

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
                _record_forbidden_import(issues, alias.name, src_path, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            _record_forbidden_import(issues, module, src_path, node.lineno)
            if module == "agentm.core":
                for alias in node.names:
                    _record_forbidden_import(
                        issues,
                        f"agentm.core.{alias.name}",
                        src_path,
                        node.lineno,
                    )
    return issues


def validate_atom_package(
    package_dir: str | Path,
) -> list[ValidationIssue]:
    """Validate every Python file in a package atom."""

    root = Path(package_dir)
    issues: list[ValidationIssue] = []
    for path in sorted(root.rglob("*.py")):
        issues.extend(validate_atom_file(path))
    return issues


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
) -> None:
    reason = _forbidden_import_reason(module)
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


def _forbidden_import_reason(module: str) -> str | None:
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
