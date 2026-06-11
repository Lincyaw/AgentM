"""``agentm validate`` subcommand -- check S11 atom contract compliance.

``agentm validate all``
  Discover and validate builtin + contrib + home atoms.

``agentm validate file <path>...``
  Validate specific atom files.

``agentm validate package <path>...``
  Validate specific atom packages (directories with ``__init__.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from agentm.extensions.validate import (
    ValidationIssue,
    validate_atom_file,
    validate_atom_package,
    validate_builtin,
)

app = typer.Typer(
    name="validate",
    help="Validate S11 atom contract compliance.",
    no_args_is_help=True,
)


def _format_issue(issue: ValidationIssue, *, path: str = "") -> str:
    """Format one issue in ruff-style ``path:line: rule message`` form."""
    prefix = f"{path}: " if path else ""
    return f"{prefix}{issue.rule} {issue.message}"


def _print_issues(issues: list[ValidationIssue]) -> int:
    """Print issues and return exit code (0 = clean, 1 = violations)."""
    if not issues:
        print("No violations found.", file=sys.stderr)
        return 0
    for issue in issues:
        severity = "error" if issue.severity == "error" else "warning"
        print(
            f"{issue.module_path}: [{severity}] {issue.rule} {issue.message}",
            file=sys.stdout,
        )
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity != "error"]
    print(
        f"\n{len(errors)} error(s), {len(warnings)} warning(s).",
        file=sys.stderr,
    )
    return 1 if errors else 0


@app.command(name="all")
def validate_all() -> None:
    """Discover and validate all builtin + contrib + home atom packages."""
    all_issues: list[ValidationIssue] = []

    # Builtin atoms (single-file).
    all_issues.extend(validate_builtin())

    # Contrib packages.
    from agentm.extensions.discover import _agentm_repo_root

    repo_root = _agentm_repo_root()
    if repo_root is not None:
        contrib_dir = repo_root / "contrib" / "extensions"
        if contrib_dir.is_dir():
            for child in sorted(contrib_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name.startswith("_") or child.name == "tests":
                    continue
                init = child / "__init__.py"
                if not init.is_file():
                    continue
                module_path = f"contrib.extensions.{child.name}"
                all_issues.extend(
                    validate_atom_package(
                        child, module_path=module_path
                    )
                )

    rc = _print_issues(all_issues)
    raise typer.Exit(code=rc)


@app.command(name="file")
def validate_file_cmd(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Path(s) to atom .py files."),
    ],
) -> None:
    """Validate specific atom files against the S11 contract."""
    all_issues: list[ValidationIssue] = []
    for p in paths:
        resolved = p.resolve()
        if not resolved.is_file():
            print(f"ERROR: {p} is not a file", file=sys.stderr)
            raise typer.Exit(code=2)
        module_path = resolved.stem
        all_issues.extend(
            validate_atom_file(resolved, module_path=module_path)
        )

    rc = _print_issues(all_issues)
    raise typer.Exit(code=rc)


@app.command(name="package")
def validate_package_cmd(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Path(s) to atom package directories."),
    ],
) -> None:
    """Validate specific atom packages against the S11 contract."""
    all_issues: list[ValidationIssue] = []
    for p in paths:
        resolved = p.resolve()
        if not resolved.is_dir():
            print(f"ERROR: {p} is not a directory", file=sys.stderr)
            raise typer.Exit(code=2)
        module_path = resolved.name
        all_issues.extend(
            validate_atom_package(resolved, module_path=module_path)
        )

    rc = _print_issues(all_issues)
    raise typer.Exit(code=rc)


def main() -> None:
    """Entry point for ``agentm validate`` subcommand."""
    app()
