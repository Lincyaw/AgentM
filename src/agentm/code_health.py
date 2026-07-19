"""Project-specific architectural checks that Ruff and mypy cannot express.

The scanner is pure AST analysis: it does not import or execute inspected
modules. These rules protect the SDK composition boundary established by the
v2 refactor.
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
    """One project-specific code-health violation."""

    path: str
    line: int
    rule: str
    message: str
    severity: str = "error"


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
ALL_RULES: Final[tuple[str, ...]] = (
    "AM017",
    "AM018",
    "AM019",
    "AM020",
    "AM021",
    "AM022",
    "AM023",
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
        if (
            isinstance(node, ast.Call)
            and _call_name(node) in _SCENARIO_EXECUTION_CALLS
        ):
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
            value = node.value
            if (
                any(_tuple_target(target) for target in targets)
                and _attribute_named(value, "provider")
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
        if isinstance(node, ast.Call)
        and _call_name(node) in _DYNAMIC_ATTRIBUTE_CALLS
    ]


def _check_abi_any(
    tree: ast.Module,
    path: str,
    file_path: Path,
) -> list[Issue]:
    """AM022: the public ABI must not erase values to typing.Any."""

    if not _contains_parts(file_path, ("core", "abi")):
        return []
    return [
        Issue(
            path=path,
            line=node.lineno,
            rule="AM022",
            message=(
                "typing.Any erases the ABI contract; use object, JsonValue, "
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
                arg.annotation
                for arg in args
                if arg.annotation is not None
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


def _check_abi_bare_dict(
    tree: ast.Module,
    path: str,
    file_path: Path,
) -> list[Issue]:
    """AM023: ABI annotations must not contain an unparameterized dict."""

    if not _contains_parts(file_path, ("core", "abi")):
        return []
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


def _directive_rules(line: str, pattern: re.Pattern[str]) -> frozenset[str]:
    match = pattern.search(line)
    if match is None:
        return frozenset()
    return frozenset(
        item.strip()
        for item in match.group("rules").split(",")
        if item.strip()
    )


def _suppressed(
    issue: Issue,
    source_lines: list[str],
) -> bool:
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


def check_file(file_path: Path) -> list[Issue]:
    """Run all project rules on one Python file."""

    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    path = str(file_path)
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [
            Issue(
                path=path,
                line=exc.lineno or 1,
                rule="AM000",
                message=f"cannot parse Python source: {exc.msg}",
            )
        ]

    issues: list[Issue] = []
    issues.extend(_check_core_atom_policy(tree, path, file_path))
    issues.extend(_check_scenario_loader_execution(tree, path, file_path))
    issues.extend(_check_legacy_extension_shape(tree, path))
    issues.extend(_check_raw_cli_extension_config(tree, path, file_path))
    issues.extend(_check_dynamic_attribute_access(tree, path))
    issues.extend(_check_abi_any(tree, path, file_path))
    issues.extend(_check_abi_bare_dict(tree, path, file_path))
    source_lines = source.splitlines()
    return [
        issue
        for issue in issues
        if not _suppressed(issue, source_lines)
    ]


def check_paths(paths: list[Path]) -> list[Issue]:
    """Run checks over files and recursively over directories."""

    issues: list[Issue] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            issues.extend(check_file(path))
        elif path.is_dir():
            for source in sorted(path.rglob("*.py")):
                if "__pycache__" not in source.parts:
                    issues.extend(check_file(source))
    return issues


def changed_files(base: str = "origin/main") -> list[Path]:
    """Return existing Python files changed from the merge-base with ``base``."""

    try:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", base],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        names = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                merge_base,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [
        Path(name)
        for name in names
        if name.endswith(".py") and Path(name).is_file()
    ]


app = typer.Typer(
    name="lint",
    help="Project-specific AST architecture checks.",
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def lint_cmd(
    paths: list[str] | None = typer.Argument(
        None,
        help="Files or directories to check. Default: src/agentm.",
    ),
    changed: bool = typer.Option(
        False,
        "--changed",
        "-c",
        help="Check Python files changed from the merge-base with origin/main.",
    ),
    rule: list[str] | None = typer.Option(
        None,
        "--rule",
        "-r",
        help="Run only the selected rule identifiers.",
    ),
    warnings_as_errors: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Treat warnings as errors.",
    ),
) -> None:
    """Run AgentM-specific architectural checks."""

    selected = changed_files() if changed else [
        Path(value)
        for value in (paths or ["src/agentm"])
    ]
    issues = check_paths(selected)
    if rule:
        enabled = set(rule)
        unknown = enabled - set(ALL_RULES)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise typer.BadParameter(f"unknown code-health rule(s): {names}")
        issues = [issue for issue in issues if issue.rule in enabled]

    issues.sort(key=lambda issue: (issue.path, issue.line, issue.rule))
    for issue in issues:
        marker = "E" if issue.severity == "error" else "W"
        typer.echo(
            f"{issue.path}:{issue.line}: {issue.rule} "
            f"[{marker}] {issue.message}"
        )
    failures = [
        issue
        for issue in issues
        if issue.severity == "error" or warnings_as_errors
    ]
    if failures:
        typer.echo(
            f"{len(failures)} code-health violation(s)",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo("No code health issues found.", err=True)


if __name__ == "__main__":
    app()
