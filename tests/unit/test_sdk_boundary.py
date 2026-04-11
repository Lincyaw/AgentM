"""Architecture guard tests for SDK/scenario dependency boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


_AGENTM_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "agentm"

# Bridge modules that intentionally lazy-load scenario code.
_ALLOWED_SCENARIO_IMPORTERS: set[str] = {
    "builder.py",
    "harness/worker_factory.py",
    "cli/judge_runner.py",
}

_SDK_DIRS = [
    "core",
    "models",
    "agents",
    "config",
    "tools",
    "middleware",
    "backends",
    "cli",
    "server",
    "harness",
]


def _collect_sdk_files() -> list[Path]:
    files: list[Path] = []
    for directory in _SDK_DIRS:
        root = _AGENTM_ROOT / directory
        if root.exists():
            files.extend(root.rglob("*.py"))
    files.extend(_AGENTM_ROOT.glob("*.py"))
    return files


def _scenario_imports(filepath: Path) -> list[str]:
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names if alias.name.startswith("agentm.scenarios"))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("agentm.scenarios"):
                imports.append(node.module)
    return imports


def _relative(filepath: Path) -> str:
    return str(filepath.relative_to(_AGENTM_ROOT))


def test_sdk_modules_do_not_import_scenarios_directly() -> None:
    violations: list[str] = []

    for filepath in _collect_sdk_files():
        rel = _relative(filepath)
        if rel in _ALLOWED_SCENARIO_IMPORTERS or "__pycache__" in rel:
            continue

        scenario_imports = _scenario_imports(filepath)
        if scenario_imports:
            violations.append(f"{rel}: {', '.join(scenario_imports)}")

    assert violations == [], "SDK modules must not import scenarios:\n" + "\n".join(violations)


def test_scenarios_do_not_cross_import_each_other() -> None:
    scenarios_dir = _AGENTM_ROOT / "scenarios"
    cross_imports: list[str] = []

    for filepath in scenarios_dir.rglob("*.py"):
        if "__pycache__" in str(filepath):
            continue

        rel = str(filepath.relative_to(scenarios_dir))
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        scenario_name = parts[0]

        for module in _scenario_imports(filepath):
            parts = module.split(".")
            if len(parts) < 3:
                continue
            imported_scenario = parts[2]
            if imported_scenario != scenario_name:
                cross_imports.append(f"{rel}: imports {module}")

    assert cross_imports == [], "Scenarios must not cross-import:\n" + "\n".join(cross_imports)
