"""Architecture guard: SDK core must never import from agentm.scenarios.

This test uses AST scanning to verify that SDK core modules do not depend
on domain-specific scenario code. The dependency direction must be:

    scenarios/* → SDK (core/, models/, agents/, ...)
    SDK         → (never import scenarios/)

Files with lazy loading from scenarios (registries, builder) are excluded —
they import from scenarios inside functions to populate registries on demand.
"""

from __future__ import annotations

import ast
from pathlib import Path


# Root of the agentm package
_AGENTM_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "agentm"

# Files that are allowed to import from agentm.scenarios (lazy-loading
# bridge modules that explicitly load scenario code on demand).
_LAZY_LOADERS: set[str] = {
    # Builder resolves scenario formatters and tools lazily
    "builder.py",
    # Agent factory calls scenarios.discover() inside function
    "agents/node/worker.py",
}

# SDK modules to scan (everything except scenarios/ and __pycache__)
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
]


def _collect_sdk_files() -> list[Path]:
    """Collect all .py files in SDK directories."""
    files: list[Path] = []
    for d in _SDK_DIRS:
        dir_path = _AGENTM_ROOT / d
        if dir_path.exists():
            files.extend(dir_path.rglob("*.py"))
    # Also include top-level files (builder.py, __init__.py)
    files.extend(_AGENTM_ROOT.glob("*.py"))
    return files


def _get_scenario_imports(filepath: Path) -> list[str]:
    """Parse a Python file and return all imports from agentm.scenarios."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    scenario_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agentm.scenarios"):
                    scenario_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("agentm.scenarios"):
                scenario_imports.append(node.module)

    return scenario_imports


def _relative_path(filepath: Path) -> str:
    """Get path relative to agentm root."""
    return str(filepath.relative_to(_AGENTM_ROOT))


class TestSDKBoundary:
    """Verify SDK core has zero import dependency on scenarios/."""

    def test_sdk_modules_do_not_import_scenarios(self) -> None:
        """All SDK core files (excluding lazy loaders) must not import from agentm.scenarios."""
        violations: list[str] = []

        for filepath in _collect_sdk_files():
            rel = _relative_path(filepath)

            # Skip lazy loaders
            if rel in _LAZY_LOADERS:
                continue

            # Skip __pycache__
            if "__pycache__" in str(filepath):
                continue

            scenario_imports = _get_scenario_imports(filepath)
            if scenario_imports:
                violations.append(f"{rel}: imports {', '.join(scenario_imports)}")

        assert violations == [], (
            "SDK core modules must not import from agentm.scenarios.\n"
            "Violations:\n" + "\n".join(f"  - {v}" for v in violations)
        )

    def test_lazy_loaders_exist(self) -> None:
        """All declared lazy loader files should actually exist."""
        for loader_path in _LAZY_LOADERS:
            full_path = _AGENTM_ROOT / loader_path
            assert full_path.exists(), f"Lazy loader not found: {loader_path}"

    def test_scenarios_directory_exists(self) -> None:
        """The scenarios directory should exist with expected sub-packages."""
        scenarios = _AGENTM_ROOT / "scenarios"
        assert scenarios.exists(), "scenarios/ directory not found"
        assert (scenarios / "__init__.py").exists()
        assert (scenarios / "rca" / "__init__.py").exists()
        assert (scenarios / "trajectory_analysis" / "__init__.py").exists()

    def test_scenarios_import_from_sdk(self) -> None:
        """Scenario files should import from SDK (not from other scenarios)."""
        scenarios_dir = _AGENTM_ROOT / "scenarios"
        cross_imports: list[str] = []

        for filepath in scenarios_dir.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            rel = str(filepath.relative_to(scenarios_dir))

            # Determine which scenario this file belongs to
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            scenario_name = parts[0]

            try:
                source = filepath.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(filepath))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("agentm.scenarios."):
                            module = alias.name
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("agentm.scenarios."):
                        module = node.module

                if module is not None:
                    # Extract which scenario it imports from
                    imported_scenario = module.split(".")[2]  # agentm.scenarios.<name>
                    if imported_scenario != scenario_name:
                        cross_imports.append(
                            f"{scenario_name}/{rel}: imports from {module}"
                        )

        assert cross_imports == [], (
            "Scenarios should not import from other scenarios.\n"
            "Cross-imports:\n" + "\n".join(f"  - {v}" for v in cross_imports)
        )
