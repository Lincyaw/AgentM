# code-health: ignore-file[AM025] -- policy engine normalizes untyped YAML, SQLite, and runtime event payloads
"""Policy rule loading shared by live runtime and offline projection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .compiler import compile_policy_file, compose_rules
from .ifc import IFCEngine, compile_labels
from .types import RuleInstance


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    rules: list[RuleInstance]
    ifc: IFCEngine | None
    file_mtimes: dict[str, float]
    paths: tuple[Path, ...]


def load_policy_bundle(
    policy_files: Sequence[str],
    *,
    cwd: Path,
    scenario_dir: Path | None = None,
) -> PolicyBundle:
    """Load only the explicitly configured policy layers."""
    layers: list[tuple[list[RuleInstance], list[str]]] = []
    file_mtimes: dict[str, float] = {}
    paths: list[Path] = []
    labels = []

    for path in _policy_paths(policy_files, cwd=cwd, scenario_dir=scenario_dir):
        content = path.read_text(encoding="utf-8")

        rules, disabled = compile_policy_file(content)
        layers.append((rules, disabled))
        paths.append(path)
        file_mtimes[str(path)] = path.stat().st_mtime
        labels.extend(_labels_from_policy(content, path))
        logger.debug("loaded {} rules from {}", len(rules), path)

    return PolicyBundle(
        rules=compose_rules(layers),
        ifc=IFCEngine(labels) if labels else None,
        file_mtimes=file_mtimes,
        paths=tuple(paths),
    )


def resolve_policy_path(
    file_ref: str,
    *,
    cwd: Path,
    scenario_dir: Path | None = None,
) -> Path | None:
    package_prefix = "package:"
    if file_ref.startswith(package_prefix):
        relative = Path(file_ref.removeprefix(package_prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid package policy path: {file_ref}")
        package_candidate = Path(__file__).parent / relative
        return package_candidate if package_candidate.exists() else None
    path = Path(file_ref)
    if path.is_absolute() and path.exists():
        return path
    candidate = cwd / file_ref
    if candidate.exists():
        return candidate
    if scenario_dir is not None:
        scenario_candidate = scenario_dir / file_ref
        if scenario_candidate.exists():
            return scenario_candidate
    home_candidate = Path.home() / ".agentm" / "policies" / file_ref
    if home_candidate.exists():
        return home_candidate
    return None


def _policy_paths(
    policy_files: Sequence[str],
    *,
    cwd: Path,
    scenario_dir: Path | None,
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for file_ref in policy_files:
        path = resolve_policy_path(
            file_ref,
            cwd=cwd,
            scenario_dir=scenario_dir,
        )
        if path is None:
            raise FileNotFoundError(f"policy file not found: {file_ref}")
        resolved.append(path)
    return tuple(resolved)


def _labels_from_policy(content: str, path: Path) -> list:
    import yaml

    try:
        data = yaml.safe_load(content)
    except Exception as exc:
        logger.debug("failed to parse policy YAML {}: {}", path, exc)
        return []
    if isinstance(data, dict) and isinstance(data.get("labels"), dict):
        return compile_labels(data["labels"])
    return []
