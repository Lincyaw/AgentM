"""Validate the repository requirements index without external plugin state."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

_STATUSES = frozenset({"implemented", "tested", "deprecated"})
_CONFIDENCES = frozenset({"confirmed", "inferred"})
_PRIORITIES = frozenset({"P0", "P1", "P2"})
_PATH_SECTIONS = ("code", "tests", "docs")


def _path_entries(
    requirement: dict[str, Any],
    section: str,
    errors: list[str],
) -> list[str]:
    raw_entries = requirement.get(section)
    requirement_id = requirement.get("id", "<missing-id>")
    if not isinstance(raw_entries, list):
        errors.append(f"{requirement_id}: {section} must be a list")
        return []

    paths: list[str] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append(
                f"{requirement_id}: {section}[{index}] must be a mapping with a string path"
            )
            continue
        path = entry["path"].strip()
        if not path:
            errors.append(f"{requirement_id}: {section}[{index}] has an empty path")
            continue
        paths.append(path)
    if len(paths) != len(set(paths)):
        errors.append(f"{requirement_id}: {section} contains duplicate paths")
    return paths


def _find_cycles(
    dependency_map: dict[str, tuple[str, ...]],
) -> Iterable[tuple[str, ...]]:
    visiting: list[str] = []
    active: set[str] = set()
    complete: set[str] = set()

    def visit(requirement_id: str) -> Iterable[tuple[str, ...]]:
        if requirement_id in complete:
            return
        if requirement_id in active:
            start = visiting.index(requirement_id)
            yield (*visiting[start:], requirement_id)
            return
        active.add(requirement_id)
        visiting.append(requirement_id)
        for dependency in dependency_map.get(requirement_id, ()):
            yield from visit(dependency)
        visiting.pop()
        active.remove(requirement_id)
        complete.add(requirement_id)

    for requirement_id in dependency_map:
        yield from visit(requirement_id)


def validate(index_path: Path) -> list[str]:
    """Return every validation error found in *index_path*."""
    root = index_path.parent
    loaded = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return ["index root must be a mapping"]
    requirements = loaded.get("requirements")
    if not isinstance(requirements, list):
        return ["index root must contain a requirements list"]

    errors: list[str] = []
    requirement_ids: list[str] = []
    dependency_map: dict[str, tuple[str, ...]] = {}

    for position, raw_requirement in enumerate(requirements):
        if not isinstance(raw_requirement, dict):
            errors.append(f"requirements[{position}] must be a mapping")
            continue
        requirement_id = raw_requirement.get("id")
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            errors.append(f"requirements[{position}] has no non-empty string id")
            continue
        requirement_ids.append(requirement_id)

        for field in ("title", "description"):
            value = raw_requirement.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{requirement_id}: {field} must be a non-empty string")

        priority = raw_requirement.get("priority")
        if priority not in _PRIORITIES:
            errors.append(
                f"{requirement_id}: priority must be one of {sorted(_PRIORITIES)}"
            )
        status = raw_requirement.get("status")
        if status not in _STATUSES:
            errors.append(f"{requirement_id}: status must be one of {sorted(_STATUSES)}")
        confidence = raw_requirement.get("confidence")
        if confidence not in _CONFIDENCES:
            errors.append(
                f"{requirement_id}: confidence must be one of {sorted(_CONFIDENCES)}"
            )

        paths_by_section = {
            section: _path_entries(raw_requirement, section, errors)
            for section in _PATH_SECTIONS
        }
        if status in {"implemented", "tested"} and not paths_by_section["code"]:
            errors.append(f"{requirement_id}: {status} requirement must map to code")
        if status == "tested" and not paths_by_section["tests"]:
            errors.append(f"{requirement_id}: tested requirement must map to tests")
        for section, paths in paths_by_section.items():
            for relative_path in paths:
                path = root / relative_path
                if not path.exists():
                    errors.append(
                        f"{requirement_id}: missing {section} path {relative_path}"
                    )

        raw_dependencies = raw_requirement.get("depends_on")
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) for item in raw_dependencies
        ):
            errors.append(f"{requirement_id}: depends_on must be a list of ids")
            dependencies: tuple[str, ...] = ()
        else:
            dependencies = tuple(raw_dependencies)
            if len(dependencies) != len(set(dependencies)):
                errors.append(f"{requirement_id}: depends_on contains duplicate ids")
            if requirement_id in dependencies:
                errors.append(f"{requirement_id}: requirement cannot depend on itself")
        dependency_map[requirement_id] = dependencies

    duplicates = sorted(
        requirement_id
        for requirement_id in set(requirement_ids)
        if requirement_ids.count(requirement_id) > 1
    )
    for requirement_id in duplicates:
        errors.append(f"duplicate requirement id: {requirement_id}")

    known_ids = set(requirement_ids)
    for requirement_id, dependencies in dependency_map.items():
        for dependency in dependencies:
            if dependency not in known_ids:
                errors.append(f"{requirement_id}: unknown dependency {dependency}")
    for cycle in _find_cycles(dependency_map):
        errors.append(f"dependency cycle: {' -> '.join(cycle)}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, nargs="?", default=Path("project-index.yaml"))
    args = parser.parse_args()

    errors = validate(args.index)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"{len(errors)} project-index violation(s)")
        return 1
    print("project-index validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
