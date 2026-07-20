#!/usr/bin/env python3
"""Require user input before Codex edits or adds files under tests/."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable


WRITE_TO_TESTS_PATTERNS = (
    re.compile(r"(?:^|[;&|]\s*)cat\b[\s\S]*?>{1,2}\s*['\"]?(?:\./)?tests/"),
    re.compile(r"(?:^|[;&|]\s*)tee\b[\s\S]*?(?:\s|^)['\"]?(?:\./)?tests/"),
    re.compile(
        r"(?:^|[;&|]\s*)(?:touch|mkdir|cp|mv|install|rsync|truncate)\b[\s\S]*?(?:\s|^)['\"]?(?:\./)?tests(?:/|\b)"
    ),
    re.compile(
        r"(?:^|[;&|]\s*)(?:sed|perl)\b[\s\S]*?(?:\s|^)-(?:[A-Za-z]*i[A-Za-z]*|p?i)\b[\s\S]*?(?:\s|^)['\"]?(?:\./)?tests/"
    ),
    re.compile(r"(?:write_text|write_bytes|open)\s*\([\s\S]*?['\"](?:\./)?tests/"),
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    if not isinstance(payload, dict):
        return 0

    tool_name = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input")
    command = _command_from_tool_input(tool_input)
    repo_root = Path(__file__).resolve().parents[2]
    cwd = _cwd(payload, repo_root)

    paths = _affected_test_paths(tool_name, tool_input, command, repo_root, cwd)
    if not paths:
        return 0

    reason = (
        "Repository policy: this operation would edit or add files under tests/. "
        "Ask the user for their opinion and explicit confirmation before retrying. "
        f"Detected path(s): {', '.join(paths[:8])}"
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


def _cwd(payload: dict[object, object], fallback: Path) -> Path:
    raw = payload.get("cwd")
    if isinstance(raw, str) and raw:
        return Path(raw)
    return fallback


def _command_from_tool_input(tool_input: object) -> str:
    if isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, str):
            return command
    if isinstance(tool_input, str):
        return tool_input
    return ""


def _affected_test_paths(
    tool_name: str,
    tool_input: object,
    command: str,
    repo_root: Path,
    cwd: Path,
) -> list[str]:
    paths: set[str] = set()
    if tool_name in {"apply_patch", "Edit", "Write"}:
        paths.update(_paths_from_patch_headers(command, repo_root, cwd))
        paths.update(_paths_from_structured_input(tool_input, repo_root, cwd))
    elif tool_name == "Bash":
        paths.update(_paths_from_bash(command, repo_root, cwd))
    return sorted(paths)


def _paths_from_patch_headers(command: str, repo_root: Path, cwd: Path) -> set[str]:
    paths: set[str] = set()
    current_is_tests_update = False
    for line in command.splitlines():
        file_match = re.match(r"^\*\*\* (Add|Update) File:\s+(.+)$", line)
        if file_match is not None:
            current_is_tests_update = False
            path = file_match.group(2).strip()
            if _is_tests_path(path, repo_root, cwd):
                current_is_tests_update = True
                paths.add(_display_path(path, repo_root, cwd))
            continue

        move_match = re.match(r"^\*\*\* Move to:\s+(.+)$", line)
        if move_match is not None and current_is_tests_update:
            path = move_match.group(1).strip()
            if _is_tests_path(path, repo_root, cwd):
                paths.add(_display_path(path, repo_root, cwd))
    return paths


def _paths_from_structured_input(
    tool_input: object, repo_root: Path, cwd: Path
) -> set[str]:
    paths: set[str] = set()
    for key, value in _walk_tool_input(tool_input):
        if key.lower() in {"path", "file_path", "filepath", "target_file", "filename"}:
            if _is_tests_path(value, repo_root, cwd):
                paths.add(_display_path(value, repo_root, cwd))
    return paths


def _walk_tool_input(value: object) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if isinstance(child, str):
                yield key_text, child
            else:
                yield from _walk_tool_input(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_tool_input(child)


def _paths_from_bash(command: str, repo_root: Path, cwd: Path) -> set[str]:
    paths: set[str] = set()
    if "*** Begin Patch" in command:
        paths.update(_paths_from_patch_headers(command, repo_root, cwd))

    for pattern in WRITE_TO_TESTS_PATTERNS:
        if pattern.search(command):
            paths.add("tests/")
            break
    return paths


def _is_tests_path(raw_path: str, repo_root: Path, cwd: Path) -> bool:
    path = _strip_path(raw_path)
    if not path:
        return False

    normalized = path.replace("\\", "/")
    if normalized == "tests" or normalized.startswith("tests/"):
        return True
    if normalized.startswith("./tests/") or normalized == "./tests":
        return True

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        relative = candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    relative_text = relative.as_posix()
    return relative_text == "tests" or relative_text.startswith("tests/")


def _display_path(raw_path: str, repo_root: Path, cwd: Path) -> str:
    path = _strip_path(raw_path)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        relative = candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path
    return relative.as_posix()


def _strip_path(raw_path: str) -> str:
    return raw_path.strip().strip("'\"")


if __name__ == "__main__":
    raise SystemExit(main())
