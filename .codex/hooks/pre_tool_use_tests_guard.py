#!/usr/bin/env python3
"""Require an explicit per-operation choice before Codex changes tests/.

Approval normally arrives through ``UserPromptSubmit``. Codex clients that do
not emit that event can use the operation-scoped fallback command printed by
the deny response after the user has made the same explicit choice.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
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
_READ_ONLY_COMMAND = re.compile(
    r"^\s*(?:rg|grep|egrep|fgrep|ag|ack|find|fd|ls|cat|head|tail|wc|file|stat|"
    r"git\s+(?:log|diff|show|blame|status|branch)|tree)\b"
)

PENDING_TTL_SECONDS = 10 * 60
APPROVAL_TTL_SECONDS = 5 * 60
ALLOW_CHOICES = frozenset(
    {"同意", "允许", "可以", "批准", "yes", "y", "allow", "approve"}
)
DENY_CHOICES = frozenset(
    {"不同意", "不允许", "拒绝", "取消", "no", "n", "deny", "reject"}
)
APPROVAL_TOKEN = re.compile(
    r"^(?P<state_id>[0-9a-f]{64})\.(?P<operation_id>[0-9a-f]{64})$"
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if len(sys.argv) > 1:
        return _handle_explicit_choice(sys.argv[1:], repo_root)

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    if not isinstance(payload, dict):
        return 0

    if payload.get("hook_event_name") == "UserPromptSubmit":
        return _handle_user_choice(payload, repo_root)

    tool_name = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input")
    command = _command_from_tool_input(tool_input)
    cwd = _cwd(payload, repo_root)

    paths = _affected_test_paths(tool_name, tool_input, command, repo_root, cwd)
    if not paths:
        return 0
    operation_id = _operation_id(tool_name, tool_input)
    state_path = _state_path(payload, repo_root)
    state = _load_state(state_path)
    if _is_approved(state, operation_id=operation_id, paths=paths):
        _remove_state(state_path)
        return 0

    _write_state(
        state_path,
        {
            "status": "pending",
            "operation_id": operation_id,
            "paths": paths,
            "expires_at": time.time() + PENDING_TTL_SECONDS,
        },
    )

    approval_token = _approval_token(state_path, operation_id)
    fallback = (
        " If this client does not apply UserPromptSubmit approval, then after the "
        "user explicitly agrees run: /usr/bin/python3 "
        f".codex/hooks/pre_tool_use_tests_guard.py approve {approval_token}"
        if approval_token is not None
        else ""
    )
    reason = (
        "Repository policy requires a user choice before changing tests/. "
        "Ask the user to reply '同意' to allow this exact operation once, or '拒绝' "
        "to cancel it. Retry the unchanged operation only after approval. "
        f"Pending path(s): {', '.join(paths[:8])}.{fallback}"
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


def _handle_explicit_choice(args: list[str], repo_root: Path) -> int:
    if len(args) != 2 or args[0] not in {"approve", "deny"}:
        print(
            "usage: pre_tool_use_tests_guard.py <approve|deny> <state-id.operation-id>",
            file=sys.stderr,
        )
        return 2

    match = APPROVAL_TOKEN.fullmatch(args[1])
    if match is None:
        print("invalid tests/ approval token", file=sys.stderr)
        return 2
    state_path = (
        repo_root / ".codex" / "test-edit-approvals" / f"{match.group('state_id')}.json"
    )
    state = _load_state(state_path)
    if (
        not _is_pending(state)
        or state is None
        or state.get("operation_id") != match.group("operation_id")
    ):
        print("tests/ approval is missing, expired, or for another operation")
        return 1

    if args[0] == "deny":
        _remove_state(state_path)
        print("tests/ operation denied")
        return 0

    state["status"] = "approved"
    state["expires_at"] = time.time() + APPROVAL_TTL_SECONDS
    _write_state(state_path, state)
    print(f"tests/ operation approved once: {', '.join(_state_paths(state)[:8])}")
    return 0


def _handle_user_choice(payload: dict[object, object], repo_root: Path) -> int:
    state_path = _state_path(payload, repo_root)
    state = _load_state(state_path)
    if not _is_pending(state):
        _remove_state(state_path)
        return 0

    prompt = payload.get("prompt")
    choice = _parse_choice(prompt if isinstance(prompt, str) else "")
    if choice is None:
        return 0

    paths = _state_paths(state)
    if choice == "deny":
        _remove_state(state_path)
        context = "The user rejected the pending tests/ change. Do not retry it."
    else:
        state["status"] = "approved"
        state["expires_at"] = time.time() + APPROVAL_TTL_SECONDS
        _write_state(state_path, state)
        context = (
            "The user approved the pending tests/ change. Retry the exact operation "
            f"once. Approved path(s): {', '.join(paths[:8])}"
        )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            }
        )
    )
    return 0


def _operation_id(tool_name: str, tool_input: object) -> str:
    serialized = json.dumps(
        {"tool_name": tool_name, "tool_input": tool_input},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _approval_token(path: Path | None, operation_id: str) -> str | None:
    if path is None:
        return None
    return f"{path.stem}.{operation_id}"


def _state_path(payload: dict[object, object], repo_root: Path) -> Path | None:
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    state_id = hashlib.sha256(session_id.encode()).hexdigest()
    return repo_root / ".codex" / "test-edit-approvals" / f"{state_id}.json"


def _load_state(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def _write_state(path: Path | None, state: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_state(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _is_pending(state: dict[str, object] | None) -> bool:
    if state is None or state.get("status") != "pending":
        return False
    expires_at = state.get("expires_at")
    return isinstance(expires_at, int | float) and expires_at >= time.time()


def _is_approved(
    state: dict[str, object] | None,
    *,
    operation_id: str,
    paths: list[str],
) -> bool:
    if state is None or state.get("status") != "approved":
        return False
    expires_at = state.get("expires_at")
    return (
        isinstance(expires_at, int | float)
        and expires_at >= time.time()
        and state.get("operation_id") == operation_id
        and _state_paths(state) == paths
    )


def _state_paths(state: dict[str, object]) -> list[str]:
    paths = state.get("paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        return []
    return sorted(paths)


def _parse_choice(prompt: str) -> str | None:
    normalized = re.sub(r"[\s，。！？!?,.]+", "", prompt.casefold())
    if normalized in DENY_CHOICES:
        return "deny"
    if normalized in ALLOW_CHOICES:
        return "allow"
    return None


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

    if not _READ_ONLY_COMMAND.match(command):
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
