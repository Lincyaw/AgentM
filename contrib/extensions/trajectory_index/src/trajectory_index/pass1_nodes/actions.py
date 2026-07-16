"""Structural action parsing — tool_call → Action node.

Parses tool_call blocks from the record to create Action nodes with
operation classification, target objects, and parameter diffs.  Pure
code, no model call (SCHEMA.md §Designed extensions: "read structurally
from the record").
"""

from __future__ import annotations

import re
from typing import Any

from ..ir.models import Action, ActionOp

_READ_CMDS = re.compile(
    r"(?:^|\s|&&\s*|;\s*)"
    r"(?:cat|head|tail|less|more|grep|find|ls|stat|wc|file|diff|sed\s+-n|awk)\s",
)
_WRITE_CMDS = re.compile(
    r"(?:^|\s|&&\s*|;\s*)"
    r"(?:sed\s+-i|tee\s|dd\s|patch\s)\s",
)
_EXECUTE_CMDS = re.compile(
    r"(?:run_flow|make\s|cmake\s|gcc\s|g\+\+\s|python\s|npm\s|cargo\s|go\s+build|go\s+run)",
)
_CHECK_CMDS = re.compile(r"(?:validate|submit|inbox|check)")
_REDIRECT_WRITE = re.compile(r"(?:>\s*\S|>>\s*\S)")
_FILE_PATH = re.compile(r"(?:/[\w./-]+|\./[\w./-]+)")

_ASSIGN_RE = re.compile(
    r"^(?:export\s+)?(\w+)\s*=\s*(.+?)(?:\s*#.*)?$", re.MULTILINE,
)


def classify_bash(cmd: str) -> ActionOp:
    if _CHECK_CMDS.search(cmd):
        return "check"
    if _EXECUTE_CMDS.search(cmd):
        return "execute"
    if _WRITE_CMDS.search(cmd) or _REDIRECT_WRITE.search(cmd):
        return "write"
    if _READ_CMDS.search(cmd):
        return "read"
    return "other"


def extract_targets_bash(cmd: str) -> tuple[str, ...]:
    paths = _FILE_PATH.findall(cmd)
    seen: set[str] = set()
    result: list[str] = []
    for p in paths:
        if p not in seen and not p.startswith("/dev/") and not p.startswith("/tmp/"):
            seen.add(p)
            result.append(p)
    return tuple(result[:10])


def parse_edit_diffs(
    old_string: str, new_string: str,
) -> tuple[tuple[str, str, str], ...]:
    old_assigns = dict(_ASSIGN_RE.findall(old_string))
    new_assigns = dict(_ASSIGN_RE.findall(new_string))
    diffs: list[tuple[str, str, str]] = []
    for key in new_assigns:
        old_val = old_assigns.get(key, "")
        new_val = new_assigns[key]
        if old_val != new_val:
            diffs.append((key, old_val.strip(), new_val.strip()))
    for key in old_assigns:
        if key not in new_assigns:
            diffs.append((key, old_assigns[key].strip(), ""))
    return tuple(diffs)


def parse_action(
    msg: dict[str, Any],
    *,
    step_id: str,
    run_id: str,
) -> Action | None:
    """Parse a single message dict into an Action, or None if not a tool_call."""
    blocks = msg.get("content", [])
    if not isinstance(blocks, list):
        return None

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype not in ("tool_call", "tool_use"):
            continue

        tool_name = str(block.get("name", ""))
        call_id = str(block.get("id", ""))
        args = block.get("arguments", block.get("input", {}))
        if not isinstance(args, dict):
            args = {}

        if tool_name == "edit":
            file_path = str(args.get("file_path", args.get("path", "")))
            old_s = str(args.get("old_string", ""))
            new_s = str(args.get("new_string", ""))
            diffs = parse_edit_diffs(old_s, new_s) if old_s and new_s else ()
            targets = (file_path,) if file_path else ()
            return Action(
                call_id=call_id, step_id=step_id, run_id=run_id,
                tool_name=tool_name, operation="write",
                targets=targets, diffs=diffs,
            )

        if tool_name in ("bash", "shell"):
            cmd = str(args.get("cmd", args.get("command", "")))
            op = classify_bash(cmd)
            targets = extract_targets_bash(cmd)
            return Action(
                call_id=call_id, step_id=step_id, run_id=run_id,
                tool_name=tool_name, operation=op,
                targets=targets,
            )

        if tool_name in ("read", "read_file"):
            file_path = str(args.get("file_path", args.get("path", "")))
            targets = (file_path,) if file_path else ()
            return Action(
                call_id=call_id, step_id=step_id, run_id=run_id,
                tool_name=tool_name, operation="read",
                targets=targets,
            )

        if tool_name == "write":
            file_path = str(args.get("file_path", args.get("path", "")))
            targets = (file_path,) if file_path else ()
            return Action(
                call_id=call_id, step_id=step_id, run_id=run_id,
                tool_name=tool_name, operation="write",
                targets=targets,
            )

        op: ActionOp = "other"
        if tool_name in ("submit", "validate", "inbox", "check_background"):
            op = "check"
        return Action(
            call_id=call_id, step_id=step_id, run_id=run_id,
            tool_name=tool_name, operation=op,
        )

    return None
