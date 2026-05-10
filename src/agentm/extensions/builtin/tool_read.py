"""Tool atom for the ``extensions.builtin.tool_read`` §7.1 row."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_read",
    description="Register the read tool backed by FileOperations.",
    registers=("tool:read",),
    config_schema={
        "type": "object",
        "properties": {
            "file_ops": {"type": "object"},
            # Glob patterns the agent is permitted to read. Patterns
            # match the resolved absolute path (symlinks collapsed). When
            # unset, all paths the process can access are allowed —
            # matches the historical behavior. Scenarios that expose
            # ``read`` to a model running over untrusted data (RCA eval
            # cases co-located with ground-truth labels, for example)
            # should always set this to fence in the blast radius.
            "allow_globs": {
                "type": "array",
                "items": {"type": "string"},
            },
            # Patterns that always deny, evaluated AFTER ``allow_globs``.
            # Useful for "broadly allow this tree, but never these files"
            # — e.g. ground-truth files like label.txt / injection.json
            # co-located with telemetry the agent legitimately needs.
            "deny_globs": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": True,
    },
    requires=(),  # Leaf tool atom: consumes Operations via ExtensionAPI.
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer", "default": 0},
        "limit": {"type": "integer", "default": 2000},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    file_ops = _coerce_file_ops(api, config.get("file_ops"))
    allow_globs = _coerce_globs(config.get("allow_globs"), api.cwd)
    deny_globs = _coerce_globs(config.get("deny_globs"), api.cwd)

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        offset = int(args.get("offset", 0))
        limit = int(args.get("limit", 2000))

        gate_error = _check_path_allowed(path, allow_globs, deny_globs)
        if gate_error is not None:
            return _error(gate_error)

        try:
            data = await file_ops.read_file(path)
            lines = data.decode("utf-8", errors="replace").splitlines()
            if limit < 0:
                sliced = lines[offset:]
            else:
                sliced = lines[offset : offset + limit]
            return _ok("\n".join(sliced))
        except Exception as exc:
            return _error(f"Failed to read {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="read",
            description="Read a UTF-8 text file from disk by line range.",
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "read"},
        )
    )


def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file


def _coerce_globs(value: Any, cwd: str) -> tuple[str, ...]:
    """Anchor relative glob patterns against the session ``cwd`` so a
    scenario manifest using ``contrib/scenarios/rca/skills/**`` keeps the
    same meaning regardless of where the user invoked the agent from."""

    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            continue
        if os.path.isabs(raw):
            out.append(raw)
        else:
            out.append(os.path.normpath(os.path.join(cwd, raw)))
    return tuple(out)


def _resolved(path: str) -> str:
    """Resolve to absolute, symlink-collapsed path for matching.

    Symlinks are followed so a deny pattern like ``**/dataset/**`` can't
    be bypassed by a symlink target that lives outside the deny tree.
    """

    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except (OSError, RuntimeError):
        return os.path.abspath(os.path.expanduser(path))


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _check_path_allowed(
    path: str,
    allow: tuple[str, ...],
    deny: tuple[str, ...],
) -> str | None:
    resolved = _resolved(path)
    if allow and not _matches_any(resolved, allow):
        return (
            f"Access denied: {path!r} is outside the configured allow_globs "
            f"({list(allow)}). Adjust the scenario manifest if this access "
            "should be permitted."
        )
    if deny and _matches_any(resolved, deny):
        return (
            f"Access denied: {path!r} matches a configured deny_glob "
            f"({list(deny)})."
        )
    return None


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
