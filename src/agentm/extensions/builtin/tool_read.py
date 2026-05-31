"""Tool atom for the ``extensions.builtin.tool_read`` §7.1 row.

Aligned with Claude Code's FileReadTool behavior: no hardcoded line cap,
max-file-size gate (default 256 KB), partial-view tracking for downstream
edit/write safety.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import PurePath, Path
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.core.lib.read_state import record_read
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


# 256 KB — matches Claude Code's MAX_OUTPUT_SIZE (0.25 * 1024 * 1024).
_DEFAULT_MAX_SIZE_BYTES: Final[int] = 262_144


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
            # Max file size in bytes. Files larger than this are rejected
            # with an error telling the model to use offset+limit.
            "max_size_bytes": {
                "type": "integer",
                "default": _DEFAULT_MAX_SIZE_BYTES,
            },
        },
        "additionalProperties": True,
    },
    requires=(),  # Leaf tool atom: consumes Operations via ExtensionAPI.
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Absolute path to the file to read.",
        },
        "offset": {
            "type": "integer",
            "description": (
                "1-based line number to start reading from. "
                "Only provide if the file is too large to read at once."
            ),
        },
        "limit": {
            "type": "integer",
            "description": (
                "Number of lines to read. "
                "Only provide if the file is too large to read at once."
            ),
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


_BINARY_EXTENSIONS: Final[frozenset[str]] = frozenset({
    # Video
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    # Audio
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a",
    # Image
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".ico", ".svg",
    # Archive
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Binary / native
    ".bin", ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".pyc", ".class",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Database
    ".sqlite", ".db",
})


def _check_binary(path: str) -> str | None:
    """Return an error string if *path* looks like a binary file, else None."""
    ext = PurePath(path).suffix.lower()
    if ext in _BINARY_EXTENSIONS:
        return (
            f"Cannot read binary file {path!r} ({ext} format). "
            "Use bash to inspect metadata (e.g. `file <path>`, `ls -la <path>`) "
            "or process it with appropriate tools."
        )
    return None


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    file_ops = _coerce_file_ops(api, config.get("file_ops"))
    allow_globs = _coerce_globs(config.get("allow_globs"), api.cwd)
    deny_globs = _coerce_globs(config.get("deny_globs"), api.cwd)
    max_size_bytes: int = int(
        config.get("max_size_bytes", _DEFAULT_MAX_SIZE_BYTES)
    )

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        raw_offset = args.get("offset")
        raw_limit = args.get("limit")

        gate_error = _check_path_allowed(path, allow_globs, deny_globs)
        if gate_error is not None:
            return _error(gate_error)

        binary_error = _check_binary(path)
        if binary_error is not None:
            return _error(binary_error)

        try:
            data = await file_ops.read_file(path)
        except Exception as exc:
            return _error(f"Failed to read {path!r}: {exc}")

        # --- max-size gate (checked on raw bytes, before decode) ---
        file_size = len(data)
        caller_wants_range = raw_offset is not None or raw_limit is not None
        if file_size > max_size_bytes and not caller_wants_range:
            return _error(
                f"File content ({file_size} bytes) exceeds maximum "
                f"allowed size ({max_size_bytes} bytes). "
                "Use offset and limit parameters to read specific "
                "portions of the file."
            )

        try:
            all_lines = data.decode("utf-8", errors="replace").splitlines()
            total = len(all_lines)

            # Offset: 1-based when provided, 0 means "from beginning".
            offset = max(0, int(raw_offset) - 1) if raw_offset is not None else 0
            limit = int(raw_limit) if raw_limit is not None else None

            if limit is not None and limit > 0:
                sliced = all_lines[offset : offset + limit]
            else:
                sliced = all_lines[offset:]

            is_partial = (
                offset > 0
                or (limit is not None and limit > 0 and offset + limit < total)
            )

            record_read(path, total_lines=total, is_partial=is_partial)

            numbered = [
                f"{offset + i + 1}\t{line}"
                for i, line in enumerate(sliced)
            ]

            if is_partial:
                end_line = offset + len(sliced)
                header = f"(showing lines {offset + 1}-{end_line} of {total})"
            else:
                header = f"({total} lines total)"

            return _ok(header + "\n" + "\n".join(numbered))
        except Exception as exc:
            return _error(f"Failed to read {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="read",
            description=(
                "Read a UTF-8 text file from disk. "
                "By default reads the entire file. "
                f"Files larger than {max_size_bytes} bytes require "
                "offset and limit parameters."
            ),
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
