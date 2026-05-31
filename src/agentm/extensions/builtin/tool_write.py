"""Tool atom for the ``extensions.builtin.tool_write`` row.

Enforces Claude-Code-style safety gates for file writes:

1. **read-before-write** -- existing files must have a prior *full* read
   (``is_partial=False``).  Partial reads (offset/limit) are rejected with
   a clear message.  New files (path does not exist) can be written freely.

2. **file-modified-since-read** -- if ``FileReadState.mtime_ns`` is
   available (set by another worker's read_state update), the on-disk
   mtime is compared before writing.  A mismatch means something else
   touched the file after the agent read it; the write is rejected so the
   agent re-reads first.

3. **post-write read_state update** -- after a successful write,
   ``record_read()`` is called with the new file's stats so downstream
   tool_edit calls see fresh state.
"""

from __future__ import annotations

import os
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.lib.read_state import get_read_state, record_read
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_write",
    description="Register the write tool backed by ResourceWriter.",
    registers=("tool:write",),
    config_schema={
        "type": "object",
        "properties": {
            "require_read": {
                "type": "boolean",
                "default": True,
                "description": "Require existing files to be read before overwriting.",
            },
        },
        "additionalProperties": True,
    },
    requires=(),
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write."},
        "content": {
            "type": "string",
            "description": "The full content to write.",
        },
        "rationale": {
            "type": "string",
            "default": "agent write via tool_write",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    writer = api.get_resource_writer()
    require_read = bool(config.get("require_read", True))

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        content = str(args["content"])
        rationale = str(args.get("rationale", "agent write via tool_write"))

        normalized = os.path.normpath(path)

        # Determine if the file already exists on disk.
        file_exists = False
        try:
            await writer.read(path)
            file_exists = True
        except Exception:
            pass

        if file_exists and require_read:
            rs = get_read_state(normalized)

            # Gate 1: must have been read at all.
            if rs is None:
                return _error(
                    f"File {path!r} already exists. Read it first before "
                    "overwriting so you can see its current content. "
                    "Use the read tool, then write."
                )

            # Gate 2: must have been a full read (no offset/limit).
            if rs.is_partial:
                return _error(
                    f"You read {path!r} with offset/limit (partial view). "
                    "Read the full file before overwriting."
                )

            # Gate 3: mtime must not have changed since the read.
            # mtime_ns is being added by another worker; fall back
            # gracefully when the field is not present yet.
            recorded_mtime = getattr(rs, "mtime_ns", None)
            if recorded_mtime is not None:
                try:
                    current_mtime = os.stat(normalized).st_mtime_ns
                except OSError:
                    current_mtime = None
                if current_mtime is not None and current_mtime != recorded_mtime:
                    return _error(
                        "File has been modified since you read it. "
                        "Read it again before writing."
                    )

        try:
            result = await writer.write(
                path,
                content.encode("utf-8"),
                rationale=rationale,
            )
            if result.error is not None:
                return _error(result.error)

            # Post-write: update read_state so subsequent tool_edit calls
            # see the file as freshly read (full content, not partial).
            total_lines = content.count("\n") + (1 if content else 0)
            record_kwargs: dict[str, Any] = {
                "total_lines": total_lines,
                "is_partial": False,
            }
            # Forward mtime_ns if record_read accepts it (added by the
            # other worker's read_state update).
            try:
                disk_mtime = os.stat(normalized).st_mtime_ns
                # Only pass mtime_ns if record_read supports it; avoids
                # TypeError on the current signature.
                import inspect
                sig = inspect.signature(record_read)
                if "mtime_ns" in sig.parameters:
                    record_kwargs["mtime_ns"] = disk_mtime
            except OSError:
                pass
            record_read(normalized, **record_kwargs)

            action = "Updated" if file_exists else "Created"
            byte_count = len(content.encode("utf-8"))
            return _ok(f"{action} {path!r} ({byte_count} bytes)")
        except Exception as exc:
            return _error(f"Failed to write {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="write",
            description=(
                "Write a UTF-8 text file. For existing files, you MUST read "
                "the full file first. Prefer the edit tool for modifying "
                "existing files — use write only for new files or complete "
                "rewrites."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "write"},
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
