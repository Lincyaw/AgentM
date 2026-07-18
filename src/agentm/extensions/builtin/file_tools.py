"""Grouped file-I/O tool atom: ``read``, ``write``, ``edit``.

Thin wrapper around :mod:`agentm_toolbox`.  In local sessions the toolbox
runs in-process (native Python call); in sandbox sessions it is uploaded
to the container and invoked via ``exec``.

The LLM-facing tool interface (names, schemas, output format) is identical
in both modes.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentm.core.abi import (
    BashOperations,
    ExtensionAPI,
    FunctionTool,
    ResourceWriter,
    TOOL_RESULT_FORMAT_METADATA_KEY,
    TextContent,
    ToolResult,
)
from agentm.core.lib import record_read
from agentm.extensions import ExtensionManifest

from agentm_toolbox import FileToolbox
from agentm_toolbox._file_ops import Result as ToolboxResult

# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

_ALL_TOOLS: Final[frozenset[str]] = frozenset({"read", "write", "edit"})

_SANDBOX_WORK_DIR_SERVICE: Final[str] = "agent_env.work_dir"
_TOOLBOX_CONTAINER_DIR: Final[str] = "/opt/agentm-toolbox"
_TOOLBOX_PKG: Final[str] = "agentm_toolbox"


class FileToolsConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tools: list[str] | None = None
    allow_globs: list[str] | None = None
    deny_globs: list[str] | None = None
    max_size_bytes: int = 262_144
    require_read: bool = True
    default_limit: int = 250
    verify_readback: bool = False

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        requested = frozenset(value)
        unknown = requested - _ALL_TOOLS
        if unknown:
            allowed = ", ".join(sorted(_ALL_TOOLS))
            bad = ", ".join(sorted(unknown))
            raise ValueError(
                f"unknown file_tools tool(s): {bad}; allowed tools: {allowed}"
            )
        return value


MANIFEST = ExtensionManifest(
    name="file_tools",
    description="Register the read, write, and edit tools for guarded file I/O.",
    registers=("tool:read", "tool:write", "tool:edit"),
    config_schema=FileToolsConfig,
    requires=(),
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _toolbox_to_tool_result(r: ToolboxResult) -> ToolResult:
    return _error(r.text) if r.is_error else _ok(r.text)


_PATH_ALIASES: Final[tuple[str, ...]] = ("file_path",)


def _required_string_arg(
    args: dict[str, Any],
    key: str,
    tool_name: str,
    *,
    aliases: tuple[str, ...] = (),
    allow_empty: bool = False,
    hint: str,
) -> tuple[str | None, ToolResult | None]:
    supplied_name = next((name for name in (key, *aliases) if name in args), None)
    if supplied_name is None:
        alias_text = ""
        if aliases:
            alias_text = f" Accepted aliases: {', '.join(repr(a) for a in aliases)}."
        return (
            None,
            _error(
                f"Invalid {tool_name} call: missing required argument {key!r}."
                f"{alias_text} Use {hint}."
            ),
        )

    value = args[supplied_name]
    if not isinstance(value, str):
        return (
            None,
            _error(
                f"Invalid {tool_name} call: argument {supplied_name!r} must be a "
                f"string, got {type(value).__name__}. Use {hint}."
            ),
        )
    if not allow_empty and value == "":
        return (
            None,
            _error(
                f"Invalid {tool_name} call: argument {supplied_name!r} must not "
                f"be empty. Use {hint}."
            ),
        )
    return value, None


def _coerce_globs(value: Any, cwd: str) -> tuple[str, ...]:
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
        return f"Access denied: {path!r} matches a configured deny_glob ({list(deny)})."
    return None


# ---------------------------------------------------------------------------
# Pydantic arg schemas
# ---------------------------------------------------------------------------


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(
        description="Path to the file to read (absolute, or relative to the session cwd)."
    )
    offset: int | None = Field(
        default=None,
        description=(
            "1-based line number to start reading from. Providing offset "
            "and/or limit lifts the size gate on large files."
        ),
    )
    limit: int | None = Field(
        default=None,
        description=(
            "Number of lines to read. Providing offset and/or limit lifts "
            "the size gate on large files."
        ),
    )


class _WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File path to write.")
    content: str = Field(description="The full content to write.")
    rationale: str = Field(default="agent write via file_tools")


class _EditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File path to edit.")
    old_string: str | None = Field(
        default=None,
        description="Exact text to find and replace. Mutually exclusive with start_line/end_line.",
    )
    new_string: str = Field(description="Replacement text.")
    start_line: int | None = Field(
        default=None,
        description="1-based start line for line-range replacement (inclusive).",
    )
    end_line: int | None = Field(
        default=None,
        description="1-based end line for line-range replacement (inclusive).",
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "Replace every occurrence of old_string. Without this, "
            "old_string must match exactly once or the edit is rejected."
        ),
    )
    rationale: str = Field(default="agent edit via file_tools")


# ---------------------------------------------------------------------------
# Sandbox toolbox uploader
# ---------------------------------------------------------------------------


def _collect_toolbox_sources() -> dict[str, bytes]:
    """Collect the agentm_toolbox package source files for container upload."""
    import agentm_toolbox as _pkg

    pkg_dir = Path(_pkg.__file__).parent
    sources: dict[str, bytes] = {}
    for py_file in sorted(pkg_dir.glob("*.py")):
        rel = f"{_TOOLBOX_PKG}/{py_file.name}"
        sources[rel] = py_file.read_bytes()
    return sources


async def _upload_toolbox(
    writer: ResourceWriter, bash_ops: BashOperations, work_dir: str
) -> None:
    """Upload the agentm_toolbox package into the sandbox container."""
    sources = _collect_toolbox_sources()
    pkg_target = f"{_TOOLBOX_CONTAINER_DIR}/{_TOOLBOX_PKG}"
    await bash_ops.exec(
        f"mkdir -p {shlex.quote(pkg_target)}",
        cwd=work_dir,
        timeout=10,
    )
    for rel_path, content in sources.items():
        target = f"{_TOOLBOX_CONTAINER_DIR}/{rel_path}"
        await writer.write(target, content, rationale="toolbox upload")
    logger.debug(
        "file_tools: uploaded {} toolbox files to container", len(sources)
    )


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------


def install(api: ExtensionAPI, config: FileToolsConfig) -> None:
    _FileToolsRuntime(api=api, config=config).install()


def _enabled_tools(configured: list[str] | None) -> frozenset[str]:
    if configured is None:
        return _ALL_TOOLS
    enabled_tools = frozenset(configured)
    unknown = enabled_tools - _ALL_TOOLS
    if unknown:
        allowed = ", ".join(sorted(_ALL_TOOLS))
        requested = ", ".join(sorted(unknown))
        raise ValueError(
            f"unknown file_tools tool(s): {requested}; allowed tools: {allowed}"
        )
    return enabled_tools


class _FileToolsRuntime:
    """Owns file_tools registration and per-session handler state."""

    def __init__(self, *, api: ExtensionAPI, config: FileToolsConfig) -> None:
        self._api = api
        self._config = config
        self._enabled_tools = _enabled_tools(config.tools)
        self._allow_globs = _coerce_globs(config.allow_globs, api.cwd)
        self._deny_globs = _coerce_globs(config.deny_globs, api.cwd)
        self._max_size_bytes = config.max_size_bytes

        # Detect sandbox vs local mode
        sandbox_work_dir = api.get_service(_SANDBOX_WORK_DIR_SERVICE)
        if sandbox_work_dir is not None:
            self._sandbox = True
            self._sandbox_work_dir = str(sandbox_work_dir)
            self._bash_ops = api.get_operations().bash
            self._writer = api.get_resource_writer()
            self._toolbox = None
            self._toolbox_uploaded = False
        else:
            self._sandbox = False
            self._toolbox = FileToolbox(
                cwd=api.cwd,
                max_size=config.max_size_bytes,
                require_read=config.require_read,
                default_limit=config.default_limit,
            )

    def install(self) -> None:
        if "read" in self._enabled_tools:
            self._register_read()
        if "write" in self._enabled_tools:
            self._register_write()
        if "edit" in self._enabled_tools:
            self._register_edit()

    # -- sandbox exec dispatch ----------------------------------------------

    async def _ensure_toolbox_uploaded(self) -> None:
        if not self._sandbox or self._toolbox_uploaded:
            return
        assert self._writer is not None
        assert self._bash_ops is not None
        await _upload_toolbox(
            self._writer, self._bash_ops, self._sandbox_work_dir
        )
        self._toolbox_uploaded = True

    async def _exec_toolbox(
        self, tool_name: str, args: dict[str, Any]
    ) -> ToolboxResult:
        await self._ensure_toolbox_uploaded()
        assert self._bash_ops is not None

        exec_args: dict[str, Any] = {
            **args,
            "_cwd": self._sandbox_work_dir,
            "_max_size": self._max_size_bytes,
            "_require_read": self._config.require_read,
            "_default_limit": self._config.default_limit,
        }

        if tool_name == "write" and "content" in exec_args:
            content_str: str = exec_args.pop("content")
            tmp = f"/tmp/.agentm-tb-{uuid.uuid4().hex}"
            assert self._writer is not None
            await self._writer.write(
                tmp, content_str.encode("utf-8"), rationale="write content"
            )
            exec_args["content_file"] = tmp

        json_args = json.dumps(exec_args, ensure_ascii=False)
        cmd = (
            f"PYTHONPATH={shlex.quote(_TOOLBOX_CONTAINER_DIR)} "
            f"python3 -m {_TOOLBOX_PKG} {tool_name} {shlex.quote(json_args)}"
        )
        result = await self._bash_ops.exec(
            cmd, cwd=self._sandbox_work_dir, timeout=30
        )
        if result.exit_code != 0 and not result.stdout:
            stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
            return ToolboxResult(
                text=f"toolbox exec failed (exit {result.exit_code}): {stderr_text}",
                is_error=True,
            )
        try:
            data = json.loads(result.stdout.decode("utf-8", errors="replace"))
            return ToolboxResult(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            stdout_text = result.stdout.decode("utf-8", errors="replace").strip()
            return ToolboxResult(
                text=f"toolbox output parse error: {exc}\nraw: {stdout_text[:500]}",
                is_error=True,
            )

    # -- dispatch helper ----------------------------------------------------

    async def _dispatch(
        self, tool_name: str, args: dict[str, Any]
    ) -> ToolboxResult:
        if self._sandbox:
            return await self._exec_toolbox(tool_name, args)
        assert self._toolbox is not None
        fn = getattr(self._toolbox, tool_name)
        return fn(**args)

    # -- sync bridge --------------------------------------------------------

    def _sync_read_state(self, path: str, result: ToolboxResult) -> None:
        if result.is_error:
            return
        state_key = self._read_state_path(path)
        record_read(
            state_key,
            total_lines=result.total_lines,
            is_partial=result.is_partial,
            content_hash=result.content_hash,
        )

    def _read_state_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self._api.cwd, path))

    # -- read ---------------------------------------------------------------

    def _register_read(self) -> None:
        default_limit = self._config.default_limit
        self._api.register_tool(
            FunctionTool(
                name="read",
                description=(
                    "Read a UTF-8 text file from disk. "
                    f"Without offset/limit, shows up to {default_limit} lines; "
                    "longer files are truncated (use offset/limit to page). "
                    f"Files larger than {self._max_size_bytes} bytes require "
                    "offset and/or limit parameters. "
                    "Output starts with a header line giving the total or "
                    "shown line range, followed by 1-based line-numbered "
                    "content (`N\\tcontent`) — the same numbers the edit "
                    "tool's start_line/end_line refer to. "
                    "Known binary formats (images, archives, pdf, ...) are "
                    "rejected; inspect those via bash instead."
                ),
                parameters=_ReadArgs,
                fn=self._read_execute,
                metadata={"file_op": "read"},
            )
        )

    async def _read_execute(self, args: dict[str, Any]) -> ToolResult:
        path, arg_error = _required_string_arg(
            args, "path", "read",
            aliases=_PATH_ALIASES, hint='{"path": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert path is not None

        gate_error = _check_path_allowed(
            path, self._allow_globs, self._deny_globs
        )
        if gate_error is not None:
            return _error(gate_error)

        result = await self._dispatch("read", {
            "path": path,
            "offset": args.get("offset"),
            "limit": args.get("limit"),
        })
        self._sync_read_state(path, result)
        return _toolbox_to_tool_result(result)

    # -- write --------------------------------------------------------------

    def _register_write(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="write",
                description=(
                    "Write a UTF-8 text file. For existing files, you MUST read "
                    "the full file first. Prefer the edit tool for modifying "
                    "existing files — use write only for new files or complete "
                    "rewrites."
                ),
                parameters=_WriteArgs,
                fn=self._write_execute,
                metadata={"file_op": "write"},
            )
        )

    async def _write_execute(self, args: dict[str, Any]) -> ToolResult:
        path, arg_error = _required_string_arg(
            args, "path", "write",
            aliases=_PATH_ALIASES,
            hint='{"path": "...", "content": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert path is not None

        content, arg_error = _required_string_arg(
            args, "content", "write",
            allow_empty=True,
            hint='{"path": "...", "content": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert content is not None

        result = await self._dispatch("write", {
            "path": path, "content": content,
        })
        self._sync_read_state(path, result)
        return _toolbox_to_tool_result(result)

    # -- edit ---------------------------------------------------------------

    def _register_edit(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="edit",
                description=(
                    "Edit a UTF-8 text file. Two modes:\n"
                    "1. String replacement: provide old_string + new_string. "
                    "old_string must match exactly once unless replace_all "
                    "is set.\n"
                    "2. Line-range replacement: provide start_line + end_line "
                    "+ new_string (1-based, inclusive).\n"
                    "You MUST read the file first (both modes); an edit whose "
                    "prior read is stale or partial is rejected. An edit that "
                    "deletes many more lines than the match explains is also "
                    "rejected — use line-range mode for large intentional "
                    "deletions. Returns a line-numbered diff snippet of the "
                    "changed region."
                ),
                parameters=_EditArgs,
                fn=self._edit_execute,
                metadata={"file_op": "edit", TOOL_RESULT_FORMAT_METADATA_KEY: "diff"},
            )
        )

    async def _edit_execute(self, args: dict[str, Any]) -> ToolResult:
        path, arg_error = _required_string_arg(
            args, "path", "edit",
            aliases=_PATH_ALIASES,
            hint='{"path": "...", "old_string": "...", "new_string": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert path is not None

        new_string, arg_error = _required_string_arg(
            args, "new_string", "edit",
            allow_empty=True,
            hint='{"path": "...", "old_string": "...", "new_string": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert new_string is not None

        result = await self._dispatch("edit", {
            "path": path,
            "old_string": args.get("old_string"),
            "new_string": new_string,
            "start_line": args.get("start_line"),
            "end_line": args.get("end_line"),
            "replace_all": bool(args.get("replace_all", False)),
        })
        self._sync_read_state(path, result)
        return _toolbox_to_tool_result(result)
