"""Grouped file-I/O tools over the host-provided resource authority."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    ResourceRef,
    ResourceTxn,
    ResourceWriter,
    TOOL_RESULT_FORMAT_METADATA_KEY,
    TextContent,
    ToolExecutionRequirements,
    ToolResult,
)
from agentm.extensions import ExtensionManifest

from agentm_toolbox import FileToolbox
from agentm_toolbox._file_ops import Result as ToolboxResult

# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

_ALL_TOOLS: Final[frozenset[str]] = frozenset({"read", "write", "edit"})


class FileToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tools: list[str] | None = None
    allow_globs: list[str] | None = None
    deny_globs: list[str] | None = None
    max_size_bytes: int = Field(default=262_144, gt=0)
    require_read: bool = True
    default_limit: int = Field(default=250, gt=0)

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
    requires=("service:resource_writer",),
    priority=AtomInstallPriority.TOOL,
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


def _invalid_call(tool_name: str, error: ValidationError) -> ToolResult:
    return _error(f"Invalid {tool_name} call: {error}")


def _resolve_globs(value: list[str] | None, cwd: str) -> tuple[str, ...]:
    if value is None:
        return ()
    resolved: list[str] = []
    for raw in value:
        if not raw:
            raise ValueError("file tool glob patterns must not be empty")
        if os.path.isabs(raw):
            resolved.append(raw)
        else:
            resolved.append(os.path.normpath(os.path.join(cwd, raw)))
    return tuple(resolved)


def _resolved(path: str, cwd: str) -> str:
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else Path(cwd) / raw
    return str(candidate.resolve(strict=False))


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _check_path_allowed(
    path: str,
    cwd: str,
    allow: tuple[str, ...],
    deny: tuple[str, ...],
) -> str | None:
    resolved = _resolved(path, cwd)
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
# install()
# ---------------------------------------------------------------------------


def install(session: AtomAPI, config: FileToolsConfig) -> None:
    _FileToolsRuntime(session=session, config=config).install()


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

    def __init__(self, *, session: AtomAPI, config: FileToolsConfig) -> None:
        self._session = session
        self._config = config
        self._enabled_tools = _enabled_tools(config.tools)
        self._allow_globs = _resolve_globs(config.allow_globs, session.ctx.cwd)
        self._deny_globs = _resolve_globs(config.deny_globs, session.ctx.cwd)
        self._max_size_bytes = config.max_size_bytes
        writer = session.get_resource_writer()
        if writer is None:
            raise RuntimeError(
                "file_tools requires a ResourceWriter; compose a resource atom "
                "or inject a host ResourceWriter"
            )
        self._writer = writer
        self._toolbox = FileToolbox(
            cwd=session.ctx.cwd,
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

    def _resource_txn(self) -> ResourceTxn | None:
        return self._session.get_resource_txn()

    async def _resource_view(
        self,
        path: str,
    ) -> tuple[ResourceTxn | ResourceWriter, bytes | None]:
        ref = ResourceRef(namespace="workspace", path=path)
        txn = self._resource_txn()
        if txn is not None:
            return txn, await txn.read(ref)
        if not await self._writer.exists(path):
            return self._writer, None
        return self._writer, await self._writer.read(path)

    async def _resource_read(
        self,
        path: str,
        *,
        offset: int | None,
        limit: int | None,
    ) -> ToolResult:
        _, current = await self._resource_view(path)
        if current is None:
            return _error(f"Failed to read {path!r}: file does not exist")
        return _toolbox_to_tool_result(
            self._toolbox.read_bytes(
                path,
                current,
                offset=offset,
                limit=limit,
            )
        )

    async def _resource_write(
        self,
        path: str,
        content: str,
        *,
        rationale: str,
    ) -> ToolResult:
        authority, current = await self._resource_view(path)
        result, data = self._toolbox.plan_write(path, current, content)
        if result.is_error or data is None:
            return _toolbox_to_tool_result(result)
        ref = ResourceRef(namespace="workspace", path=path)
        if isinstance(authority, ResourceTxn):
            if current is None:
                await authority.create(ref, data, rationale=rationale)
            else:
                await authority.replace(
                    ref,
                    current,
                    data,
                    rationale=rationale,
                )
        elif current is None:
            write_result = await authority.write(path, data, rationale=rationale)
            if write_result.error is not None:
                return _error(write_result.error)
        else:
            write_result = await authority.replace(
                path,
                current,
                data,
                rationale=rationale,
            )
            if write_result.error is not None:
                return _error(write_result.error)
        self._toolbox.accept_content(path, data)
        return _toolbox_to_tool_result(result)

    async def _resource_edit(
        self,
        path: str,
        args: _EditArgs,
    ) -> ToolResult:
        authority, current = await self._resource_view(path)
        result, new_bytes = self._toolbox.plan_edit(
            path,
            current,
            old_string=args.old_string,
            new_string=args.new_string,
            start_line=args.start_line,
            end_line=args.end_line,
            replace_all=args.replace_all,
        )
        if result.is_error or new_bytes is None:
            return _toolbox_to_tool_result(result)
        if current is None:
            return _error(f"Failed to read {path!r}: file does not exist")
        ref = ResourceRef(namespace="workspace", path=path)
        if isinstance(authority, ResourceTxn):
            await authority.replace(
                ref,
                current,
                new_bytes,
                rationale=args.rationale,
            )
        else:
            write_result = await authority.replace(
                path,
                current,
                new_bytes,
                rationale=args.rationale,
            )
            if write_result.error is not None:
                return _error(write_result.error)
        self._toolbox.accept_content(path, new_bytes, read_ranges=result.read_ranges)
        return _toolbox_to_tool_result(result)

    # -- read ---------------------------------------------------------------

    def _register_read(self) -> None:
        default_limit = self._config.default_limit
        self._session.register_tool(
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
                execution_requirements=ToolExecutionRequirements(filesystem="read"),
            )
        )

    async def _read_execute(self, args: dict[str, object]) -> ToolResult:
        try:
            parsed = _ReadArgs.model_validate(args)
        except ValidationError as error:
            return _invalid_call("read", error)

        gate_error = _check_path_allowed(
            parsed.path,
            self._session.ctx.cwd,
            self._allow_globs,
            self._deny_globs,
        )
        if gate_error is not None:
            return _error(gate_error)

        return await self._resource_read(
            parsed.path,
            offset=parsed.offset,
            limit=parsed.limit,
        )

    # -- write --------------------------------------------------------------

    def _register_write(self) -> None:
        self._session.register_tool(
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
                execution_requirements=ToolExecutionRequirements(filesystem="write"),
            )
        )

    async def _write_execute(self, args: dict[str, object]) -> ToolResult:
        try:
            parsed = _WriteArgs.model_validate(args)
        except ValidationError as error:
            return _invalid_call("write", error)

        gate_error = _check_path_allowed(
            parsed.path,
            self._session.ctx.cwd,
            self._allow_globs,
            self._deny_globs,
        )
        if gate_error is not None:
            return _error(gate_error)

        return await self._resource_write(
            parsed.path,
            parsed.content,
            rationale=parsed.rationale,
        )

    # -- edit ---------------------------------------------------------------

    def _register_edit(self) -> None:
        self._session.register_tool(
            FunctionTool(
                name="edit",
                description=(
                    "Edit a UTF-8 text file. Two modes:\n"
                    "1. String replacement: provide old_string + new_string. "
                    "old_string must match exactly once unless replace_all "
                    "is set.\n"
                    "2. Line-range replacement: provide start_line + end_line "
                    "+ new_string (1-based, inclusive).\n"
                    "You MUST read the affected lines first (both modes); "
                    "a full-file read is not required for targeted edits. "
                    "An edit whose prior read is stale or does not cover the "
                    "edited lines is rejected. An edit that "
                    "deletes many more lines than the match explains is also "
                    "rejected — use line-range mode for large intentional "
                    "deletions. Returns a line-numbered diff snippet of the "
                    "changed region."
                ),
                parameters=_EditArgs,
                fn=self._edit_execute,
                metadata={"file_op": "edit", TOOL_RESULT_FORMAT_METADATA_KEY: "diff"},
                execution_requirements=ToolExecutionRequirements(filesystem="write"),
            )
        )

    async def _edit_execute(self, args: dict[str, object]) -> ToolResult:
        try:
            parsed = _EditArgs.model_validate(args)
        except ValidationError as error:
            return _invalid_call("edit", error)

        gate_error = _check_path_allowed(
            parsed.path,
            self._session.ctx.cwd,
            self._allow_globs,
            self._deny_globs,
        )
        if gate_error is not None:
            return _error(gate_error)

        return await self._resource_edit(parsed.path, parsed)
