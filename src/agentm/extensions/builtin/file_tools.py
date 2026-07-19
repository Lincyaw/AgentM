"""Grouped local file-I/O tool atom: ``read``, ``write``, ``edit``."""

from __future__ import annotations

import fnmatch
import json
import os
import shlex
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentm.core.abi import (
    AtomInstallPriority,
    BashOperations,
    EnvironmentOperations,
    ENVIRONMENT_OPERATIONS_SERVICE,
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
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tools: list[str] | None = None
    allow_globs: list[str] | None = None
    deny_globs: list[str] | None = None
    max_size_bytes: int = 262_144
    require_read: bool = True
    default_limit: int = 250
    verify_readback: bool = False
    toolbox_command: str = "python3 -m agentm_toolbox"
    toolbox_state_file: str | None = None

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


def _write_result_text(path: str, verb: str) -> str:
    return f"{verb}: {path}"


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


def _resolved(path: str, cwd: str) -> str:
    try:
        raw = Path(path).expanduser()
        candidate = raw if raw.is_absolute() else Path(cwd) / raw
        return str(candidate.resolve(strict=False))
    except (OSError, RuntimeError):
        raw_text = os.path.expanduser(path)
        if os.path.isabs(raw_text):
            return os.path.abspath(raw_text)
        return os.path.abspath(os.path.join(cwd, raw_text))


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


def _apply_text_edit(
    old_text: str,
    args: dict[str, Any],
    new_string: str,
) -> str | ToolResult:
    old_string = args.get("old_string")
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    has_string_mode = old_string is not None
    has_line_mode = start_line is not None or end_line is not None
    if has_string_mode and has_line_mode:
        return _error(
            "Invalid edit call: old_string mode is mutually exclusive with "
            "start_line/end_line mode."
        )
    if has_string_mode:
        if not isinstance(old_string, str) or old_string == "":
            return _error("Invalid edit call: old_string must be a non-empty string.")
        count = old_text.count(old_string)
        replace_all = bool(args.get("replace_all", False))
        if count == 0:
            return _error("Edit failed: old_string was not found.")
        if not replace_all and count != 1:
            return _error(
                f"Edit failed: old_string matched {count} times; set replace_all "
                "or provide a more specific string."
            )
        return old_text.replace(
            old_string,
            new_string,
            -1 if replace_all else 1,
        )
    if has_line_mode:
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            return _error(
                "Invalid edit call: start_line and end_line must both be integers."
            )
        lines = old_text.splitlines(keepends=True)
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            return _error(
                "Invalid edit call: line range must be 1-based, inclusive, "
                "and within the current resource."
            )
        replacement = new_string
        if (
            replacement
            and not replacement.endswith("\n")
            and lines[end_line - 1].endswith("\n")
        ):
            replacement += "\n"
        return "".join(lines[: start_line - 1] + [replacement] + lines[end_line:])
    return _error(
        "Invalid edit call: provide either old_string or start_line/end_line."
    )


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


def install(session: Any, config: FileToolsConfig) -> None:
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


def _coerce_bash_ops(session: Any) -> BashOperations | None:
    service = session.services.get("operations:bash")
    if isinstance(service, BashOperations):
        return service
    return None


def _coerce_environment_id(session: Any) -> str | None:
    service = session.services.get(ENVIRONMENT_OPERATIONS_SERVICE)
    if isinstance(service, EnvironmentOperations):
        ref = service.ref
        return f"{ref.kind}:{ref.id}"
    return None


class _FileToolsRuntime:
    """Owns file_tools registration and per-session handler state."""

    def __init__(self, *, session: Any, config: FileToolsConfig) -> None:
        self._session = session
        self._config = config
        self._enabled_tools = _enabled_tools(config.tools)
        self._allow_globs = _coerce_globs(config.allow_globs, session.ctx.cwd)
        self._deny_globs = _coerce_globs(config.deny_globs, session.ctx.cwd)
        self._max_size_bytes = config.max_size_bytes
        self._bash_ops = _coerce_bash_ops(session)
        self._environment_id = _coerce_environment_id(session)
        self._toolbox_command = config.toolbox_command
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

    async def _dispatch(
        self, tool_name: str, args: dict[str, Any]
    ) -> ToolboxResult:
        if self._bash_ops is not None:
            return await self._exec_toolbox(tool_name, args)
        fn = getattr(self._toolbox, tool_name)
        return fn(**args)

    def _resource_writer(self) -> ResourceWriter | None:
        getter = getattr(self._session, "get_resource_writer", None)
        if not callable(getter):
            return None
        writer = getter()
        return writer if isinstance(writer, ResourceWriter) else None

    def _resource_txn(self) -> ResourceTxn | None:
        getter = getattr(self._session, "get_resource_txn", None)
        if not callable(getter):
            return None
        txn = getter()
        return txn if isinstance(txn, ResourceTxn) else None

    async def _resource_write(
        self,
        path: str,
        content: str,
        *,
        rationale: str,
    ) -> ToolResult | None:
        writer = self._resource_writer()
        txn = self._resource_txn()
        if writer is None and txn is None:
            return None
        data = content.encode("utf-8")
        ref = ResourceRef(namespace="workspace", path=path)
        if txn is not None:
            mutation = await txn.write(ref, data, rationale=rationale)
            return _ok(_write_result_text(mutation.ref.path, "queued write"))
        assert writer is not None
        result = await writer.write(path, data, rationale=rationale)
        if result.error is not None:
            return _error(result.error)
        return _ok(_write_result_text(result.path, "wrote"))

    async def _resource_edit(
        self,
        path: str,
        args: dict[str, Any],
        *,
        new_string: str,
        rationale: str,
    ) -> ToolResult | None:
        writer = self._resource_writer()
        txn = self._resource_txn()
        if writer is None and txn is None:
            return None
        if writer is None:
            return _error("edit requires a ResourceWriter for reading current content")
        try:
            old_bytes = await writer.read(path)
        except Exception as exc:  # noqa: BLE001
            return _error(f"read before edit failed: {exc}")
        try:
            old_text = old_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            return _error(f"edit only supports UTF-8 text resources: {exc}")
        edited = _apply_text_edit(old_text, args, new_string)
        if isinstance(edited, ToolResult):
            return edited
        new_bytes = edited.encode("utf-8")
        ref = ResourceRef(namespace="workspace", path=path)
        if txn is not None:
            mutation = await txn.replace(
                ref,
                old_bytes,
                new_bytes,
                rationale=rationale,
            )
            return _ok(_write_result_text(mutation.ref.path, "queued edit"))
        result = await writer.replace(path, old_bytes, new_bytes, rationale=rationale)
        if result.error is not None:
            return _error(result.error)
        return _ok(_write_result_text(result.path, "edited"))

    async def _exec_toolbox(
        self, tool_name: str, args: dict[str, Any]
    ) -> ToolboxResult:
        bash_ops = self._bash_ops
        if bash_ops is None:
            return ToolboxResult(
                text=f"toolbox exec unavailable for {tool_name!r}: no bash service",
                is_error=True,
            )
        exec_args = {
            **args,
            "_cwd": self._session.ctx.cwd,
            "_max_size": self._max_size_bytes,
            "_require_read": self._config.require_read,
            "_default_limit": self._config.default_limit,
            "_state_namespace": self._toolbox_state_namespace(),
        }
        if self._config.toolbox_state_file:
            exec_args["_state_file"] = self._config.toolbox_state_file
        payload = json.dumps(exec_args, ensure_ascii=False).encode("utf-8")
        command = f"{self._toolbox_command} {shlex.quote(tool_name)} -"
        result = await bash_ops.exec(
            command,
            cwd=self._session.ctx.cwd,
            timeout=30,
            stdin=payload,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        json_line = next(
            (line.strip() for line in reversed(stdout.splitlines()) if line.strip()),
            "",
        )
        if not json_line:
            details = stderr.strip() or stdout.strip() or "no output"
            if result.timed_out:
                details = f"timed out; {details}"
            return ToolboxResult(
                text=f"toolbox exec failed for {tool_name!r}: {details}",
                is_error=True,
            )
        try:
            decoded = json.loads(json_line)
            return ToolboxResult(**decoded)
        except (TypeError, json.JSONDecodeError) as exc:
            details = stderr.strip() or stdout.strip() or json_line
            return ToolboxResult(
                text=(
                    f"toolbox output parse error for {tool_name!r}: {exc}; "
                    f"output: {details}"
                ),
                is_error=True,
            )

    def _toolbox_state_namespace(self) -> str:
        return json.dumps(
            {
                "root_session_id": self._session.ctx.root_session_id,
                "session_id": self._session.ctx.session_id,
                "environment_id": self._environment_id,
                "cwd": self._session.ctx.cwd,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

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

    async def _read_execute(self, args: dict[str, Any]) -> ToolResult:
        path, arg_error = _required_string_arg(
            args, "path", "read",
            aliases=_PATH_ALIASES, hint='{"path": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert path is not None

        gate_error = _check_path_allowed(
            path, self._session.ctx.cwd, self._allow_globs, self._deny_globs
        )
        if gate_error is not None:
            return _error(gate_error)

        result = await self._dispatch("read", {
            "path": path,
            "offset": args.get("offset"),
            "limit": args.get("limit"),
        })
        return _toolbox_to_tool_result(result)

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

        gate_error = _check_path_allowed(
            path, self._session.ctx.cwd, self._allow_globs, self._deny_globs
        )
        if gate_error is not None:
            return _error(gate_error)

        rationale = str(args.get("rationale") or "agent write via file_tools")
        resource_result = await self._resource_write(
            path,
            content,
            rationale=rationale,
        )
        if resource_result is not None:
            return resource_result

        result = await self._dispatch("write", {
            "path": path, "content": content,
        })
        return _toolbox_to_tool_result(result)

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
                execution_requirements=ToolExecutionRequirements(filesystem="write"),
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

        gate_error = _check_path_allowed(
            path, self._session.ctx.cwd, self._allow_globs, self._deny_globs
        )
        if gate_error is not None:
            return _error(gate_error)

        rationale = str(args.get("rationale") or "agent edit via file_tools")
        resource_result = await self._resource_edit(
            path,
            args,
            new_string=new_string,
            rationale=rationale,
        )
        if resource_result is not None:
            return resource_result

        result = await self._dispatch("edit", {
            "path": path,
            "old_string": args.get("old_string"),
            "new_string": new_string,
            "start_line": args.get("start_line"),
            "end_line": args.get("end_line"),
            "replace_all": bool(args.get("replace_all", False)),
        })
        return _toolbox_to_tool_result(result)
