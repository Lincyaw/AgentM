"""Flat-file contrib atom that exposes the Feishu / Lark CLI as a tool.

The atom registers a single ``lark`` tool that shells out to ``lark-cli``
(https://github.com/larksuite/cli) through ``BashOperations``. It does **not**
own credentials or transport: ``lark-cli`` reads its own env vars
(``LARKSUITE_CLI_APP_ID`` / ``_APP_SECRET`` / ``_USER_ACCESS_TOKEN`` /
``_TENANT_ACCESS_TOKEN``) or local keychain after ``lark-cli auth login``.
Custom credential / transport wrappers (per the Lark CLI extension docs)
are deployed by replacing the ``binary`` config with a wrapper executable
such as ``my-lark-cli``.

Safety model: by default only read-only invocations are allowed. Writes
must be explicitly enabled via ``allow_write: true`` in the scenario
config — typically combined with the ``permission`` atom for per-call
approval. Interactive subcommands (``auth login`` / ``auth logout`` /
``config init``) are always blocked: they require a TTY / browser and
will hang or mutate user-global state in ways the agent should not drive.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.operations import BashOperations
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


_DEFAULT_TIMEOUT_SECONDS: Final[float] = 60.0
_DEFAULT_BINARY: Final[str] = "lark-cli"

# Subcommands that mutate user-global state or require a TTY / browser.
# Always blocked, regardless of ``allow_write``.
_ALWAYS_BLOCKED: Final[frozenset[tuple[str, ...]]] = frozenset(
    {
        ("auth", "login"),
        ("auth", "logout"),
        ("config", "init"),
    }
)

# Read-only forms permitted when ``allow_write`` is False.
_READ_ONLY_AUTH: Final[frozenset[tuple[str, ...]]] = frozenset(
    {("auth", "status"), ("auth", "check")}
)
_READ_ONLY_TOPLEVEL: Final[frozenset[str]] = frozenset(
    {"help", "schema", "--help", "-h", "version", "--version"}
)


MANIFEST = ExtensionManifest(
    name="lark_cli",
    description="Expose `lark-cli` (Feishu / Lark CLI) as the `lark` tool.",
    registers=("tool:lark",),
    config_schema={
        "type": "object",
        "properties": {
            "binary": {
                "type": "string",
                "description": (
                    "Executable name or path. Override to use a wrapper "
                    "(e.g. './my-lark-cli') that injects custom Credential "
                    "or Transport providers."
                ),
                "default": _DEFAULT_BINARY,
            },
            "allow_write": {
                "type": "boolean",
                "description": (
                    "When false (default), only GET / status / schema / help "
                    "calls are permitted. Set true to allow POST/PUT/PATCH/"
                    "DELETE — combine with the `permission` atom for "
                    "per-call approval."
                ),
                "default": False,
            },
            "default_timeout": {
                "type": "number",
                "minimum": 0,
                "default": _DEFAULT_TIMEOUT_SECONDS,
            },
            "default_identity": {
                "type": ["string", "null"],
                "enum": ["user", "bot", None],
                "description": (
                    "Forwarded as `--as <identity>` when the call does not "
                    "specify one. Leave null to defer to lark-cli defaults."
                ),
                "default": None,
            },
        },
        "additionalProperties": False,
    },
    requires=(),
)


_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "argv": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": (
                "Argument vector passed to lark-cli, e.g. "
                '["api","GET","/open-apis/calendar/v4/calendars"] '
                'or ["schema","im.v1.message.create"]. '
                "Do NOT include the binary name itself."
            ),
        },
        "as_identity": {
            "type": "string",
            "enum": ["user", "bot"],
            "description": (
                "Maps to lark-cli `--as`. 'user' uses the authenticated "
                "user's UAT (their personal calendar / messages); 'bot' "
                "uses the app's TAT. Omit to use the atom default."
            ),
        },
        "dry_run": {
            "type": "boolean",
            "description": "Append `--dry-run` (resolves identity / URL but skips the request).",
            "default": False,
        },
        "stdin": {
            "type": "string",
            "description": "Optional stdin payload (e.g. JSON body for POST calls).",
        },
        "timeout": {"type": "number", "minimum": 0},
    },
    "required": ["argv"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    binary = str(config.get("binary", _DEFAULT_BINARY))
    allow_write = bool(config.get("allow_write", False))
    default_timeout = float(config.get("default_timeout", _DEFAULT_TIMEOUT_SECONDS))
    default_identity = config.get("default_identity")
    if default_identity is not None:
        default_identity = str(default_identity)

    bash_ops: BashOperations = api.get_operations().bash

    description = (
        "Invoke the Feishu / Lark CLI (`lark-cli`) to operate Feishu "
        "resources: messages, docs, calendar, bitable, mail, etc. "
        "Pass the lark-cli argv (without the binary). "
        "Use `schema <api>` to discover request shape before calling "
        "`api <METHOD> <path>`. "
        f"{'Write methods (POST/PUT/PATCH/DELETE) ENABLED.' if allow_write else 'Read-only mode: only GET / status / schema / help are accepted.'}"
    )

    async def _execute(args: dict[str, Any]) -> ToolResult:
        argv_raw = args.get("argv") or []
        if not isinstance(argv_raw, list) or not argv_raw:
            return _error("`argv` must be a non-empty array of strings.")
        argv = [str(a) for a in argv_raw]

        block_reason = _classify(argv, allow_write=allow_write)
        if block_reason is not None:
            return _error(block_reason)

        identity = args.get("as_identity") or default_identity
        full_argv: list[str] = [binary, *argv]
        if identity and not _has_flag(argv, "--as"):
            full_argv.extend(["--as", str(identity)])
        if bool(args.get("dry_run")) and not _has_flag(argv, "--dry-run"):
            full_argv.append("--dry-run")

        cmd = shlex.join(full_argv)
        stdin = args.get("stdin")
        if stdin is not None:
            cmd = f"printf %s {shlex.quote(str(stdin))} | {cmd}"

        timeout = float(args.get("timeout", default_timeout))
        try:
            result = await bash_ops.exec(cmd, cwd=api.cwd, timeout=timeout)
        except Exception as exc:
            return _error(f"Failed to invoke lark-cli: {exc}")

        stdout_text = result.stdout.decode("utf-8", errors="replace")
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        payload: dict[str, Any] = {
            "argv": full_argv,
            "exit_code": result.exit_code,
            "stdout": _maybe_json(stdout_text),
            "stderr": stderr_text,
            "timed_out": result.timed_out,
        }
        is_error = result.exit_code != 0 or result.timed_out
        text = json.dumps(payload, default=str, indent=2, sort_keys=True)
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=is_error,
        )

    api.register_tool(
        FunctionTool(
            name="lark",
            description=description,
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


def _classify(argv: list[str], *, allow_write: bool) -> str | None:
    """Return a block reason, or ``None`` to allow."""
    head2 = tuple(argv[:2])
    if head2 in _ALWAYS_BLOCKED:
        return (
            f"lark-cli `{' '.join(head2)}` is interactive / user-global and "
            "must be run by a human in a terminal — refusing to invoke."
        )

    if allow_write:
        return None

    top = argv[0]
    if top in _READ_ONLY_TOPLEVEL:
        return None
    if head2 in _READ_ONLY_AUTH:
        return None
    if top == "api":
        if len(argv) < 2:
            return "lark-cli `api` requires a method (GET/POST/...)."
        method = argv[1].upper()
        if method == "GET":
            return None
        return (
            f"lark-cli `api {method}` is a write call; this atom is in "
            "read-only mode (set `allow_write: true` in the scenario "
            "config to enable, ideally guarded by the `permission` atom)."
        )

    return (
        f"lark-cli subcommand `{top}` is not on the read-only allowlist; "
        "set `allow_write: true` to permit it."
    )


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _maybe_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[0] in "{[":
        try:
            return json.loads(stripped)
        except ValueError:
            pass
    return text


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
