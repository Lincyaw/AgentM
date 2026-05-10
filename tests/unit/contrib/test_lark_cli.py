"""Tests for the ``contrib/extensions/lark_cli.py`` atom.

The atom shells out to ``lark-cli``; we substitute a stub ``BashOperations``
that records the executed command line so tests stay hermetic.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agentm.core.abi.operations import ExecResult
from contrib.extensions import lark_cli


class _StubBash:
    def __init__(self, exit_code: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[dict[str, Any]] = []

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_data: Any = None,
        signal: Any = None,
    ) -> ExecResult:
        self.calls.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
        return ExecResult(
            stdout=self.stdout, stderr=self.stderr, exit_code=self.exit_code, timed_out=False
        )


class _Api:
    def __init__(self, bash: _StubBash, cwd: str = "/tmp") -> None:
        self.cwd = cwd
        self._bash = bash
        self.tools: dict[str, Any] = {}

    def register_tool(self, tool: Any) -> None:
        self.tools[tool.name] = tool

    def get_operations(self) -> Any:
        class _Ops:
            bash = self._bash
        return _Ops()


def _install(config: dict[str, Any] | None = None) -> tuple[_Api, _StubBash]:
    bash = _StubBash(stdout=b'{"code":0,"data":{}}')
    api = _Api(bash)
    lark_cli.install(api, config or {})  # type: ignore[arg-type]
    return api, bash


def _call(api: _Api, args: dict[str, Any]) -> tuple[dict[str, Any] | str, bool]:
    result = asyncio.run(api.tools["lark"].fn(args))
    text = result.content[0].text
    try:
        return json.loads(text), bool(result.is_error)
    except ValueError:
        return text, bool(result.is_error)


def test_registers_lark_tool() -> None:
    api, _ = _install()
    assert "lark" in api.tools
    assert "argv" in api.tools["lark"].parameters["properties"]


def test_read_only_get_passes_through() -> None:
    api, bash = _install()
    payload, is_err = _call(
        api,
        {"argv": ["api", "GET", "/open-apis/calendar/v4/calendars"]},
    )
    assert not is_err
    assert isinstance(payload, dict)
    assert bash.calls, "bash.exec should have been invoked"
    cmd = bash.calls[0]["cmd"]
    assert cmd.startswith("lark-cli api GET ")
    assert "/open-apis/calendar/v4/calendars" in cmd


def test_read_only_blocks_post() -> None:
    api, bash = _install()
    payload, is_err = _call(
        api,
        {"argv": ["api", "POST", "/open-apis/im/v1/messages"]},
    )
    assert is_err
    assert not bash.calls
    assert isinstance(payload, str)
    assert "read-only" in payload.lower()


def test_allow_write_permits_post() -> None:
    api, bash = _install({"allow_write": True})
    _, is_err = _call(api, {"argv": ["api", "POST", "/open-apis/im/v1/messages"]})
    assert not is_err
    assert bash.calls


def test_auth_login_always_blocked_even_with_allow_write() -> None:
    api, bash = _install({"allow_write": True})
    payload, is_err = _call(api, {"argv": ["auth", "login"]})
    assert is_err
    assert not bash.calls
    assert isinstance(payload, str)
    assert "auth login" in payload


def test_auth_status_is_read_only_allowed() -> None:
    api, bash = _install()
    _, is_err = _call(api, {"argv": ["auth", "status"]})
    assert not is_err
    assert bash.calls


def test_schema_subcommand_allowed_in_read_only() -> None:
    api, bash = _install()
    _, is_err = _call(api, {"argv": ["schema", "im.v1.message.create"]})
    assert not is_err
    assert bash.calls


def test_identity_flag_appended_when_default_set() -> None:
    api, bash = _install({"default_identity": "user"})
    _call(api, {"argv": ["api", "GET", "/x"]})
    assert "--as user" in bash.calls[0]["cmd"]


def test_identity_arg_overrides_default_and_not_duplicated_when_explicit() -> None:
    api, bash = _install({"default_identity": "user"})
    _call(api, {"argv": ["api", "GET", "/x", "--as", "bot"]})
    cmd = bash.calls[0]["cmd"]
    # Atom must not append a second --as when the call already carries one.
    assert cmd.count("--as") == 1
    assert "--as bot" in cmd


def test_dry_run_appends_flag() -> None:
    api, bash = _install()
    _call(api, {"argv": ["api", "GET", "/x"], "dry_run": True})
    assert "--dry-run" in bash.calls[0]["cmd"]


def test_custom_binary_used() -> None:
    api, bash = _install({"binary": "./my-lark-cli"})
    _call(api, {"argv": ["auth", "status"]})
    assert bash.calls[0]["cmd"].startswith("./my-lark-cli ")


def test_stdin_piped_via_printf() -> None:
    api, bash = _install({"allow_write": True})
    _call(
        api,
        {
            "argv": ["api", "POST", "/open-apis/im/v1/messages"],
            "stdin": '{"msg":"hi"}',
        },
    )
    cmd = bash.calls[0]["cmd"]
    assert cmd.startswith("printf %s ")
    assert "| lark-cli api POST" in cmd


def test_empty_argv_rejected() -> None:
    api, _ = _install()
    payload, is_err = _call(api, {"argv": []})
    assert is_err
    assert isinstance(payload, str)
    assert "argv" in payload


def test_unknown_subcommand_blocked_in_read_only() -> None:
    api, bash = _install()
    payload, is_err = _call(api, {"argv": ["bitable", "list"]})
    assert is_err
    assert not bash.calls
    assert isinstance(payload, str)
    assert "allow_write" in payload
