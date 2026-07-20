from __future__ import annotations

from collections.abc import Callable

import pytest

from agentm.core.abi.operations import ExecResult
from agentm.extensions.builtin.tool_bash import _BashTool


class _FakeBashOperations:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        on_data: Callable[[bytes], None] | None = None,
        signal: object | None = None,
        log_path: str | None = None,
    ) -> ExecResult:
        del env, stdin, on_data, signal
        self.calls.append(
            {"cmd": cmd, "cwd": cwd, "timeout": timeout, "log_path": log_path}
        )
        return ExecResult(
            stdout=b"one\ntwo\n",
            stderr=b"warn\n",
            exit_code=7,
            timed_out=False,
        )


@pytest.mark.asyncio
async def test_bash_tool_writes_structured_result_extras() -> None:
    fake = _FakeBashOperations()
    tool = _BashTool(
        session=object(),  # type: ignore[arg-type]
        bash_ops=fake,  # type: ignore[arg-type]
        default_timeout=120.0,
        parameters={},
    )

    result = await tool._execute_with(  # noqa: SLF001
        {"cmd": "make test", "timeout": 3.5},
        bash_ops=fake,  # type: ignore[arg-type]
        cwd="/workspace",
        signal=None,
    )

    assert result.is_error
    assert result.extras["exit_code"] == 7
    assert isinstance(result.extras["duration_ms"], int)
    assert result.extras["timed_out"] is False
    assert result.extras["timeout_s"] == 3.5
    assert result.extras["stdout"] == "one\ntwo\n"
    assert result.extras["stderr"] == "warn\n"
    assert result.extras["stdout_lines"] == 3
    assert result.extras["stderr_lines"] == 2
    assert fake.calls == [
        {"cmd": "make test", "cwd": "/workspace", "timeout": 3.5, "log_path": None}
    ]
