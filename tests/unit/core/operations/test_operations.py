from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time

import pytest

from agentm.core.operations import LocalBashOperations, LocalFileOperations


@pytest.mark.asyncio
async def test_local_file_operations_roundtrip(tmp_path: Path) -> None:
    ops = LocalFileOperations()
    target = tmp_path / "payload.bin"

    payload = b"hello\x00world"
    await ops.write_file(str(target), payload)

    assert await ops.read_file(str(target)) == payload
    assert await ops.list_dir(str(tmp_path)) == ["payload.bin"]
    assert await ops.access(str(target)) is True


@pytest.mark.asyncio
async def test_local_file_operations_missing_file(tmp_path: Path) -> None:
    ops = LocalFileOperations()
    missing = tmp_path / "missing.bin"

    assert await ops.access(str(missing)) is False
    with pytest.raises(FileNotFoundError):
        await ops.read_file(str(missing))


@pytest.mark.asyncio
async def test_local_bash_operations_success(tmp_path: Path) -> None:
    ops = LocalBashOperations()

    result = await ops.exec("echo hello", cwd=str(tmp_path))

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.startswith(b"hello")
    assert result.stderr == b""


@pytest.mark.asyncio
async def test_local_bash_operations_timeout_reaps_process(tmp_path: Path) -> None:
    ops = LocalBashOperations()
    pid_file = tmp_path / "child.pid"

    start = time.monotonic()
    result = await ops.exec(
        f"sh -c 'echo $$ > {pid_file}; sleep 5'",
        cwd=str(tmp_path),
        timeout=0.1,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    assert result.timed_out is True
    assert result.exit_code != 0

    child_pid = int(pid_file.read_text().strip())
    with pytest.raises(ChildProcessError):
        os.waitpid(child_pid, os.WNOHANG)


@pytest.mark.asyncio
async def test_local_bash_operations_signal_abort(tmp_path: Path) -> None:
    ops = LocalBashOperations()
    abort = asyncio.Event()

    async def trigger_abort() -> None:
        await asyncio.sleep(0.1)
        abort.set()

    trigger_task = asyncio.create_task(trigger_abort())
    start = time.monotonic()
    result = await ops.exec(
        "sleep 5",
        cwd=str(tmp_path),
        signal=abort,
    )
    elapsed = time.monotonic() - start
    await trigger_task

    assert elapsed < 1.0
    assert result.exit_code != 0
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_local_bash_operations_streams_stdout_chunks(tmp_path: Path) -> None:
    ops = LocalBashOperations()
    chunks: list[bytes] = []

    result = await ops.exec(
        "printf 'a\nb\nc\n'",
        cwd=str(tmp_path),
        on_data=chunks.append,
    )

    assert chunks
    assert b"".join(chunks) == result.stdout
    assert result.stdout == b"a\nb\nc\n"
