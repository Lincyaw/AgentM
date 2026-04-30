# Task: Phase 2 Group A0 — Operations Ports (FileOps / BashOps)

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §10b.6
**Architecture**: [pluggable-architecture.md](../designs/pluggable-architecture.md) §3.2 (acceptance scenario 2 — bash over SSH)
**Agent**: implementer (sonnet)
**Status**: READY (sequential prerequisite — Group D is BLOCKED on this)

## Why this is mandatory before Group D

Acceptance scenario 2 in `pluggable-architecture.md` §6 demands "run `bash` tool over SSH" without forking core. If `general_purpose`'s tools call `subprocess` / `pathlib` directly, that scenario fails. A0 introduces the port boundary so swapping `LocalBashOperations` for `SshBashOperations` is a one-line config change.

## Scope

ONE new module: `src/agentm/core/operations.py`. Defines two `Protocol`s and two default impls.

```python
# src/agentm/core/operations.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable
import asyncio

@dataclass(frozen=True, slots=True)
class ExecResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool

class FileOperations(Protocol):
    async def read_file(self, path: str) -> bytes: ...
    async def write_file(self, path: str, content: bytes) -> None: ...
    async def access(self, path: str) -> bool: ...   # True if path exists & readable
    async def list_dir(self, path: str) -> list[str]: ...

class BashOperations(Protocol):
    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_data: Callable[[bytes], None] | None = None,
        signal: asyncio.Event | None = None,
    ) -> ExecResult: ...

class LocalFileOperations:
    """Default impl — stdlib `pathlib`, async via `asyncio.to_thread`."""
    ...

class LocalBashOperations:
    """Default impl — `asyncio.create_subprocess_shell`."""
    ...
```

### Behavior requirements

- `LocalBashOperations.exec`:
  - Honor `timeout` (kill the process group on overrun, set `timed_out=True`).
  - Honor `signal` (asyncio.Event from caller; on set, terminate process).
  - Stream stdout chunks to `on_data` if provided; always also accumulate into `stdout`.
  - Capture stderr separately.
- `LocalFileOperations`:
  - All methods run via `asyncio.to_thread` (no blocking I/O on the event loop).
  - `read_file`/`write_file` are byte-level (UTF-8 decoding is the tool's job, not the port's).
  - `access` returns True iff the path exists AND the process can read it.

### Constructor API

`LocalFileOperations()` and `LocalBashOperations()` take no args. They are stateless. Future `SshBashOperations(host, user, key_path)` will fit the same Protocol.

## HARD constraints

- Imports allowed: stdlib only.
- **No** dependency on `agentm.harness.*`, `agentm.extensions.*`, or `agentm.llm.*`. This module is core-layer.
- `from __future__ import annotations` at top.
- Module docstring references `extension-as-scenario.md` §10b.6.

## Tests

`tests/unit/core/operations/test_operations.py`:

1. **LocalFileOperations roundtrip**: write bytes to a `tmp_path` file, read them back, assert equality.
2. **LocalFileOperations missing file**: `access` returns False on a non-existent path; `read_file` raises `FileNotFoundError`.
3. **LocalBashOperations success**: `exec("echo hello")` returns `exit_code=0`, stdout starts with `b"hello"`.
4. **LocalBashOperations timeout**: `exec("sleep 5", timeout=0.1)` returns `timed_out=True` within 0.5s wall clock; the spawned process is reaped (assert no zombie via `os.waitpid` timing).
5. **LocalBashOperations signal**: spawn `sleep 5`, set the abort signal after 0.1s, assert the call returns within 1s with non-zero exit_code or `timed_out=False` + killed.
6. **on_data streaming**: `exec("printf 'a\\nb\\nc\\n'", on_data=collector.append)` invokes the callback at least once and the concatenated chunks equal final stdout.

## Quality gates

```bash
uv run ruff check src/agentm/core/operations.py tests/unit/core/operations/
uv run mypy src/agentm/core/operations.py
uv run pytest tests/unit/core/operations/ tests/unit/kernel/ tests/unit/harness_v2/ tests/unit/llm/ -q
```

All green; existing 60 tests stay green.

## Report format (≤200 words)

1. File added (absolute path) + LoC.
2. Test count + any flake risk on timeout/signal tests (these can be racy — note the wall-clock budget you used).
3. Any deviation from the Protocol shapes above and why.
4. Confirmation: zero imports of `harness`, `extensions`, `llm`.
