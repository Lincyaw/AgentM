"""Tool atom for the ``extensions.builtin.tool_bash`` §7.1 row.

Besides the ``bash`` tool itself, this atom owns the
``bash_output_tails`` session service (:class:`BashOutputTails`): rolling
tail buffers of live exec output, streamed via ``BashOperations.exec``'s
``on_data`` callback. A consumer (``background_exec``) binds an opaque key
around a call; when the bash tool sees a bound key it streams output into
a bounded buffer under that key, which the consumer can read while the
command is still running. No key bound → no buffering.
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from contextvars import ContextVar, Token
from typing import Callable, Final, cast

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentm.core.abi import (
    AtomInstallPriority,
    BASH_OPERATIONS_SERVICE,
    AtomAPI,
    BashOperations,
    CancelSignal,
    EnvironmentOperations,
    TextContent,
    ToolExecutionRequirements,
    ToolResult,
)
from agentm.core.abi.services import ServiceNotFound, ServiceTypeMismatch
from agentm.core.abi.tool_executor import EnvironmentExecutableTool
from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest

_DEFAULT_TIMEOUT_SECONDS: Final[float] = 120.0

# Session-service name; consumed by background_exec via api.get_service.
BASH_OUTPUT_TAILS_SERVICE: Final[str] = "bash_output_tails"
_TAIL_MAX_BYTES: Final[int] = 4096


class _TailState:
    __slots__ = ("buffer", "last_data_at")

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.last_data_at: float | None = None


class BashOutputTails:
    """Rolling tail buffers for in-flight bash commands, keyed opaquely.

    Contract with consumers: ``bind(key, log_path=…)`` a contextvar before
    creating the task that runs the bash tool call (context propagates into
    the task), ``tail(key)`` / ``last_data_at(key)`` while it runs,
    ``discard(key)`` when done. When a ``log_path`` is bound, the bash tool
    passes it to ``BashOperations.exec`` so the full output is ALSO written
    to that file at the execution site (locally or inside the remote
    sandbox) — the tail buffer here stays bounded and holds only the last
    few KB for quick inspection. The registry never evicts on its own.
    """

    def __init__(self, max_bytes: int = _TAIL_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._current: ContextVar[str | None] = ContextVar(
            "bash_output_tail_key", default=None
        )
        self._tails: dict[str, _TailState] = {}
        self._log_paths: dict[str, str] = {}

    def bind(self, key: str, *, log_path: str | None = None) -> Token[str | None]:
        if log_path is not None:
            self._log_paths[key] = log_path
        return self._current.set(key)

    def unbind(self, token: Token[str | None]) -> None:
        self._current.reset(token)

    def current(self) -> str | None:
        return self._current.get()

    def log_path(self, key: str) -> str | None:
        return self._log_paths.get(key)

    def open(self, key: str) -> Callable[[bytes], None]:
        state = _TailState()
        self._tails[key] = state

        def feed(chunk: bytes) -> None:
            state.buffer.extend(chunk)
            if len(state.buffer) > self._max_bytes:
                del state.buffer[: len(state.buffer) - self._max_bytes]
            state.last_data_at = time.monotonic()

        return feed

    def tail(self, key: str) -> str | None:
        state = self._tails.get(key)
        if state is None or not state.buffer:
            return None
        return state.buffer.decode("utf-8", errors="replace")

    def last_data_at(self, key: str) -> float | None:
        state = self._tails.get(key)
        return state.last_data_at if state is not None else None

    def discard(self, key: str) -> None:
        self._tails.pop(key, None)
        self._log_paths.pop(key, None)


class ToolBashConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    default_timeout: float = Field(
        default=_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        allow_inf_nan=False,
    )


class _BashArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cmd: str = Field(min_length=1, description="Shell command to execute.")
    timeout: float = Field(
        default=_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Max seconds the command may run before it is killed and the "
            "result is flagged TIMED OUT."
        ),
    )


MANIFEST = ExtensionManifest(
    name="tool_bash",
    description="Register the bash tool backed by BashOperations.",
    registers=("tool:bash",),
    config_schema=ToolBashConfig,
    requires=("service:operations:bash",),
    priority=AtomInstallPriority.TOOL,
)


class _ToolBashRuntime:
    def __init__(self, session: AtomAPI, config: ToolBashConfig) -> None:
        self._session = session
        _require_bash_ops(session)
        self._default_timeout = config.default_timeout

    def install(self) -> None:
        tails = BashOutputTails()
        self._session.services.register(
            BASH_OUTPUT_TAILS_SERVICE,
            tails,
            scope="session",
        )
        self._session.register_tool(
            _BashTool(
                session=self._session,
                default_timeout=self._default_timeout,
                parameters=pydantic_to_tool_schema(_BashArgs),
                tails=tails,
            )
        )


def install(session: AtomAPI, config: ToolBashConfig) -> None:
    _ToolBashRuntime(session, config).install()


class _BashTool(EnvironmentExecutableTool):
    name = "bash"
    execution_requirements = ToolExecutionRequirements(
        filesystem="write",
        network=True,
        interrupt="cancel",
    )
    description = (
        "Execute a shell command in the session cwd. The result reports the "
        "exit code, wall time, stdout/stderr line counts, and the captured "
        "stdout/stderr; a non-zero exit or timeout is flagged as an error."
    )

    def __init__(
        self,
        *,
        session: AtomAPI,
        default_timeout: float,
        parameters: dict[str, object],
        tails: BashOutputTails | None = None,
    ) -> None:
        self.parameters = parameters
        self._session = session
        self._default_timeout = default_timeout
        self._tails = tails

    async def execute(
        self,
        args: dict[str, object],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult:
        return await self._execute_with(
            args,
            bash_ops=_require_bash_ops(self._session),
            cwd=self._session.ctx.cwd,
            signal=signal,
        )

    async def execute_in_environment(
        self,
        args: Mapping[str, object],
        *,
        environment: EnvironmentOperations,
        cwd: str | None = None,
        signal: CancelSignal | None = None,
    ) -> ToolResult:
        resolved_cwd = cwd
        if resolved_cwd is None:
            candidate = environment.ref.metadata.get("cwd")
            match candidate:
                case str() if candidate:
                    resolved_cwd = candidate
                case _:
                    return _error(
                        f"Environment {environment.ref.id!r} does not declare a cwd"
                    )
        return await self._execute_with(
            args,
            bash_ops=environment.bash,
            cwd=resolved_cwd,
            signal=signal,
        )

    async def _execute_with(
        self,
        args: Mapping[str, object],
        *,
        bash_ops: BashOperations,
        cwd: str,
        signal: CancelSignal | None,
    ) -> ToolResult:
        try:
            parsed = _BashArgs.model_validate(dict(args))
        except ValidationError as exc:
            return _error(f"Invalid bash call: {exc}")
        cmd = parsed.cmd
        timeout = parsed.timeout if "timeout" in args else self._default_timeout
        on_data: Callable[[bytes], None] | None = None
        log_path: str | None = None
        if self._tails is not None:
            tail_key = self._tails.current()
            if tail_key is not None:
                on_data = self._tails.open(tail_key)
                log_path = self._tails.log_path(tail_key)
        t0 = time.monotonic()
        try:
            result = await bash_ops.exec(
                cmd,
                cwd=cwd,
                timeout=timeout,
                signal=signal,
                on_data=on_data,
                log_path=log_path,
            )
        except Exception as exc:
            logger.debug("tool_bash: exec failed for {!r}: {}", cmd, exc)
            return _error(f"Failed to run command {cmd!r}: {exc}")
        elapsed_s = time.monotonic() - t0
        duration_ms = max(0, int(round(elapsed_s * 1000)))
        wall_time = round(elapsed_s, 1)

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        is_error = result.exit_code != 0 or result.timed_out

        sections: list[str] = []
        sections.append(f"Exit code: {result.exit_code}")
        sections.append(f"Wall time: {wall_time}s")
        if result.timed_out:
            sections.append(f"TIMED OUT after {timeout}s")

        stdout_lines = stdout.count("\n") + (1 if stdout else 0)
        stderr_lines = stderr.count("\n") + (1 if stderr else 0)
        sections.append(f"Stdout lines: {stdout_lines}")
        if stderr_lines:
            sections.append(f"Stderr lines: {stderr_lines}")

        if stdout:
            sections.append(f"Stdout:\n{stdout}")
        if stderr:
            sections.append(f"Stderr:\n{stderr}")

        text = "\n".join(sections)
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=is_error,
            extras={
                "exit_code": result.exit_code,
                "duration_ms": duration_ms,
                "wall_time_s": wall_time,
                "timed_out": result.timed_out,
                "timeout_s": timeout,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_lines": stdout_lines,
                "stderr_lines": stderr_lines,
                "log_path": log_path,
            },
        )


def _require_bash_ops(session: AtomAPI) -> BashOperations:
    try:
        return session.services.require(
            BASH_OPERATIONS_SERVICE,
            cast(type[BashOperations], BashOperations),
        )
    except (ServiceNotFound, ServiceTypeMismatch) as exc:
        raise RuntimeError(
            "tool_bash requires the operations atom to register bash"
        ) from exc


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
