"""Realtime repository index behavior for policy IFG validation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import json

import pytest

from agentm.core.abi import (
    BASH_OPERATIONS_SERVICE,
    BeforeRunEvent,
    BusPriority,
    ServiceRegistry,
    SessionContext,
    SessionReadyEvent,
    SessionShutdownEvent,
    TextContent,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)
from agentm.core.abi.operations import ExecResult
from policy_engine import PolicyEngineConfig, _PolicyEngineRuntime
from policy_engine.ifg import IfgToolEvent, extract_ifg_from_tool_events
from policy_engine.repository_index import (
    REPOSITORY_INDEX_SERVICE,
    RepositoryIndex,
    RepositoryRefreshPlan,
    repository_refresh_plan,
)
from policy_engine.source_parser import SymbolExtractionInput


class _FakeBash:
    def __init__(self, payloads: list[object]) -> None:
        self._payloads = list(payloads)
        self.commands: list[str] = []

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
        del cwd, timeout, env, stdin, on_data, signal, log_path
        self.commands.append(cmd)
        payload = self._payloads.pop(0)
        if isinstance(payload, ExecResult):
            return payload
        stdout = (
            payload.encode()
            if isinstance(payload, str)
            else json.dumps(payload).encode()
        )
        return ExecResult(
            stdout=stdout,
            stderr=b"",
            exit_code=0,
            timed_out=False,
        )


class _FakeAPI:
    def __init__(self, *, root: str, bash: _FakeBash) -> None:
        self.ctx = SessionContext(
            session_id="session-1",
            root_session_id="session-1",
            cwd=root,
        )
        self.services = ServiceRegistry()
        self.services.register(BASH_OPERATIONS_SERVICE, bash)
        self.handlers: dict[str, list[tuple[int, Callable[[object], object]]]] = {}
        self.tools: list[object] = []

    def on(
        self,
        channel: str,
        handler: Callable[[object], object],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]:
        entry = (priority, handler)
        self.handlers.setdefault(channel, []).append(entry)

        def unsubscribe() -> None:
            self.handlers[channel].remove(entry)

        return unsubscribe

    def register_tool(self, tool: object) -> None:
        self.tools.append(tool)

    @contextmanager
    def track_background(self) -> Iterator[None]:
        yield


def _document(path: str, symbol: str) -> Mapping[str, object]:
    return {
        "path": path,
        "language": "Python",
        "items": [
            {
                "role": "item",
                "symbolType": "function",
                "name": symbol,
                "range": {
                    "start": {"line": 0, "column": 0},
                    "end": {"line": 1, "column": 0},
                },
                "signature": f"def {symbol}():",
                "astKind": "function_definition",
                "isExported": True,
            }
        ],
    }


def _symbol_input(path: str) -> SymbolExtractionInput:
    return SymbolExtractionInput(
        session_id="session-1",
        extractor_version="test",
        action_id="action-1",
        source_unit_id="unit-1",
        path=path,
        relation="read",
        turn=0,
        event_id=None,
        tool_name="bash",
        content_hash=None,
        unit_hash=None,
        content_text=None,
        metadata={"cwd": "/remote/repo", "content_scope": "full_file"},
        raw_evidence={},
    )


@pytest.mark.asyncio
async def test_repository_index_scans_and_incrementally_replaces_symbols() -> None:
    path = "/remote/repo/src/app.py"
    bash = _FakeBash(
        ["/remote/repo\n", [_document(path, "foo")], [_document(path, "bar")]]
    )
    index = RepositoryIndex(root="/remote/repo", bash=bash)

    assert await index.scan_all()
    assert index.contains_file(path)
    assert [
        fact.qualified_name
        for fact in index.extract_symbols(
            (_symbol_input(path),), extractor_version="test"
        ).symbols
    ] == ["foo"]
    assert [
        fact.qualified_name
        for fact in index.extract_symbols(
            (_symbol_input("src/app.py"),), extractor_version="test"
        ).symbols
    ] == ["foo"]

    assert await index.refresh(RepositoryRefreshPlan(paths=(path,), reason="write"))
    assert [
        fact.qualified_name
        for fact in index.extract_symbols(
            (_symbol_input(path),), extractor_version="test"
        ).symbols
    ] == ["bar"]
    assert index.status.scans == 1
    assert index.status.refreshes == 1


@pytest.mark.asyncio
async def test_repository_index_keeps_full_outline_in_remote_worker() -> None:
    path = "/remote/repo/src/app.py"
    document = _document(path, "foo")
    bash = _FakeBash(
        [
            "/remote/repo\n",
            {
                "version": 1,
                "ok": True,
                "root": "/remote/repo",
                "files": 1,
                "paths": [path],
                "documents": [],
            },
            {
                "version": 1,
                "ok": True,
                "root": "/remote/repo",
                "files": 1,
                "paths": [path],
                "documents": [document],
            },
        ]
    )
    index = RepositoryIndex(
        root="/remote/repo",
        bash=bash,
        remote_worker_command=(
            "PYTHONPATH=/opt/agentm-toolbox python3 -m agentm_toolbox"
        ),
        remote_db_path="/logs/artifacts/agentm/repository-index.sqlite",
    )

    assert await index.scan_all()
    assert index.contains_file(path)
    assert not index.extract_symbols(
        (_symbol_input(path),), extractor_version="test"
    ).symbols

    assert await index.refresh(RepositoryRefreshPlan(paths=(path,), reason="read"))
    symbols = index.extract_symbols(
        (_symbol_input(path),), extractor_version="test"
    ).symbols

    assert [symbol.qualified_name for symbol in symbols] == ["foo"]
    assert "repository-index" in bash.commands[1]
    assert "--replace" in bash.commands[1]
    assert "--include-documents" in bash.commands[2]


@pytest.mark.asyncio
async def test_ifg_uses_remote_repository_index_for_bash_file_and_symbol() -> None:
    path = "/remote/repo/src/app.py"
    index = RepositoryIndex(
        root="/remote/repo",
        bash=_FakeBash(["/remote/repo\n", [_document(path, "foo")]]),
    )
    assert await index.scan_all()
    event = IfgToolEvent(
        session_id="session-1",
        turn=0,
        event_id=None,
        tool_call_id="bash-1",
        phase="post",
        tool_name="bash",
        args={"cmd": "rg foo src/app.py"},
        result={},
        processed={},
        state={},
        cwd="/remote/repo",
        ts=0.0,
        source="test",
        raw_evidence={},
    )

    rows = extract_ifg_from_tool_events((event,), repository_index=index)

    assert len(rows.file_edges) == 1
    assert rows.file_edges[0].metadata["resolution"] == "repository"
    assert any(symbol.qualified_name == "foo" for symbol in rows.symbols)
    assert any(
        edge.metadata.get("resolution") != "unresolved"
        for edge in rows.action_symbol_edges
    )


def test_repository_refresh_plan_is_targeted_until_bash_scope_is_unknown() -> None:
    read_plan = repository_refresh_plan(
        tool_name="read",
        args={"path": "src/app.py"},
        cwd="/repo",
    )
    search_plan = repository_refresh_plan(
        tool_name="bash",
        args={"cmd": "rg foo src/app.py"},
        cwd="/repo",
    )
    git_plan = repository_refresh_plan(
        tool_name="bash",
        args={"cmd": "git checkout main"},
        cwd="/repo",
    )

    assert read_plan == RepositoryRefreshPlan(
        paths=("/repo/src/app.py",),
        reason="tool:read",
    )
    assert search_plan is not None
    assert search_plan.paths == ("/repo/src/app.py",)
    assert not search_plan.full_scan
    assert git_plan is not None and git_plan.full_scan


@pytest.mark.asyncio
async def test_repository_index_refuses_filesystem_root_scan() -> None:
    bash = _FakeBash(["/\n"])
    index = RepositoryIndex(root="/", bash=bash)

    assert not await index.scan_all()
    assert (
        index.status.last_error == "repository scan skipped: refusing filesystem root"
    )
    assert bash.commands == ["git rev-parse --show-toplevel"]


@pytest.mark.asyncio
async def test_repository_index_discovers_harbor_worktree_under_search_root() -> None:
    path = "/repo/better-auth/src/app.py"
    failed_git = ExecResult(
        stdout=b"",
        stderr=b"not a repository",
        exit_code=128,
        timed_out=False,
    )
    bash = _FakeBash(
        [
            failed_git,
            "/repo/better-auth/.git\n",
            "/repo/better-auth\n",
            [_document(path, "foo")],
        ]
    )
    index = RepositoryIndex(root="/", bash=bash, search_roots=("/repo",))

    assert await index.scan_all()
    assert index.status.root == "/repo/better-auth"
    assert index.contains_file(path)
    assert any(command.startswith("find /repo ") for command in bash.commands)


@pytest.mark.asyncio
async def test_repository_index_reports_missing_outline_command() -> None:
    missing = ExecResult(
        stdout=b"",
        stderr=b"bash: ast-grep: command not found",
        exit_code=127,
        timed_out=False,
    )
    bash = _FakeBash(["/remote/repo\n", missing])
    index = RepositoryIndex(root="/remote/repo", bash=bash)

    assert not await index.scan_all()
    assert index.status.last_error == (
        "outline failed: bash: ast-grep: command not found"
    )
    assert len(bash.commands) == 2


@pytest.mark.asyncio
async def test_policy_atom_mounts_scan_and_refresh_handlers(tmp_path) -> None:
    root = "/remote/repo"
    path = f"{root}/src/app.py"
    bash = _FakeBash(
        ["/remote/repo\n", [_document(path, "foo")], [_document(path, "bar")]]
    )
    api = _FakeAPI(root=root, bash=bash)
    runtime = _PolicyEngineRuntime(
        api,  # type: ignore[arg-type]
        PolicyEngineConfig(
            db_path=str(tmp_path / "policy.db"),
            ifg_repository_scan=True,
        ),
    )
    runtime.install()

    runtime._on_repository_session_ready(SessionReadyEvent(session_id="session-1"))
    await runtime._on_before_run_repository(BeforeRunEvent())
    index = api.services.get(REPOSITORY_INDEX_SERVICE)
    assert isinstance(index, RepositoryIndex)
    assert index.status.ready

    runtime._on_repository_tool_call(
        ToolCallEvent(
            tool_call_id="read-1",
            tool_name="read",
            args={"path": "src/app.py"},
        )
    )
    await runtime._on_repository_tool_result(
        ToolResultEvent(
            tool_call_id="read-1",
            tool_name="read",
            args={"path": "src/app.py"},
            result=ToolResult(content=[TextContent(type="text", text="ok")]),
        )
    )

    assert index.status.refreshes == 1
    assert (
        index.extract_symbols((_symbol_input(path),), extractor_version="test")
        .symbols[0]
        .qualified_name
        == "bar"
    )
    await runtime._on_repository_shutdown(SessionShutdownEvent())
    runtime._on_session_shutdown(SessionShutdownEvent())
