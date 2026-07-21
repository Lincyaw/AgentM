# code-health: ignore-file[AM025] -- repository outlines are external JSON payloads
"""Live ast-grep repository outline index for realtime IFG validation."""

from __future__ import annotations

import asyncio
import json
import posixpath
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loguru import logger

from agentm.core.abi import BashOperations

from .source_parser import (
    SymbolExtractionInput,
    SymbolExtractionResult,
    extract_symbols_from_repository_outline,
    parse_bash_segments,
)
from .source_semantics import analyze_bash_segment


REPOSITORY_INDEX_SERVICE = "policy:repository_index"
_WRITE_ACTIONS = frozenset({"write", "edit", "create", "delete"})
_GIT_READ_SUBCOMMANDS = frozenset(
    {"check-ignore", "diff", "grep", "log", "ls-files", "show", "status"}
)


@dataclass(frozen=True, slots=True)
class RepositoryRefreshPlan:
    full_scan: bool = False
    paths: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RepositoryIndexStatus:
    root: str
    ready: bool
    files: int
    scans: int
    refreshes: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class _RepositoryScanResult:
    documents: Mapping[str, Mapping[str, object]]
    paths: tuple[str, ...]
    files: int


class RepositoryIndex:
    """Session-local facade over a local or sandbox-resident outline index."""

    def __init__(
        self,
        *,
        root: str,
        bash: BashOperations,
        scan_timeout: float = 120.0,
        refresh_timeout: float = 30.0,
        search_roots: Sequence[str] = (),
        remote_worker_command: str | None = None,
        remote_db_path: str | None = None,
    ) -> None:
        self._root = _normalize_root(root)
        self._search_roots = tuple(
            dict.fromkeys(_normalize_root(path) for path in search_roots if path)
        )
        self._bash = bash
        self._scan_timeout = scan_timeout
        self._refresh_timeout = refresh_timeout
        self._remote_worker_command = remote_worker_command
        self._remote_db_path = remote_db_path
        if self._remote_worker_command and not self._remote_db_path:
            raise ValueError("remote repository worker requires a remote db path")
        self._documents: dict[str, Mapping[str, object]] = {}
        self._paths: set[str] = set()
        self._refresh_lock = asyncio.Lock()
        self._ready = False
        self._scans = 0
        self._refreshes = 0
        self._last_error: str | None = None

    @property
    def root(self) -> str:
        return self._root

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def status(self) -> RepositoryIndexStatus:
        return RepositoryIndexStatus(
            root=self._root,
            ready=self._ready,
            files=len(self._paths),
            scans=self._scans,
            refreshes=self._refreshes,
            last_error=self._last_error,
        )

    async def scan_all(self, *, force: bool = False) -> bool:
        async with self._refresh_lock:
            if self._ready and not force:
                return True
            repository_root, root_error = await self._discover_repository_root()
            if root_error is not None:
                self._scans += 1
                self._last_error = root_error
                return False
            self._root = repository_root
            scan, error = await self._scan_target(
                self._root,
                timeout=self._scan_timeout,
                replace=True,
            )
            self._scans += 1
            if error is not None:
                self._last_error = error
                return False
            if scan is None:
                self._last_error = "repository scan returned no result"
                return False
            self._documents = dict(scan.documents)
            self._paths = set(scan.paths)
            self._ready = True
            self._last_error = None
            return True

    async def _discover_repository_root(self) -> tuple[str, str | None]:
        root = await self._git_repository_root(self._root)
        if root is not None:
            return self._validated_repository_root(root)
        for search_root in self._search_roots:
            candidate = await self._find_repository_under(search_root)
            if candidate is None:
                continue
            root = await self._git_repository_root(candidate)
            if root is not None:
                return self._validated_repository_root(root)
        return self._root, "repository scan skipped: no git repository found"

    async def _git_repository_root(self, cwd: str) -> str | None:
        try:
            result = await self._bash.exec(
                "git rev-parse --show-toplevel",
                cwd=cwd,
                timeout=min(self._refresh_timeout, 10.0),
            )
        except Exception as exc:
            logger.debug("repository git-root discovery failed in {}: {}", cwd, exc)
            return None
        if result.timed_out or result.exit_code != 0:
            return None
        root = result.stdout.decode("utf-8", errors="replace").strip()
        return root if root and posixpath.isabs(root) else None

    async def _find_repository_under(self, search_root: str) -> str | None:
        quoted = shlex.quote(search_root)
        command = (
            f"find {quoted} -mindepth 1 -maxdepth 4 -name .git -print -quit 2>/dev/null"
        )
        try:
            result = await self._bash.exec(
                command,
                cwd=search_root,
                timeout=min(self._refresh_timeout, 10.0),
            )
        except Exception as exc:
            logger.debug("repository search failed under {}: {}", search_root, exc)
            return None
        if result.timed_out or result.exit_code != 0:
            return None
        marker = result.stdout.decode("utf-8", errors="replace").strip()
        if not marker or posixpath.basename(marker) != ".git":
            return None
        return posixpath.dirname(posixpath.normpath(marker))

    def _validated_repository_root(self, root: str) -> tuple[str, str | None]:
        normalized = posixpath.normpath(root)
        if normalized == "/":
            return self._root, "repository scan skipped: refusing filesystem root"
        return normalized, None

    async def refresh(self, plan: RepositoryRefreshPlan) -> bool:
        if plan.full_scan:
            return await self.scan_all(force=True)
        async with self._refresh_lock:
            success = True
            for path in plan.paths:
                scan, error = await self._scan_target(
                    path,
                    timeout=self._refresh_timeout,
                    replace=False,
                )
                self._refreshes += 1
                if error is not None:
                    self._last_error = error
                    success = False
                    continue
                if scan is None:
                    self._last_error = "repository refresh returned no result"
                    success = False
                    continue
                self._remove_scope(path)
                self._paths.update(scan.paths)
                self._documents.update(scan.documents)
            if success:
                self._last_error = None
            return success

    def contains_file(self, path: str) -> bool:
        return self._normalize_path(path) in self._paths

    def canonical_path(self, path: str) -> str:
        """Return the repository index's canonical path for an event path."""

        return self._normalize_path(path)

    def extract_symbols(
        self,
        source_units: Sequence[SymbolExtractionInput],
        *,
        extractor_version: str,
    ) -> SymbolExtractionResult:
        documents: dict[str, Mapping[str, object]] = {}
        for unit in source_units:
            normalized = self._normalize_path(unit.path)
            document = self._documents.get(normalized)
            if document is not None:
                documents[posixpath.normpath(unit.path)] = document
        return extract_symbols_from_repository_outline(
            source_units,
            documents_by_path=documents,
            extractor_version=extractor_version,
        )

    async def _scan_target(
        self,
        target: str,
        *,
        timeout: float,
        replace: bool,
    ) -> tuple[_RepositoryScanResult | None, str | None]:
        normalized = self._normalize_path(target)
        if self._remote_worker_command is not None:
            return await self._scan_target_remote(
                normalized,
                timeout=timeout,
                replace=replace,
            )
        quoted = shlex.quote(normalized)
        command = (
            f"if [ -e {quoted} ]; then "
            f"ast-grep outline {quoted} --json=compact --items all "
            "--view expanded --threads 1; "
            "else printf '[]'; fi"
        )
        try:
            result = await self._bash.exec(command, cwd=self._root, timeout=timeout)
        except Exception as exc:
            return None, f"outline execution failed: {type(exc).__name__}: {exc}"
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if result.timed_out:
            return None, f"outline timed out after {timeout:g}s"
        if result.exit_code != 0:
            detail = stderr[:500] if stderr else f"exit {result.exit_code}"
            return None, f"outline failed: {detail}"
        try:
            payload = json.loads(
                result.stdout.decode("utf-8", errors="replace") or "[]"
            )
        except json.JSONDecodeError as exc:
            return None, f"outline returned invalid JSON: {exc}"
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            return None, "outline returned a non-array payload"
        documents: dict[str, Mapping[str, object]] = {}
        for raw_document in payload:
            if not isinstance(raw_document, Mapping):
                continue
            raw_path = raw_document.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            path = self._normalize_path(raw_path)
            documents[path] = {**dict(raw_document), "path": path}
        return _RepositoryScanResult(
            documents=documents,
            paths=tuple(documents),
            files=len(documents),
        ), None

    async def _scan_target_remote(
        self,
        target: str,
        *,
        timeout: float,
        replace: bool,
    ) -> tuple[_RepositoryScanResult | None, str | None]:
        worker = self._remote_worker_command
        database = self._remote_db_path
        if worker is None or database is None:
            return None, "remote repository worker is not configured"
        flags = " --replace" if replace else " --include-documents"
        command = (
            f"{worker} repository-index"
            f" --root {shlex.quote(self._root)}"
            f" --target {shlex.quote(target)}"
            f" --db {shlex.quote(database)}{flags}"
        )
        try:
            result = await self._bash.exec(command, cwd=self._root, timeout=timeout)
        except Exception as exc:
            return None, f"remote index execution failed: {type(exc).__name__}: {exc}"
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        payload: object = None
        if stdout:
            try:
                payload = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError as exc:
                return None, f"remote index returned invalid JSON: {exc}"
        if result.timed_out:
            return None, f"remote index timed out after {timeout:g}s"
        if not isinstance(payload, Mapping):
            detail = result.stderr.decode("utf-8", errors="replace").strip()[:500]
            return None, detail or f"remote index failed with exit {result.exit_code}"
        if result.exit_code != 0 or payload.get("ok") is not True:
            error = payload.get("error")
            detail = error if isinstance(error, str) else f"exit {result.exit_code}"
            return None, f"remote index failed: {detail}"

        raw_paths = payload.get("paths")
        paths = (
            tuple(
                self._normalize_path(path)
                for path in raw_paths
                if isinstance(path, str) and path
            )
            if isinstance(raw_paths, Sequence)
            and not isinstance(raw_paths, (str, bytes))
            else ()
        )
        raw_documents = payload.get("documents")
        documents: dict[str, Mapping[str, object]] = {}
        if isinstance(raw_documents, Sequence) and not isinstance(
            raw_documents, (str, bytes)
        ):
            for raw_document in raw_documents:
                if not isinstance(raw_document, Mapping):
                    continue
                raw_path = raw_document.get("path")
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                path = self._normalize_path(raw_path)
                documents[path] = {**dict(raw_document), "path": path}
        raw_files = payload.get("files")
        files = raw_files if isinstance(raw_files, int) else len(paths)
        return _RepositoryScanResult(
            documents=documents,
            paths=paths,
            files=files,
        ), None

    def _remove_scope(self, path: str) -> None:
        normalized = self._normalize_path(path)
        prefix = f"{normalized.rstrip('/')}/"
        for indexed_path in tuple(self._documents):
            if indexed_path == normalized or indexed_path.startswith(prefix):
                del self._documents[indexed_path]
        self._paths = {
            indexed_path
            for indexed_path in self._paths
            if indexed_path != normalized and not indexed_path.startswith(prefix)
        }

    def _normalize_path(self, path: str) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(self._root, path))


def repository_refresh_plan(
    *,
    tool_name: str,
    args: Mapping[str, object],
    cwd: str,
) -> RepositoryRefreshPlan | None:
    """Plan the repository refresh caused by one tool call."""

    if tool_name in {"read", "write", "edit"}:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            return None
        return RepositoryRefreshPlan(
            paths=(_normalize_event_path(path, cwd),),
            reason=f"tool:{tool_name}",
        )
    if tool_name != "bash":
        return None
    command = args.get("cmd")
    if not isinstance(command, str) or not command.strip():
        return None
    return _bash_refresh_plan(command, cwd=cwd)


def _bash_refresh_plan(command: str, *, cwd: str) -> RepositoryRefreshPlan | None:
    paths: list[str] = []
    full_scan = False
    reasons: list[str] = []
    current_cwd = _normalize_root(cwd)
    for segment in parse_bash_segments(command):
        if not segment.argv:
            continue
        semantics = analyze_bash_segment(segment)
        command_name = semantics.command
        mutates = semantics.action_kind in _WRITE_ACTIONS
        if command_name == "git":
            subcommand = segment.argv[1] if len(segment.argv) > 1 else ""
            mutates = subcommand not in _GIT_READ_SUBCOMMANDS
            if mutates:
                full_scan = True
                reasons.append(f"bash:git:{subcommand or 'unknown'}")
        elif semantics.confidence == "low":
            mutates = True
            full_scan = True
            reasons.append(f"bash:{command_name}:unknown-effects")
        elif semantics.action_kind == "exec":
            mutates = True

        concrete_refs = 0
        for ref in semantics.path_refs:
            if ref.path_kind == "pattern":
                if mutates:
                    full_scan = True
                    reasons.append(f"bash:{command_name}:pattern")
                continue
            normalized = _normalize_event_path(ref.path, current_cwd)
            if ref.path_kind == "file" or mutates:
                paths.append(normalized)
                concrete_refs += 1
        if mutates and concrete_refs == 0:
            full_scan = True
            reasons.append(f"bash:{command_name}:unscoped")
        current_cwd = _next_shell_cwd(segment.argv, current_cwd)

    unique_paths = tuple(dict.fromkeys(paths))
    if not full_scan and not unique_paths:
        return None
    return RepositoryRefreshPlan(
        full_scan=full_scan,
        paths=() if full_scan else unique_paths,
        reason=",".join(dict.fromkeys(reasons)) or "bash:paths",
    )


def _normalize_root(root: str) -> str:
    return posixpath.normpath(root or ".")


def _normalize_event_path(path: str, cwd: str) -> str:
    clean = path.strip().strip("'\"")
    if posixpath.isabs(clean):
        return posixpath.normpath(clean)
    return posixpath.normpath(posixpath.join(_normalize_root(cwd), clean))


def _next_shell_cwd(argv: Sequence[str], cwd: str) -> str:
    if not argv or argv[0] not in {"cd", "pushd"} or len(argv) < 2:
        return cwd
    target = argv[1]
    if target.startswith("-") or "$" in target or "`" in target:
        return cwd
    return _normalize_event_path(target, cwd)
