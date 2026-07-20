# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""Local filesystem implementation of resource read/write protocols."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import IO, Iterator, cast

import yaml

from agentm.core.abi.resource import (
    PathClass,
    ResourceMutation,
    ResourceMutationOp,
    ResourceRecoveryContext,
    ResourceRef,
    ResourceStore,
    ResourceTransactionRef,
    ResourceTxn,
    ResourceTxnContext,
    TransactionalResourceWriter,
    WriteResult,
    WriterAuthor,
)
from agentm.core.lib.async_cancel import await_known_outcome, settle_known_outcome


_DEFAULT_NAMESPACES = (
    "artifact",
    "sandbox",
    "summary",
    "content",
    "catalog",
    "observability",
    "environment",
)


class _LocalResourceFiles:
    """Path resolution, mutation policy, and locked filesystem operations."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        root: Path,
        namespace_roots: Mapping[str, str | Path] | None,
        manifest_path: Path | None,
    ) -> None:
        self.workspace_root = workspace_root
        self.root = root
        self.namespace_roots: dict[str, Path] = {
            "workspace": workspace_root,
            **{namespace: root / namespace for namespace in _DEFAULT_NAMESPACES},
        }
        if namespace_roots:
            self.namespace_roots.update(
                {namespace: Path(path) for namespace, path in namespace_roots.items()}
            )
        self._lock_path = root / "resource.lock"
        (
            self._constitution_globs,
            self._managed_globs,
        ) = _load_resource_globs(manifest_path)

    def read_ref(self, ref: ResourceRef) -> bytes:
        return self.resolve_ref(ref).read_bytes()

    def exists_ref(self, ref: ResourceRef) -> bool:
        return self.resolve_ref(ref).exists()

    def list_ref(self, ref: ResourceRef) -> list[ResourceRef]:
        path = self.resolve_ref(ref)
        if not path.exists():
            return []
        if not path.is_dir():
            return [ref]
        root = self.namespace_root(ref.namespace)
        return [
            ResourceRef(namespace=ref.namespace, path=str(child.relative_to(root)))
            for child in sorted(path.iterdir())
        ]

    def write_ref(
        self,
        ref: ResourceRef,
        content: bytes,
        rationale: str,
        author: WriterAuthor,
    ) -> ResourceMutation:
        with self.locked():
            self.assert_mutation_allowed(ref)
            path = self.resolve_ref(ref)
            before = _digest_bytes(path.read_bytes()) if path.exists() else None
            _atomic_write_bytes(path, content)
            return _resource_mutation(
                ref,
                "write",
                before,
                _digest_bytes(content),
                rationale,
                author,
            )

    def replace_ref(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        rationale: str,
        author: WriterAuthor,
    ) -> ResourceMutation:
        with self.locked():
            self.assert_mutation_allowed(ref)
            path = self.resolve_ref(ref)
            if not path.exists():
                raise FileNotFoundError(path)
            current = path.read_bytes()
            if current != old:
                raise ValueError("old content does not match current resource")
            _atomic_write_bytes(path, new)
            return _resource_mutation(
                ref,
                "replace",
                _digest_bytes(current),
                _digest_bytes(new),
                rationale,
                author,
            )

    def delete_ref(
        self,
        ref: ResourceRef,
        rationale: str,
        author: WriterAuthor,
    ) -> ResourceMutation:
        with self.locked():
            self.assert_mutation_allowed(ref)
            path = self.resolve_ref(ref)
            before = _digest_bytes(path.read_bytes()) if path.exists() else None
            if path.exists():
                _unlink_and_fsync(path)
            return _resource_mutation(
                ref,
                "delete",
                before,
                None,
                rationale,
                author,
            )

    def read_workspace(self, path: str) -> bytes:
        return self.resolve_workspace_path(path).read_bytes()

    def exists_workspace(self, path: str) -> bool:
        return self.resolve_workspace_path(path).exists()

    def list_dir(self, path: str) -> list[str]:
        resolved = self.resolve_workspace_path(path)
        if not resolved.exists():
            return []
        return [child.name for child in sorted(resolved.iterdir())]

    def write_workspace(self, path: str, content: bytes) -> Path:
        with self.locked():
            resolved = self.resolve_workspace_path(path)
            _atomic_write_bytes(resolved, content)
            return resolved

    def replace_workspace(
        self,
        path: str,
        old: bytes,
        new: bytes,
    ) -> tuple[Path, bool]:
        with self.locked():
            resolved = self.resolve_workspace_path(path)
            if not resolved.exists():
                return resolved, False
            current = resolved.read_bytes()
            if current != old:
                return resolved, False
            _atomic_write_bytes(resolved, new)
            return resolved, True

    def delete_workspace(self, path: str) -> Path:
        with self.locked():
            resolved = self.resolve_workspace_path(path)
            if resolved.exists():
                _unlink_and_fsync(resolved)
            return resolved

    def classify(self, path: str) -> PathClass:
        candidates = self.workspace_path_candidates(path)
        if any(
            _matches_resource_glob(pattern, candidate)
            for candidate in candidates
            for pattern in self._constitution_globs
        ):
            return "constitution"
        if any(
            _matches_resource_glob(pattern, candidate)
            for candidate in candidates
            for pattern in self._managed_globs
        ):
            return "managed"
        return "unmanaged"

    def resolve_ref(self, ref: ResourceRef) -> Path:
        return _resolve_inside(self.namespace_root(ref.namespace), ref.path)

    def resolve_workspace_path(self, path: str) -> Path:
        return _resolve_inside(self.workspace_root, path)

    def workspace_path_candidates(self, path: str) -> tuple[str, ...]:
        resolved = self.resolve_workspace_path(path)
        physical = resolved.relative_to(_real_path(self.workspace_root)).as_posix()
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                lexical = (
                    candidate.absolute()
                    .relative_to(self.workspace_root.absolute())
                    .as_posix()
                )
            except ValueError:
                lexical = physical
        else:
            lexical = candidate.as_posix()
        return tuple(dict.fromkeys((lexical, physical)))

    def assert_mutation_allowed(self, ref: ResourceRef) -> None:
        if ref.namespace == "workspace" and self.classify(ref.path) == "constitution":
            raise PermissionError(
                f"refusing to modify constitution resource {ref.uri()!r}"
            )

    @contextmanager
    def locked(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as handle:
            _lock_file(handle)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def namespace_root(self, namespace: str) -> Path:
        try:
            return self.namespace_roots[namespace]
        except KeyError as exc:
            raise ValueError(f"unknown resource namespace: {namespace}") from exc


class _LocalTransactionJournal:
    """Crash-recoverable prepare/apply/commit records for resource turns."""

    def __init__(self, files: _LocalResourceFiles) -> None:
        self._files = files
        self._transactions_root = files.root / "resource_transactions"

    def recover(self, context: ResourceRecoveryContext) -> None:
        with self._files.locked():
            committed = {
                transaction.id for transaction in context.committed_transactions
            }
            seen: set[str] = set()
            if self._transactions_root.exists():
                for txn_dir in sorted(self._transactions_root.iterdir()):
                    manifest_path = txn_dir / "manifest.json"
                    if not manifest_path.exists():
                        _rmtree_and_fsync(txn_dir)
                        continue
                    manifest = _read_manifest(manifest_path)
                    session_id = _manifest_str(manifest, "session_id")
                    if session_id != context.session_id:
                        continue
                    transaction_id = _manifest_str(manifest, "transaction_id")
                    _validate_manifest_identity(
                        manifest,
                        transaction_id=transaction_id,
                    )
                    seen.add(transaction_id)
                    if transaction_id in committed:
                        self._commit_locked(transaction_id)
                    elif _manifest_status(manifest) != "committed":
                        self._abandon_locked(transaction_id)
            missing = committed - seen
            if missing:
                raise RuntimeError(
                    "resource transaction staging is missing for committed turns: "
                    + ", ".join(sorted(missing))
                )

    def transaction_dir(self, transaction_id: str) -> Path:
        return self._transactions_root / transaction_id.removeprefix("sha256:")

    def commit(self, transaction_id: str) -> None:
        with self._files.locked():
            self._commit_locked(transaction_id)

    def _commit_locked(self, transaction_id: str) -> None:
        txn_dir = self.transaction_dir(transaction_id)
        manifest_path = txn_dir / "manifest.json"
        manifest = _read_manifest(manifest_path)
        _validate_manifest_identity(manifest, transaction_id=transaction_id)
        if _manifest_status(manifest) == "committed":
            return
        if _manifest_status(manifest) == "prepared":
            self._apply_locked(transaction_id)
            manifest = _read_manifest(manifest_path)
            _validate_manifest_identity(manifest, transaction_id=transaction_id)
        if _manifest_status(manifest) != "applied":
            raise RuntimeError(
                f"resource transaction {transaction_id} has invalid status "
                f"{manifest.get('status')!r}"
            )
        manifest["status"] = "committed"
        _write_manifest(manifest_path, manifest)

    def apply(self, transaction_id: str) -> None:
        with self._files.locked():
            self._apply_locked(transaction_id)

    def _apply_locked(self, transaction_id: str) -> None:
        txn_dir = self.transaction_dir(transaction_id)
        manifest_path = txn_dir / "manifest.json"
        manifest = _read_manifest(manifest_path)
        _validate_manifest_identity(manifest, transaction_id=transaction_id)
        status = _manifest_status(manifest)
        if status in {"applied", "committed"}:
            return
        if status != "prepared":
            raise RuntimeError(
                f"resource transaction {transaction_id} has invalid status {status!r}"
            )
        for operation in _manifest_operations(manifest):
            ref = _operation_ref(operation)
            path = self._files.resolve_ref(ref)
            before = _optional_manifest_str(operation, "before_version")
            after = _optional_manifest_str(operation, "after_version")
            current = _path_digest(path)
            if current == after:
                continue
            if current != before:
                raise RuntimeError(
                    f"resource changed during transaction {transaction_id}: "
                    f"{ref.uri()} is {current!r}, expected {before!r}"
                )
            new_file = _optional_manifest_str(operation, "new_file")
            if new_file is None:
                if path.exists():
                    _unlink_and_fsync(path)
            else:
                _atomic_write_bytes(path, (txn_dir / new_file).read_bytes())
        manifest["status"] = "applied"
        _write_manifest(manifest_path, manifest)

    def abandon(self, transaction_id: str) -> None:
        with self._files.locked():
            self._abandon_locked(transaction_id)

    def _abandon_locked(self, transaction_id: str) -> None:
        txn_dir = self.transaction_dir(transaction_id)
        manifest_path = txn_dir / "manifest.json"
        if not manifest_path.exists():
            if txn_dir.exists():
                _rmtree_and_fsync(txn_dir)
            return
        manifest = _read_manifest(manifest_path)
        _validate_manifest_identity(manifest, transaction_id=transaction_id)
        if _manifest_status(manifest) == "committed":
            raise RuntimeError(
                f"cannot abandon committed resource transaction {transaction_id}"
            )
        for operation in reversed(_manifest_operations(manifest)):
            ref = _operation_ref(operation)
            path = self._files.resolve_ref(ref)
            before = _optional_manifest_str(operation, "before_version")
            after = _optional_manifest_str(operation, "after_version")
            current = _path_digest(path)
            if current == before:
                continue
            if current != after:
                raise RuntimeError(
                    f"resource changed while abandoning transaction {transaction_id}: "
                    f"{ref.uri()} is {current!r}, expected {after!r}"
                )
            before_file = _optional_manifest_str(operation, "before_file")
            if before_file is None:
                if path.exists():
                    _unlink_and_fsync(path)
            else:
                _atomic_write_bytes(path, (txn_dir / before_file).read_bytes())
        _rmtree_and_fsync(txn_dir)

    def forget(self, transaction_id: str) -> None:
        with self._files.locked():
            txn_dir = self.transaction_dir(transaction_id)
            manifest = _read_manifest(txn_dir / "manifest.json")
            _validate_manifest_identity(manifest, transaction_id=transaction_id)
            if _manifest_status(manifest) != "committed":
                raise RuntimeError(
                    f"cannot discard uncommitted resource transaction {transaction_id}"
                )
            _rmtree_and_fsync(txn_dir)


class LocalResourceStore(TransactionalResourceWriter, ResourceStore):
    """Map logical ``ResourceRef`` namespaces to local directories."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        root: str | Path | None = None,
        namespace_roots: Mapping[str, str | Path] | None = None,
        manifest_path: str | Path | None = None,
        discover_manifest: bool = True,
    ) -> None:
        workspace_path = Path(workspace_root)
        resource_root = Path(root) if root is not None else workspace_path / ".agentm"
        resolved_manifest = (
            Path(manifest_path)
            if manifest_path is not None
            else workspace_path / "core-manifest.yaml"
            if discover_manifest and (workspace_path / "core-manifest.yaml").is_file()
            else None
        )
        self._files = _LocalResourceFiles(
            workspace_root=workspace_path,
            root=resource_root,
            namespace_roots=namespace_roots,
            manifest_path=resolved_manifest,
        )
        self._journal = _LocalTransactionJournal(self._files)

    async def read_ref(self, ref: ResourceRef) -> bytes:
        return await asyncio.to_thread(self._files.read_ref, ref)

    async def exists_ref(self, ref: ResourceRef) -> bool:
        return await asyncio.to_thread(self._files.exists_ref, ref)

    async def list_ref(self, ref: ResourceRef) -> list[ResourceRef]:
        return await asyncio.to_thread(self._files.list_ref, ref)

    async def write_ref(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        return await await_known_outcome(
            asyncio.to_thread(
                self._files.write_ref,
                ref,
                content,
                rationale,
                author,
            )
        )

    async def replace_ref(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        return await await_known_outcome(
            asyncio.to_thread(
                self._files.replace_ref,
                ref,
                old,
                new,
                rationale,
                author,
            )
        )

    async def delete_ref(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        return await await_known_outcome(
            asyncio.to_thread(
                self._files.delete_ref,
                ref,
                rationale,
                author,
            )
        )

    async def read(self, path: str) -> bytes:
        return await asyncio.to_thread(self._files.read_workspace, path)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(self._files.exists_workspace, path)

    async def list_dir(self, path: str) -> list[str]:
        return await asyncio.to_thread(self._files.list_dir, path)

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        path_class = self.classify(path)
        if path_class == "constitution":
            return _constitution_write_error(path)
        resolved = await await_known_outcome(
            asyncio.to_thread(
                self._files.write_workspace,
                path,
                content,
            )
        )
        return WriteResult(path=str(resolved), path_class=self.classify(path))

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        path_class = self.classify(path)
        if path_class == "constitution":
            return _constitution_write_error(path)
        resolved, replaced = await await_known_outcome(
            asyncio.to_thread(
                self._files.replace_workspace,
                path,
                old,
                new,
            )
        )
        if not replaced:
            return WriteResult(
                path=str(resolved),
                path_class=self.classify(path),
                error="old content does not match current resource",
            )
        return WriteResult(path=str(resolved), path_class=self.classify(path))

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        path_class = self.classify(path)
        if path_class == "constitution":
            return _constitution_write_error(path)
        resolved = await await_known_outcome(
            asyncio.to_thread(self._files.delete_workspace, path)
        )
        return WriteResult(path=str(resolved), path_class=self.classify(path))

    def classify(self, path: str) -> PathClass:
        return self._files.classify(path)

    async def begin_txn(self, context: ResourceTxnContext) -> ResourceTxn:
        return _LocalResourceTxn(self, context)

    async def recover(self, context: ResourceRecoveryContext) -> None:
        await await_known_outcome(asyncio.to_thread(self._journal.recover, context))

    async def fork_for_environment(
        self,
        *,
        workspace_root: str,
        child_session_id: str,
    ) -> "LocalResourceStore":
        """Rebind workspace paths while sharing non-workspace namespaces."""

        del child_session_id
        namespace_roots = {
            namespace: root
            for namespace, root in self._files.namespace_roots.items()
            if namespace != "workspace"
        }
        return LocalResourceStore(
            workspace_root=workspace_root,
            root=self._files.root,
            namespace_roots=namespace_roots,
            discover_manifest=True,
        )


@dataclass(slots=True)
class _PendingMutation:
    op: ResourceMutationOp
    ref: ResourceRef
    old: bytes | None = None
    new: bytes | None = None
    rationale: str = ""
    author: WriterAuthor = "agent"


class _LocalResourceTxn(ResourceTxn):
    def __init__(self, store: LocalResourceStore, context: ResourceTxnContext) -> None:
        self._store = store
        self._context = context
        self._pending: list[_PendingMutation] = []
        self._transaction_id = _transaction_id(context)
        self._prepared: tuple[ResourceMutation, ...] | None = None
        self._applied = False
        self._closed = False

    async def read(self, ref: ResourceRef) -> bytes | None:
        self._ensure_open()
        for pending in reversed(self._pending):
            if pending.ref != ref:
                continue
            return None if pending.op == "delete" else pending.new
        return await asyncio.to_thread(
            self._read_base,
            ref,
        )

    def _read_base(self, ref: ResourceRef) -> bytes | None:
        path = self._store._files.resolve_ref(ref)
        return path.read_bytes() if path.exists() else None

    async def create(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        self._ensure_open()
        self._store._files.assert_mutation_allowed(ref)
        self._pending.append(
            _PendingMutation(
                "create",
                ref,
                new=bytes(content),
                rationale=rationale,
                author=author,
            )
        )
        return self._mutation(
            ref,
            "create",
            None,
            _digest_bytes(content),
            rationale,
            author,
        )

    async def replace(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        self._ensure_open()
        self._store._files.assert_mutation_allowed(ref)
        self._pending.append(
            _PendingMutation(
                "replace",
                ref,
                old=bytes(old),
                new=bytes(new),
                rationale=rationale,
                author=author,
            )
        )
        return self._mutation(
            ref, "replace", _digest_bytes(old), _digest_bytes(new), rationale, author
        )

    async def delete(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        self._ensure_open()
        self._store._files.assert_mutation_allowed(ref)
        before = await asyncio.to_thread(
            _path_digest,
            self._store._files.resolve_ref(ref),
        )
        self._pending.append(
            _PendingMutation("delete", ref, rationale=rationale, author=author)
        )
        return self._mutation(ref, "delete", before, None, rationale, author)

    async def prepare(self) -> tuple[ResourceMutation, ...]:
        self._ensure_open()
        if self._prepared is None:
            prepared, caller_cancelled = await settle_known_outcome(
                asyncio.to_thread(self._prepare)
            )
            self._prepared = prepared
            if caller_cancelled:
                raise asyncio.CancelledError
        return self._prepared

    async def commit(self) -> None:
        self._ensure_open()
        if self._prepared is None:
            raise RuntimeError("resource transaction must be prepared before commit")
        if not self._applied:
            raise RuntimeError("resource transaction must be applied before commit")
        _, caller_cancelled = await settle_known_outcome(
            asyncio.to_thread(
                self._store._journal.commit,
                self._transaction_id,
            )
        )
        self._closed = True
        if caller_cancelled:
            raise asyncio.CancelledError

    async def apply(self) -> None:
        self._ensure_open()
        if self._prepared is None:
            raise RuntimeError("resource transaction must be prepared before apply")
        _, caller_cancelled = await settle_known_outcome(
            asyncio.to_thread(
                self._store._journal.apply,
                self._transaction_id,
            )
        )
        self._applied = True
        if caller_cancelled:
            raise asyncio.CancelledError

    async def abandon(self) -> None:
        if self._closed:
            return
        caller_cancelled = False
        if self._prepared is not None:
            _, caller_cancelled = await settle_known_outcome(
                asyncio.to_thread(
                    self._store._journal.abandon,
                    self._transaction_id,
                )
            )
        self._closed = True
        self._pending.clear()
        if caller_cancelled:
            raise asyncio.CancelledError

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("resource transaction is closed")

    def _mutation(
        self,
        ref: ResourceRef,
        op: ResourceMutationOp,
        before: str | None,
        after: str | None,
        rationale: str,
        author: WriterAuthor,
    ) -> ResourceMutation:
        return ResourceMutation(
            ref=ref,
            op=op,
            transaction=ResourceTransactionRef(
                id=self._transaction_id,
                session_id=self._context.session_id,
                turn_id=self._context.turn_id,
                turn_index=self._context.turn_index,
            ),
            before_version=before,
            after_version=after,
            metadata={
                "rationale": rationale,
                "author": author,
            },
        )

    def _prepare(self) -> tuple[ResourceMutation, ...]:
        with self._store._files.locked():
            return self._prepare_locked()

    def _prepare_locked(self) -> tuple[ResourceMutation, ...]:
        txn_dir = self._store._journal.transaction_dir(self._transaction_id)
        manifest_path = txn_dir / "manifest.json"
        if manifest_path.exists():
            manifest = _read_manifest(manifest_path)
            _validate_manifest_identity(
                manifest,
                transaction_id=self._transaction_id,
                context=self._context,
            )
            return tuple(_manifest_mutations(manifest))

        txn_dir.mkdir(parents=True, exist_ok=False)
        virtual: dict[ResourceRef, bytes | None] = {}
        mutations: list[ResourceMutation] = []
        operations: list[dict[str, object]] = []
        try:
            for index, pending in enumerate(self._pending):
                path = self._store._files.resolve_ref(pending.ref)
                current = (
                    virtual[pending.ref]
                    if pending.ref in virtual
                    else path.read_bytes()
                    if path.exists()
                    else None
                )
                if pending.op == "create" and current is not None:
                    raise FileExistsError(pending.ref.uri())
                if pending.op == "replace" and (
                    current is None or current != pending.old
                ):
                    raise ValueError("old content does not match current resource")
                new_content = None if pending.op == "delete" else pending.new or b""
                before = _digest_bytes(current) if current is not None else None
                after = _digest_bytes(new_content) if new_content is not None else None
                before_file = None
                if current is not None:
                    before_file = f"{index}.before"
                    _atomic_write_bytes(txn_dir / before_file, current)
                new_file = None
                if new_content is not None:
                    new_file = f"{index}.new"
                    _atomic_write_bytes(txn_dir / new_file, new_content)
                mutation = self._mutation(
                    pending.ref,
                    pending.op,
                    before,
                    after,
                    pending.rationale,
                    pending.author,
                )
                mutations.append(mutation)
                operations.append(
                    {
                        "ref": {
                            "namespace": pending.ref.namespace,
                            "path": pending.ref.path,
                        },
                        "op": pending.op,
                        "before_version": before,
                        "after_version": after,
                        "before_file": before_file,
                        "new_file": new_file,
                        "metadata": dict(mutation.metadata),
                    }
                )
                virtual[pending.ref] = new_content
            _write_manifest(
                manifest_path,
                {
                    "transaction_id": self._transaction_id,
                    "session_id": self._context.session_id,
                    "turn_id": self._context.turn_id,
                    "turn_index": self._context.turn_index,
                    "status": "prepared",
                    "operations": operations,
                },
            )
            return tuple(mutations)
        except BaseException:
            if txn_dir.exists():
                _rmtree_and_fsync(txn_dir)
            raise


def _resolve_inside(root: Path, path: str) -> Path:
    if Path(path).is_absolute():
        candidate = _real_path(Path(path))
    else:
        candidate = _real_path(root / path)
    real_root = _real_path(root)
    if candidate != real_root and real_root not in candidate.parents:
        raise ValueError(f"path escapes resource root: {path}")
    return candidate


def _constitution_write_error(path: str) -> WriteResult:
    return WriteResult(
        path=path,
        path_class="constitution",
        error=f"refusing to modify constitution path {path!r}",
    )


def _lock_file(handle: IO[bytes]) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        raise RuntimeError("resource store file locking failed") from exc


def _load_resource_globs(
    manifest_path: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if manifest_path is None:
        return (), (".agentm/**",)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"resource manifest must be an object: {manifest_path}")
    constitution = payload.get("constitution", {})
    managed = payload.get("managed", {})
    if not isinstance(constitution, Mapping) or not isinstance(managed, Mapping):
        raise ValueError(f"resource manifest sections must be objects: {manifest_path}")
    return (
        _string_sequence(
            constitution.get("paths", ()),
            field="constitution.paths",
            path=manifest_path,
        ),
        (
            ".agentm/**",
            *_string_sequence(
                managed.get("globs", ()),
                field="managed.globs",
                path=manifest_path,
            ),
        ),
    )


def _string_sequence(
    value: object,
    *,
    field: str,
    path: Path,
) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{field} must be a list of strings: {path}")
    return tuple(value)


def _matches_resource_glob(pattern: str, path: str) -> bool:
    return _compile_resource_glob(pattern).fullmatch(path) is not None


@cache
def _compile_resource_glob(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                if index + 2 < len(pattern) and pattern[index + 2] == "/":
                    parts.append("(?:.*/)?")
                    index += 3
                else:
                    parts.append(".*")
                    index += 2
            else:
                parts.append("[^/]*")
                index += 1
        elif char == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(char))
            index += 1
    return re.compile("".join(parts))


def _real_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _unlink_and_fsync(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def _rmtree_and_fsync(path: Path) -> None:
    parent = path.parent
    shutil.rmtree(path)
    _fsync_directory(parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _resource_mutation(
    ref: ResourceRef,
    op: ResourceMutationOp,
    before: str | None,
    after: str | None,
    rationale: str,
    author: WriterAuthor,
) -> ResourceMutation:
    return ResourceMutation(
        ref=ref,
        op=op,
        before_version=before,
        after_version=after,
        metadata={"rationale": rationale, "author": author},
    )


def _transaction_id(context: ResourceTxnContext) -> str:
    payload = (f"{context.session_id}\0{context.turn_id}\0{context.turn_index}").encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _path_digest(path: Path) -> str | None:
    return _digest_bytes(path.read_bytes()) if path.exists() else None


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    _atomic_write_bytes(
        path,
        (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"resource transaction manifest is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"resource transaction manifest must be an object: {path}")
    return value


def _manifest_str(manifest: Mapping[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"resource transaction manifest has invalid {key!r}")
    return value


def _optional_manifest_str(
    manifest: Mapping[str, object],
    key: str,
) -> str | None:
    value = manifest.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"resource transaction manifest has invalid {key!r}")
    return value


def _manifest_operations(
    manifest: Mapping[str, object],
) -> list[Mapping[str, object]]:
    operations = manifest.get("operations")
    if not isinstance(operations, list):
        raise ValueError("resource transaction manifest has no operations list")
    if not all(isinstance(operation, Mapping) for operation in operations):
        raise ValueError("resource transaction manifest contains an invalid operation")
    return list(operations)


def _manifest_status(manifest: Mapping[str, object]) -> str:
    status = manifest.get("status")
    if status not in {"prepared", "applied", "committed"}:
        raise ValueError(f"resource transaction manifest has invalid status {status!r}")
    return cast(str, status)


def _manifest_turn_index(manifest: Mapping[str, object]) -> int:
    value = manifest.get("turn_index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("resource transaction manifest has invalid 'turn_index'")
    return value


def _validate_manifest_identity(
    manifest: Mapping[str, object],
    *,
    transaction_id: str,
    context: ResourceTxnContext | None = None,
) -> None:
    actual_transaction_id = _manifest_str(manifest, "transaction_id")
    if actual_transaction_id != transaction_id:
        raise ValueError(
            "resource transaction manifest identity does not match its staging path"
        )
    _manifest_status(manifest)
    session_id = _manifest_str(manifest, "session_id")
    turn_id = _manifest_str(manifest, "turn_id")
    turn_index = _manifest_turn_index(manifest)
    _manifest_operations(manifest)
    if context is None:
        return
    if (
        session_id != context.session_id
        or turn_id != context.turn_id
        or turn_index != context.turn_index
    ):
        raise ValueError("resource transaction staging belongs to a different turn")


def _operation_ref(operation: Mapping[str, object]) -> ResourceRef:
    raw_ref = operation.get("ref")
    if not isinstance(raw_ref, Mapping):
        raise ValueError("resource transaction operation has no ref")
    namespace = raw_ref.get("namespace")
    path = raw_ref.get("path")
    if not isinstance(namespace, str) or not isinstance(path, str):
        raise ValueError("resource transaction operation has an invalid ref")
    return ResourceRef(namespace=namespace, path=path)


def _operation_type(operation: Mapping[str, object]) -> ResourceMutationOp:
    op = operation.get("op")
    if op not in {"create", "write", "replace", "delete"}:
        raise ValueError("resource transaction operation has an invalid op")
    return cast(ResourceMutationOp, op)


def _manifest_mutations(
    manifest: Mapping[str, object],
) -> list[ResourceMutation]:
    transaction_id = _manifest_str(manifest, "transaction_id")
    transaction = ResourceTransactionRef(
        id=transaction_id,
        session_id=_manifest_str(manifest, "session_id"),
        turn_id=_manifest_str(manifest, "turn_id"),
        turn_index=_manifest_turn_index(manifest),
    )
    mutations: list[ResourceMutation] = []
    for operation in _manifest_operations(manifest):
        raw_metadata = operation.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("resource transaction operation has invalid metadata")
        metadata: dict[str, str | int | float | bool | None] = {}
        for key, value in raw_metadata.items():
            if not isinstance(key, str) or not (
                value is None or isinstance(value, (str, int, float, bool))
            ):
                raise ValueError("resource transaction operation has invalid metadata")
            metadata[key] = value
        mutations.append(
            ResourceMutation(
                ref=_operation_ref(operation),
                op=_operation_type(operation),
                transaction=transaction,
                before_version=_optional_manifest_str(
                    operation,
                    "before_version",
                ),
                after_version=_optional_manifest_str(
                    operation,
                    "after_version",
                ),
                metadata=metadata,
            )
        )
    return mutations


__all__ = ["LocalResourceStore"]
