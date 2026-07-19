"""Local filesystem implementation of resource read/write protocols."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from agentm.core.abi.resource import (
    PathClass,
    ResourceMutation,
    ResourceRef,
    ResourceStore,
    ResourceTxn,
    ResourceTxnContext,
    TransactionalResourceWriter,
    WriteResult,
    WriterAuthor,
)


_DEFAULT_NAMESPACES = (
    "artifact",
    "sandbox",
    "summary",
    "content",
    "catalog",
    "observability",
    "environment",
)


class LocalResourceStore(TransactionalResourceWriter, ResourceStore):
    """Map logical ``ResourceRef`` namespaces to local directories."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        root: str | Path | None = None,
        namespace_roots: Mapping[str, str | Path] | None = None,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._root = Path(root) if root is not None else self._workspace_root / ".agentm"
        self._namespace_roots: dict[str, Path] = {
            "workspace": self._workspace_root,
            **{namespace: self._root / namespace for namespace in _DEFAULT_NAMESPACES},
        }
        if namespace_roots:
            self._namespace_roots.update(
                {namespace: Path(path) for namespace, path in namespace_roots.items()}
            )

    async def read_ref(self, ref: ResourceRef) -> bytes:
        return self._resolve_ref(ref).read_bytes()

    async def exists_ref(self, ref: ResourceRef) -> bool:
        return self._resolve_ref(ref).exists()

    async def list_ref(self, ref: ResourceRef) -> list[ResourceRef]:
        path = self._resolve_ref(ref)
        if not path.exists():
            return []
        if not path.is_dir():
            return [ref]
        root = self._namespace_root(ref.namespace)
        return [
            ResourceRef(namespace=ref.namespace, path=str(child.relative_to(root)))
            for child in sorted(path.iterdir())
        ]

    async def write_ref(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        path = self._resolve_ref(ref)
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

    async def replace_ref(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        path = self._resolve_ref(ref)
        current = path.read_bytes() if path.exists() else b""
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

    async def delete_ref(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        path = self._resolve_ref(ref)
        before = _digest_bytes(path.read_bytes()) if path.exists() else None
        if path.exists():
            path.unlink()
        return _resource_mutation(
            ref,
            "delete",
            before,
            None,
            rationale,
            author,
        )

    async def read(self, path: str) -> bytes:
        return self._resolve_workspace_path(path).read_bytes()

    async def exists(self, path: str) -> bool:
        return self._resolve_workspace_path(path).exists()

    async def list_dir(self, path: str) -> list[str]:
        resolved = self._resolve_workspace_path(path)
        if not resolved.exists():
            return []
        return [child.name for child in sorted(resolved.iterdir())]

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        resolved = self._resolve_workspace_path(path)
        _atomic_write_bytes(resolved, content)
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
        resolved = self._resolve_workspace_path(path)
        current = resolved.read_bytes() if resolved.exists() else b""
        if current != old:
            return WriteResult(
                path=str(resolved),
                path_class=self.classify(path),
                error="old content does not match current resource",
            )
        _atomic_write_bytes(resolved, new)
        return WriteResult(path=str(resolved), path_class=self.classify(path))

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        resolved = self._resolve_workspace_path(path)
        if resolved.exists():
            resolved.unlink()
        return WriteResult(path=str(resolved), path_class=self.classify(path))

    def classify(self, path: str) -> PathClass:
        relative = self._workspace_relative(path)
        if relative.parts and relative.parts[0] in {".claude", "core-manifest.yaml"}:
            return "constitution"
        if relative.parts and relative.parts[0] == ".agentm":
            return "managed"
        return "unmanaged"

    @asynccontextmanager
    async def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> AsyncIterator["_LocalBatchHandle"]:
        del rationale, author
        yield _LocalBatchHandle(self)

    async def begin_txn(self, context: ResourceTxnContext) -> ResourceTxn:
        return _LocalResourceTxn(self, context)

    def _resolve_ref(self, ref: ResourceRef) -> Path:
        root = self._namespace_root(ref.namespace)
        return _resolve_inside(root, ref.path)

    def _resolve_workspace_path(self, path: str) -> Path:
        return _resolve_inside(self._workspace_root, path)

    def _workspace_relative(self, path: str) -> Path:
        resolved = self._resolve_workspace_path(path)
        return resolved.relative_to(_real_path(self._workspace_root))

    def _namespace_root(self, namespace: str) -> Path:
        try:
            return self._namespace_roots[namespace]
        except KeyError as exc:
            raise ValueError(f"unknown resource namespace: {namespace}") from exc


class _LocalBatchHandle:
    def __init__(self, store: LocalResourceStore) -> None:
        self._store = store

    async def write(self, path: str, content: bytes) -> None:
        _atomic_write_bytes(self._store._resolve_workspace_path(path), content)

    async def replace(self, path: str, old: bytes, new: bytes) -> None:
        resolved = self._store._resolve_workspace_path(path)
        current = resolved.read_bytes() if resolved.exists() else b""
        if current != old:
            raise ValueError("old content does not match current resource")
        _atomic_write_bytes(resolved, new)

    async def delete(self, path: str) -> None:
        resolved = self._store._resolve_workspace_path(path)
        if resolved.exists():
            resolved.unlink()


@dataclass(slots=True)
class _PendingMutation:
    op: str
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
        self._closed = False

    async def write(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        self._ensure_open()
        self._pending.append(
            _PendingMutation("write", ref, new=bytes(content), rationale=rationale, author=author)
        )
        return self._mutation(ref, "write", None, _digest_bytes(content), rationale, author)

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
        return self._mutation(ref, "replace", _digest_bytes(old), _digest_bytes(new), rationale, author)

    async def delete(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        self._ensure_open()
        path = self._store._resolve_ref(ref)
        before = _digest_bytes(path.read_bytes()) if path.exists() else None
        self._pending.append(
            _PendingMutation("delete", ref, rationale=rationale, author=author)
        )
        return self._mutation(ref, "delete", before, None, rationale, author)

    async def commit(self) -> list[ResourceMutation]:
        self._ensure_open()
        mutations: list[ResourceMutation] = []
        try:
            for pending in self._pending:
                path = self._store._resolve_ref(pending.ref)
                if pending.op == "write":
                    content = pending.new or b""
                    before = _digest_bytes(path.read_bytes()) if path.exists() else None
                    _atomic_write_bytes(path, content)
                    mutations.append(
                        self._mutation(
                            pending.ref,
                            "write",
                            before,
                            _digest_bytes(content),
                            pending.rationale,
                            pending.author,
                        )
                    )
                elif pending.op == "replace":
                    current = path.read_bytes() if path.exists() else b""
                    if current != (pending.old or b""):
                        raise ValueError("old content does not match current resource")
                    content = pending.new or b""
                    _atomic_write_bytes(path, content)
                    mutations.append(
                        self._mutation(
                            pending.ref,
                            "replace",
                            _digest_bytes(current),
                            _digest_bytes(content),
                            pending.rationale,
                            pending.author,
                        )
                    )
                elif pending.op == "delete":
                    before = _digest_bytes(path.read_bytes()) if path.exists() else None
                    if path.exists():
                        path.unlink()
                    mutations.append(
                        self._mutation(
                            pending.ref,
                            "delete",
                            before,
                            None,
                            pending.rationale,
                            pending.author,
                        )
                    )
        finally:
            self._closed = True
        return mutations

    async def abandon(self) -> None:
        self._closed = True
        self._pending.clear()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("resource transaction is closed")

    def _mutation(
        self,
        ref: ResourceRef,
        op: str,
        before: str | None,
        after: str | None,
        rationale: str,
        author: WriterAuthor,
    ) -> ResourceMutation:
        return ResourceMutation(
            ref=ref,
            op=op,  # type: ignore[arg-type]
            before_version=before,
            after_version=after,
            metadata={
                "session_id": self._context.session_id,
                "turn_id": self._context.turn_id,
                "turn_index": self._context.turn_index,
                "rationale": rationale,
                "author": author,
            },
        )


def _resolve_inside(root: Path, path: str) -> Path:
    if Path(path).is_absolute():
        candidate = _real_path(Path(path))
    else:
        candidate = _real_path(root / path)
    real_root = _real_path(root)
    if candidate != real_root and real_root not in candidate.parents:
        raise ValueError(f"path escapes resource root: {path}")
    return candidate


def _real_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _digest_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _resource_mutation(
    ref: ResourceRef,
    op: str,
    before: str | None,
    after: str | None,
    rationale: str,
    author: WriterAuthor,
) -> ResourceMutation:
    return ResourceMutation(
        ref=ref,
        op=op,  # type: ignore[arg-type]
        before_version=before,
        after_version=after,
        metadata={"rationale": rationale, "author": author},
    )


__all__ = ["LocalResourceStore"]
