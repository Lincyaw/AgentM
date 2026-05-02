"""Freeze the currently loaded atom into the on-disk catalog."""

from __future__ import annotations

import inspect
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from agentm.core._internal.catalog import _layout
from agentm.core._internal.catalog.hashing import compute_atom_hash
from agentm.core.abi import EventBus
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import discover_builtin

if TYPE_CHECKING:
    from agentm.harness.resource_writer import GitBackedResourceWriter, WriteResult


_INDEXER_SESSION_ID = "catalog-freeze"


def freeze_current(
    name: str,
    source: str,
    manifest: ExtensionManifest,
    *,
    root: Path | None = None,
) -> str:
    if manifest.name != name:
        raise ValueError(
            f"manifest.name {manifest.name!r} does not match atom name {name!r}"
        )

    from agentm.harness.resource_writer import GitBackedResourceWriter

    cwd_root = (root or Path.cwd()).resolve()
    atom_path = _resolve_atom_source_path(name, root=cwd_root)
    relative_path = atom_path.relative_to(cwd_root)
    writer = GitBackedResourceWriter(
        cwd=str(cwd_root),
        session_id=_INDEXER_SESSION_ID,
        bus=EventBus(),
    )
    result = _write_via_writer(writer, relative_path, source)
    version_key = (
        result.commit_sha_after
        or result.commit_sha_before
        or compute_atom_hash(source)
    )
    _layout.atom_runs_dir(name, version_key, root=cwd_root).mkdir(
        parents=True,
        exist_ok=True,
    )
    return version_key


def source_path_for_hash(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    raise RuntimeError(
        "catalog source blobs moved to git; browse.py legacy helpers are handled in issue #3"
    )


def _resolve_atom_source_path(name: str, *, root: Path) -> Path:
    candidate = root / "src" / "agentm" / "extensions" / "builtin" / f"{name}.py"
    if candidate.is_file():
        return candidate.resolve()

    entry = discover_builtin().get(name)
    if entry is not None:
        source_path = inspect.getsourcefile(entry.module)
        if source_path is not None:
            resolved = Path(source_path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"cannot freeze {name!r} under root {root}: source path {resolved} is outside the repo root"
                ) from exc
            return resolved

    raise ValueError(f"cannot resolve source path for atom {name!r} under {root}")


def _write_via_writer(
    writer: "GitBackedResourceWriter",
    relative_path: Path,
    source: str,
) -> "WriteResult":
    import asyncio

    async def _write():
        return await writer.write(
            str(relative_path),
            source.encode("utf-8"),
            rationale="freeze_current snapshot",
            author="indexer",
        )

    result: list[WriteResult] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(_write()))
        except BaseException as exc:  # pragma: no cover - exercised by caller
            error.append(exc)

    thread = threading.Thread(target=_runner, name="agentm-freeze-current")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    assert result
    write_result = result[0]
    if write_result.error is not None:
        raise RuntimeError(write_result.error)
    return write_result
