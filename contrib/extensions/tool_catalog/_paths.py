"""Shared path resolution for the tool catalog extension package."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from agentm.core.abi.extension import AtomInfo, ExtensionAPI


class ResolvedCatalogPath:
    def __init__(
        self,
        *,
        atom_name: str | None,
        display_path: str,
        git_path: str,
        writer_path: str,
    ) -> None:
        self.atom_name = atom_name
        self.display_path = display_path
        self.git_path = git_path
        self.writer_path = writer_path


def catalog_root(api: ExtensionAPI, config: dict[str, Any]) -> Path:
    raw_root = config.get("root")
    if raw_root is None:
        root = Path(api.cwd)
    else:
        root = Path(str(raw_root))
        if not root.is_absolute():
            root = Path(api.cwd) / root
    return root.resolve()


def resolve_catalog_path(
    api: ExtensionAPI,
    raw_path: str,
    root: Path,
) -> ResolvedCatalogPath:
    atoms = api.list_atoms()
    atom = _atom_by_name(atoms, raw_path)
    if atom is not None and atom.source_path is not None:
        source_path = Path(atom.source_path).resolve()
        git_path = PurePosixPath(source_path.relative_to(root)).as_posix()
        return ResolvedCatalogPath(
            atom_name=atom.name,
            display_path=raw_path,
            git_path=git_path,
            writer_path=str(source_path),
        )

    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path {raw_path!r} is outside repo root {root}") from exc
        git_path = PurePosixPath(relative).as_posix()
        writer_path = str(resolved)
    else:
        resolved = (root / candidate).resolve()
        git_path = PurePosixPath(candidate).as_posix()
        writer_path = raw_path

    matched = _atom_by_source_path(atoms, resolved)
    if matched is not None:
        return ResolvedCatalogPath(
            atom_name=matched.name,
            display_path=raw_path,
            git_path=git_path,
            writer_path=str(Path(matched.source_path).resolve())
            if matched.source_path is not None
            else writer_path,
        )

    return ResolvedCatalogPath(
        atom_name=None,
        display_path=raw_path,
        git_path=git_path,
        writer_path=writer_path,
    )


def _atom_by_name(atoms: list[AtomInfo], name: str) -> AtomInfo | None:
    for atom in atoms:
        if atom.name == name:
            return atom
    return None


def _atom_by_source_path(atoms: list[AtomInfo], resolved: Path) -> AtomInfo | None:
    for atom in atoms:
        if atom.source_path is None:
            continue
        try:
            if Path(atom.source_path).resolve() == resolved:
                return atom
        except OSError:
            continue
    return None
