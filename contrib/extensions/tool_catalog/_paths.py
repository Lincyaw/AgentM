"""Shared path resolution for the tool catalog extension package."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from agentm.core.abi import AtomInfo, ExtensionAPI
from agentm.core.lib import expand_path, expand_path_from_cwd

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

def catalog_root(api: ExtensionAPI, raw_root: str | None = None) -> Path:
    if raw_root is None:
        return expand_path(api.cwd).resolve()
    return expand_path_from_cwd(raw_root, api.cwd).resolve()

def resolve_catalog_path(
    api: ExtensionAPI,
    raw_path: str,
    root: Path,
) -> ResolvedCatalogPath:
    atoms = api.list_atoms()
    atom = _atom_by_name(atoms, raw_path)
    if atom is not None and atom.source_path is not None:
        source_path = expand_path(atom.source_path).resolve()
        git_path = PurePosixPath(source_path.relative_to(root)).as_posix()
        return ResolvedCatalogPath(
            atom_name=atom.name,
            display_path=raw_path,
            git_path=git_path,
            writer_path=str(source_path),
        )

    candidate = expand_path(raw_path)
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
        writer_path = str(resolved)

    matched = _atom_by_source_path(atoms, resolved)
    if matched is not None:
        return ResolvedCatalogPath(
            atom_name=matched.name,
            display_path=raw_path,
            git_path=git_path,
            writer_path=(
                str(expand_path(matched.source_path).resolve())
                if matched.source_path is not None
                else writer_path
            ),
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
            if expand_path(atom.source_path).resolve() == resolved:
                return atom
        except OSError:
            continue
    return None
