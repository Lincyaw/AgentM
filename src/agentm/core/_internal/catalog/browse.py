"""Read-only helpers for browsing git-backed catalog history."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast

from agentm.core.abi.catalog import ActiveSetFingerprint, ManifestSnapshot

ManifestLiteral: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | tuple["ManifestLiteral", ...]
    | list["ManifestLiteral"]
    | dict["ManifestLiteral", "ManifestLiteral"]
)
HistoricalManifest: TypeAlias = ManifestSnapshot

# ``_layout`` lives in :mod:`agentm.core.runtime.catalog._layout`. Browse
# stays in the kernel-internal catalog package and therefore computes the
# runs-dir path inline rather than importing the runtime module.


@dataclass(frozen=True, slots=True)
class CatalogAtom:
    name: str
    versions: tuple[str, ...]


class UnparseableManifestError(ValueError):
    """Raised when a historical ``MANIFEST`` cannot be statically decoded."""


def list_versions(path: str, root: Path | None = None) -> list[str]:
    root_path = _root_path(root)
    pathspec = _resolve_history_path(path, root=root_path)
    result = _run_git(
        ("log", "--format=%H", "--", pathspec),
        root=root_path,
        check=False,
    )
    result = cast(subprocess.CompletedProcess[str], result)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def current_version(path: str, root: Path | None = None) -> str:
    root_path = _root_path(root)
    pathspec = _resolve_history_path(path, root=root_path)
    result = _run_git(
        ("log", "-n", "1", "--format=%H", "--", pathspec),
        root=root_path,
        check=False,
    )
    result = cast(subprocess.CompletedProcess[str], result)
    if result.returncode != 0 or not result.stdout.strip():
        raise KeyError(f"No git history for path {path!r}")
    return result.stdout.strip()


def get_source_at(path: str, sha: str, root: Path | None = None) -> bytes:
    root_path = _root_path(root)
    pathspec = _resolve_history_path(path, root=root_path)
    commit_sha = _verify_commit_sha(sha, root=root_path)
    result = _run_git(
        ("cat-file", "-p", f"{commit_sha}:{pathspec}"),
        root=root_path,
        check=False,
        text=False,
    )
    result = cast(subprocess.CompletedProcess[bytes], result)
    if result.returncode != 0:
        stderr = _stderr_text(result).lower()
        if "exists on disk, but not in" in stderr or "path '" in stderr:
            raise KeyError(f"Path {pathspec!r} did not exist at {commit_sha}")
        raise KeyError(f"Unable to read {pathspec!r} at {commit_sha}: {stderr}")
    return result.stdout


def get_manifest_at(
    name: str, version: str, root: Path | None = None
) -> HistoricalManifest:
    source = get_source_at(name, version, root=root)
    try:
        tree = ast.parse(source.decode("utf-8"))
    except SyntaxError as exc:
        raise UnparseableManifestError(f"Failed to parse historical source: {exc}") from exc

    manifest_call = _find_manifest_call(tree)
    payload: HistoricalManifest = {
        key: _literal_value(node)
        for key, node in manifest_call.keywords.items()
    }
    payload["content_hash"] = _verify_commit_sha(version, root=_root_path(root))
    return payload


def runs_for(fingerprint: ActiveSetFingerprint | str, root: Path | None = None) -> list[str]:
    refs = _normalize_fingerprint(fingerprint)
    if not refs:
        return []

    cwd_root = (root or Path.cwd())
    trace_sets: list[set[str]] = []
    for name, version in refs.items():
        runs_dir = (
            cwd_root / ".agentm" / "catalog" / "atoms" / name / version / "runs"
        )
        if not runs_dir.exists():
            return []
        trace_sets.append({entry.name for entry in runs_dir.iterdir()})
    if not trace_sets:
        return []
    trace_ids = set.intersection(*trace_sets)
    return sorted(trace_ids)


def _normalize_fingerprint(
    fingerprint: ActiveSetFingerprint | str,
) -> dict[str, str]:
    if isinstance(fingerprint, str):
        atom, version = _split_atom_ref(fingerprint, None)
        return {atom: version}
    if not isinstance(fingerprint, dict):
        raise TypeError("fingerprint must be a dict or 'atom@version' string")

    raw_atoms = fingerprint.get("atoms")
    atom_map = raw_atoms if isinstance(raw_atoms, dict) else fingerprint
    refs: dict[str, str] = {}
    for key, value in atom_map.items():
        atom, version = _split_atom_ref(str(value), str(key))
        refs[atom] = version
    return refs


def _split_atom_ref(value: str, fallback_atom: str | None) -> tuple[str, str]:
    atom, separator, version = value.partition("@")
    if separator:
        if not atom or not version:
            raise ValueError(f"Invalid atom reference: {value!r}")
        return atom, version
    if fallback_atom is None:
        raise ValueError(f"Expected 'atom@version', got {value!r}")
    return fallback_atom, value


def _root_path(root: Path | None) -> Path:
    return (root or Path.cwd()).resolve()


def _resolve_history_path(path: str, *, root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path {path!r} is outside repo root {root}") from exc
        return PurePosixPath(relative).as_posix()

    if any(sep in path for sep in ("/", "\\")) or candidate.suffix:
        return PurePosixPath(candidate).as_posix()

    builtin_candidate = root / "src" / "agentm" / "extensions" / "builtin" / f"{path}.py"
    if builtin_candidate.exists():
        return PurePosixPath(builtin_candidate.relative_to(root)).as_posix()

    matches = sorted(root.rglob(f"{path}.py"))
    if len(matches) == 1:
        return PurePosixPath(matches[0].relative_to(root)).as_posix()
    if len(matches) > 1:
        raise ValueError(f"Atom name {path!r} is ambiguous under {root}")

    return PurePosixPath(candidate).as_posix()


def _verify_commit_sha(sha: str, *, root: Path) -> str:
    result = _run_git(
        ("rev-parse", "--verify", f"{sha}^{{commit}}"),
        root=root,
        check=False,
    )
    result = cast(subprocess.CompletedProcess[str], result)
    if result.returncode != 0:
        raise ValueError(f"Malformed or unknown commit SHA {sha!r}")
    return result.stdout.strip()


def _run_git(
    args: tuple[str, ...],
    *,
    root: Path,
    check: bool,
    text: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=text,
    )
    if check and completed.returncode != 0:
        stderr = _stderr_text(completed).strip() or _stdout_text(completed).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr or '<no output>'}")
    return completed


def _stdout_text(
    result: subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes],
) -> str:
    if isinstance(result.stdout, bytes):
        return result.stdout.decode("utf-8", errors="replace")
    return result.stdout


def _stderr_text(
    result: subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes],
) -> str:
    if isinstance(result.stderr, bytes):
        return result.stderr.decode("utf-8", errors="replace")
    return result.stderr


@dataclass(frozen=True, slots=True)
class _ManifestCall:
    keywords: dict[str, ast.expr]


def _find_manifest_call(tree: ast.Module) -> _ManifestCall:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "MANIFEST":
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            raise UnparseableManifestError("MANIFEST must be assigned from ExtensionManifest(...)")
        if not _is_extension_manifest_ctor(call.func):
            raise UnparseableManifestError("MANIFEST must call ExtensionManifest(...)")
        keywords: dict[str, ast.expr] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise UnparseableManifestError("MANIFEST may not use **kwargs expansion")
            keywords[keyword.arg] = keyword.value
        return _ManifestCall(keywords=keywords)
    raise UnparseableManifestError("No module-level MANIFEST = ExtensionManifest(...) found")


def _is_extension_manifest_ctor(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "ExtensionManifest"
    if isinstance(func, ast.Attribute):
        return func.attr == "ExtensionManifest"
    return False


def _literal_value(node: ast.expr) -> ManifestLiteral:
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise UnparseableManifestError("MANIFEST constants must be JSON-like scalars")
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_literal_value(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        payload: dict[ManifestLiteral, ManifestLiteral] = {}
        for key, item_value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise UnparseableManifestError("MANIFEST dict literals may not use unpacking")
            payload[_literal_value(key)] = _literal_value(item_value)
        return payload
    raise UnparseableManifestError(
        "MANIFEST keyword values must be literals, tuples, lists, or dicts of literals"
    )
