"""Path normalization helpers shared by read/search tools."""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
import unicodedata

from agentm.core.abi.operations import FileOperations

_UNICODE_SPACE_MAP = {
    ord(ch): ord(" ")
    for ch in "\u00A0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000"
}
_NARROW_NO_BREAK_SPACE = "\u202F"


def expand_path(p: str) -> str:
    normalized = p.lstrip("@").translate(_UNICODE_SPACE_MAP)
    if normalized == "~":
        return os.path.expanduser("~")
    if normalized.startswith("~/"):
        return os.path.expanduser(normalized)
    return normalized


def resolve_to_cwd(p: str, cwd: str) -> str:
    expanded = expand_path(p)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(cwd, expanded))


def resolve_read_path(p: str, cwd: str) -> str:
    resolved = resolve_to_cwd(p, cwd)
    variants = (
        resolved,
        resolved.replace(" AM.", f"{_NARROW_NO_BREAK_SPACE}AM.").replace(
            " PM.", f"{_NARROW_NO_BREAK_SPACE}PM."
        ),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "\u2019"),
        unicodedata.normalize("NFD", resolved).replace("'", "\u2019"),
    )
    for candidate in variants:
        if os.path.exists(candidate):
            return candidate
    return resolved


def load_gitignore_patterns(root: str, *, extra: Iterable[str] = ()) -> list[str]:
    patterns = list(extra)
    for dirpath, _dirnames, filenames in os.walk(root):
        if ".gitignore" not in filenames:
            continue
        prefix = os.path.relpath(dirpath, root).replace(os.sep, "/")
        prefix = "" if prefix == "." else prefix
        raw_text = Path(os.path.join(dirpath, ".gitignore")).read_text(encoding="utf-8")
        for raw in raw_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if prefix:
                if line.startswith("/"):
                    patterns.append(f"{prefix}{line}")
                else:
                    patterns.append(f"{prefix}/{line.lstrip('/')}")
                continue
            patterns.append(line)
    return patterns


async def load_gitignore_patterns_from_file_ops(
    file_ops: FileOperations,
    root: str,
    *,
    extra: Iterable[str] = (),
) -> list[str]:
    patterns = list(extra)
    if not await file_ops.is_dir(root):
        return patterns

    async def _walk(path: str, rel_dir: str) -> None:
        try:
            names = await file_ops.list_dir(path)
        except Exception:
            return
        if ".gitignore" in names:
            gitignore_path = os.path.join(path, ".gitignore")
            try:
                raw_text = (await file_ops.read_file(gitignore_path)).decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                raw_text = ""
            patterns.extend(_prefix_gitignore_lines(raw_text, rel_dir))

        for name in names:
            if name == ".gitignore":
                continue
            child = os.path.join(path, name)
            try:
                is_dir = await file_ops.is_dir(child)
            except Exception:
                is_dir = False
            if is_dir:
                child_rel = f"{rel_dir}/{name}".strip("/")
                await _walk(child, child_rel)

    await _walk(root, "")
    return patterns


def _prefix_gitignore_lines(raw_text: str, prefix: str) -> list[str]:
    patterns: list[str] = []
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if prefix:
            if line.startswith("/"):
                patterns.append(f"{prefix}{line}")
            else:
                patterns.append(f"{prefix}/{line.lstrip('/')}")
            continue
        patterns.append(line)
    return patterns
