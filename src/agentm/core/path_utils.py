"""Path normalization helpers shared by read/search tools."""

from __future__ import annotations

import os
import unicodedata

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
