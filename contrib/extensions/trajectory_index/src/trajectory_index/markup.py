"""Trajectory markup — the unified Pass 1 annotation language.

The extractor re-emits each annotated message's body verbatim with
``⟦tag key=value|content⟧`` spans inserted. Code verifies by stripping
every annotation and comparing against the original body: equality makes
every span offset exact — no substring search, no boundary drift, and
multi-segment provenance (agent/observation sandwiches) for free.

Grammar (nesting allowed):

    annotation := "⟦" tag attrs? "|" content "⟧"
    tag        := [a-z_]+
    attrs      := (SP key "=" value)*     value := '"' [^"]* '"' | bare-token

Delimiters are U+27E6/U+27E7 (mathematical white brackets): they do not
occur in web dumps — unlike ``[[...]]``, which collides with wiki links.

Tags in use — input side: ``known`` (already-extracted symbol, informative
only). Output side: ``sym`` (symbol declaration; attrs ``kind``, ``class``,
optional canonical ``name`` when the surface is not canonical), ``obs``
(retrieved/environment segment), ``claim`` (settled-fact assertion).

Verification is whitespace-tolerant: models occasionally normalize a
whitespace run while copying. Both sides are compared with runs collapsed,
and offsets are mapped back through a two-pointer alignment — content is
never altered, only whitespace differences are forgiven.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

OPEN = "⟦"    # ⟦
CLOSE = "⟧"   # ⟧

_TAG_RE = re.compile(r"[a-z_]+")
_ATTR_RE = re.compile(r'\s+([a-z_]+)=("(?:[^"]*)"|[^\s|⟦⟧]+)')


class MarkupError(ValueError):
    """Malformed annotation markup (unbalanced, missing '|', bad tag)."""


@dataclass(frozen=True, slots=True)
class Annotation:
    tag: str
    attrs: dict[str, str]
    start: int          # offsets into the PLAIN (stripped) text
    end: int
    depth: int = 0      # 0 = top-level; nested annotations have depth > 0

    @property
    def is_nested(self) -> bool:
        return self.depth > 0


def parse(annotated: str) -> tuple[str, list[Annotation]]:
    """Strip annotations, returning (plain text, spans with plain offsets).

    Raises :class:`MarkupError` on malformed markup — the caller rejects
    the whole message's annotations (logged), never repairs silently.
    """
    plain: list[str] = []
    plain_len = 0
    out: list[Annotation] = []
    stack: list[tuple[str, dict[str, str], int]] = []   # (tag, attrs, start)
    i = 0
    n = len(annotated)
    while i < n:
        ch = annotated[i]
        if ch == OPEN:
            m = _TAG_RE.match(annotated, i + 1)
            if m is None:
                raise MarkupError(f"missing tag after {OPEN!r} at {i}")
            tag = m.group(0)
            j = m.end()
            attrs: dict[str, str] = {}
            while True:
                am = _ATTR_RE.match(annotated, j)
                if am is None:
                    break
                val = am.group(2)
                attrs[am.group(1)] = val[1:-1] if val.startswith('"') else val
                j = am.end()
            if j >= n or annotated[j] != "|":
                raise MarkupError(f"missing '|' after tag {tag!r} at {i}")
            stack.append((tag, attrs, plain_len))
            i = j + 1
        elif ch == CLOSE:
            if not stack:
                raise MarkupError(f"unbalanced {CLOSE!r} at {i}")
            tag, attrs, start = stack.pop()
            out.append(Annotation(
                tag=tag, attrs=attrs, start=start, end=plain_len, depth=len(stack),
            ))
            i += 1
        else:
            plain.append(ch)
            plain_len += 1
            i += 1
    if stack:
        raise MarkupError(f"unclosed {OPEN!r} for tag {stack[-1][0]!r}")
    out.sort(key=lambda a: (a.start, -a.end))
    return "".join(plain), out


def strip(annotated: str) -> str:
    """Plain text only (parse and discard spans)."""
    return parse(annotated)[0]


# ---------------------------------------------------------------------------
# Whitespace-tolerant alignment
# ---------------------------------------------------------------------------


def align(emitted: str, original: str) -> list[int] | None:
    """Map each offset of ``emitted`` to an offset of ``original``.

    Exact match maps identically. Otherwise both are walked with two
    pointers, forgiving whitespace-run differences only: any non-space
    mismatch fails (returns None). The returned list has
    ``len(emitted) + 1`` entries (the last maps end-of-string), each an
    offset into ``original``.
    """
    if emitted == original:
        return list(range(len(emitted) + 1))
    mapping: list[int] = []
    i = j = 0
    ne, no = len(emitted), len(original)
    while i < ne:
        ce = emitted[i]
        if j < no and ce == original[j]:
            mapping.append(j)
            i += 1
            j += 1
            continue
        if ce.isspace():
            # forgive: swallow the emitted whitespace run, sync original
            k = i
            while k < ne and emitted[k].isspace():
                k += 1
            jj = j
            while jj < no and original[jj].isspace():
                jj += 1
            mapping.extend([min(j, no)] * (k - i))
            i, j = k, jj
            continue
        if j < no and original[j].isspace():
            j += 1
            continue
        return None
    while j < no and original[j].isspace():
        j += 1
    if j != no:
        return None
    mapping.append(no)
    return mapping


# ---------------------------------------------------------------------------
# Input-side marking (known symbols)
# ---------------------------------------------------------------------------

_IDENT_CHAR = re.compile(r"[A-Za-z0-9_.\-/]")


def _at_word_boundary(text: str, start: int, end: int) -> bool:
    if start > 0 and _IDENT_CHAR.match(text[start - 1]):
        return False
    return not (end < len(text) and _IDENT_CHAR.match(text[end]))


def mark_known(text: str, known_names: list[str]) -> str:
    """Wrap known symbol occurrences with ``⟦known|...⟧`` (word-boundary,
    longest-name-first, never inside an existing mark)."""
    if not known_names:
        return text
    for name in sorted(known_names, key=len, reverse=True):
        if not name or len(name) < 2:
            continue
        lower = text.lower()
        search = name.lower()
        parts: list[str] = []
        pos = 0
        while pos < len(text):
            idx = lower.find(search, pos)
            if idx < 0:
                parts.append(text[pos:])
                break
            end = idx + len(name)
            already = text.rfind(OPEN, 0, idx) > text.rfind(CLOSE, 0, idx)
            if already or not _at_word_boundary(text, idx, end):
                parts.append(text[pos:end])
                pos = end
                continue
            parts.append(text[pos:idx])
            parts.append(f"{OPEN}known|{text[idx:end]}{CLOSE}")
            pos = end
        text = "".join(parts)
    return text
