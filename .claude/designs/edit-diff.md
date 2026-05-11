# Design: Edit-Diff Module (Line-Ending / BOM / Fuzzy-Match Foundations)

**Status**: DRAFT
**Created**: 2026-05-01
**Last Updated**: 2026-05-01

## Overview

A pure-function module `core/lib/edit_diff.py` containing the edit-application algorithm used by `tool_edit` (and reusable by future scenarios needing in-place file mutation): line-ending detection/restoration, BOM handling, NFKC + smart-quote / dash / space fuzzy normalization, and the multi-edit application pipeline that produces a final new content string from a base content + a list of `Edit{old, new}` operations.

## Motivation

AgentM's existing `tool_edit` does plain string replace. That fails in three real scenarios:

1. **CRLF files** — the algorithm reads CRLF, normalizes to LF for matching, applies the edit, then restores CRLF on write. Plain `str.replace` mangles line endings if the model's `old_string` was learned from a previous `read` call (which usually returns LF-normalized text).
2. **BOM-prefixed files** — UTF-8 BOM gets accidentally consumed/emitted by naive readers; the edit succeeds but the file's BOM disappears.
3. **Smart-quote / em-dash drift** — the model writes `—` (U+2014) in `old_string`; the file has `--`. Exact match fails. Fuzzy normalization (NFKC + Unicode-dash → ASCII hyphen + smart-quote → ASCII quote + Unicode-space → space) lets the match succeed and the replacement uses the *normalized* content (acceptable: the side effect is normalizing minor formatting).

This module owns those rules so they live in one place and are independently testable.

## Design Details

### Public API

```python
# core/lib/edit_diff.py
from typing import Literal, NamedTuple

LineEnding = Literal["\n", "\r\n"]

@dataclass(frozen=True, slots=True)
class Edit:
    old_text: str
    new_text: str

@dataclass(frozen=True, slots=True)
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str   # original or fuzzy-normalized

@dataclass(frozen=True, slots=True)
class AppliedEditsResult:
    base_content: str          # the content edits were applied against (after BOM strip + LF normalize, possibly fuzzy-normalized)
    new_content: str           # the result, ready to write *after* re-applying BOM and original line ending

def detect_line_ending(content: str) -> LineEnding: ...
def normalize_to_lf(text: str) -> str: ...
def restore_line_endings(text: str, ending: LineEnding) -> str: ...
def normalize_for_fuzzy_match(text: str) -> str: ...
def strip_bom(content: str) -> tuple[str, str]:
    """Returns `(bom, text_without_bom)`. `bom` is `"﻿"` or `""`."""
def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult: ...
def apply_edits(base: str, edits: list[Edit]) -> AppliedEditsResult:
    """Apply edits sequentially. Each edit is matched (exact then fuzzy) against
    the current content. Raises if any edit's old_text is not found, or if it
    matches multiple times AND the caller did not opt into replace_all (handled
    one level up in tool_edit). Implementation detail: when a fuzzy match
    occurs, the *whole content* switches to the fuzzy-normalized form for the
    rest of the edit chain — keeps later edits from oscillating between
    normalized and original."""
```

### Edit-application algorithm

1. Read raw bytes, decode UTF-8 (errors=`replace` for safety).
2. `bom, text = strip_bom(raw)`.
3. `ending = detect_line_ending(text)`.
4. `lf = normalize_to_lf(text)`.
5. `current = lf`.
6. For each `Edit(old, new)`:
   a. `m = fuzzy_find_text(current, old)`.
   b. If not found: raise `EditNotFound(path, edit_index, total_edits)` with a helpful message ("Could not find the exact text in <path>. The old text must match exactly including all whitespace and newlines.").
   c. Count occurrences. If > 1 and the caller required uniqueness: raise `EditAmbiguous(path, edit_index, count)`.
   d. `current = m.content_for_replacement[:m.index] + new + m.content_for_replacement[m.index + m.match_length:]`.
7. `final = bom + restore_line_endings(current, ending)`.
8. Return `AppliedEditsResult(base_content=lf, new_content=final)`.

The `current = m.content_for_replacement` swap on first fuzzy match is critical and subtle — without it, a sequence of edits where edit #1 fuzzy-matches and edit #2 exact-matches against the *original* content can produce inconsistent results.

### Detection rules (port verbatim)

- `detect_line_ending`: find first `\r\n` and first `\n`; if no `\n`, return `\n`; else if `\r\n` index < `\n` index (or no `\r\n` found, but `\n` found), return `\n`; else `\r\n`. Tie-breaks favor LF.
- `strip_bom`: `content[0] == "﻿"` → `("﻿", content[1:])` else `("", content)`.
- `normalize_for_fuzzy_match`:
  1. `unicodedata.normalize("NFKC", text)`.
  2. Strip trailing whitespace per line.
  3. Smart single quotes (U+2018, U+2019, U+201A, U+201B) → `'`.
  4. Smart double quotes (U+201C, U+201D, U+201E, U+201F) → `"`.
  5. Unicode hyphens / dashes (U+2010, U+2011, U+2012, U+2013, U+2014, U+2015, U+2212) → `-`.
  6. Unicode spaces (U+00A0, U+2002–U+200A, U+202F, U+205F, U+3000) → ` `.

### How `tool_edit` uses it

Tier 2 work — the implementer of the `tool_edit` upgrade must:

1. Replace the current naive `str.replace` body with `apply_edits([Edit(old, new)])`.
2. Add `replace_all: bool` parameter — when `False` (default), uniqueness check raises `EditAmbiguous`; when `True`, all matches are replaced (Python `str.replace` with no count limit, but in the *fuzzy-normalized space* if the first match was fuzzy).
3. Return a small diff preview in `ToolResult.extras` (line-level: count of additions/deletions). Use stdlib `difflib.unified_diff` for the actual rendered diff text.
4. Persist via `FileOperations.write_file(path, final_bytes)` — re-encode `final` as UTF-8.

The atom imports `from agentm.core.lib import edit_diff` directly: `agentm.core.lib.*` is on the §11.1 allow-list as the constitution's pure-function utility shelf (stdlib-style). No allow-list amendment is needed.

Same arrangement applies for `core/lib/text_truncate.py` and `core/lib/path_utils.py` from [search-tools.md](search-tools.md).

### Python-equivalent for Node specifics

| Pi (Node.js) | AgentM (Python) |
|---|---|
| `diff` npm package for unified diff | stdlib `difflib.unified_diff` |
| Buffer-level BOM detection | String-level (we decode first) |
| `String.prototype.normalize("NFKC")` | `unicodedata.normalize("NFKC", s)` |

## Interface Definition

See API block above. Stdlib-only module: imports `unicodedata`, `re`, `difflib`. No I/O. No external deps.

## Acceptance Scenarios

1. CRLF file edited by an LF-formatted `old_string` succeeds; written file retains CRLF.
2. UTF-8 BOM file edited; output still starts with BOM.
3. Edit with `old_string` containing em-dash succeeds against file with ASCII hyphens (fuzzy match); diagnostic notes fuzzy match used.
4. Two-edit chain where edit 1 is fuzzy and edit 2 is exact: both apply; neither silently no-ops.
5. `EditNotFound` raised when `old_string` matches nothing even after fuzzy normalization.
6. `EditAmbiguous` raised when `old_string` (or its fuzzy form) matches multiple places and `replace_all=False`.
7. `apply_edits` is pure: same input → same output; no side effects.
8. `restore_line_endings` round-trip: `restore(normalize_to_lf(s), detect(s)) == s` for any `s` that started with consistent line endings.

## Related Concepts

- [extension-as-scenario.md](extension-as-scenario.md) §7.1 (`tool_edit` row)
- [search-tools.md](search-tools.md) — sibling Tier 2 module sharing the `core/lib/path_utils.py` neighbor
- [pluggable-architecture.md](pluggable-architecture.md) §3.2 — `FileOperations` port the upgraded `tool_edit` writes through

## Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Fuzzy-match swaps the *whole content* to normalized form on first hit | Avoids inconsistent state across multi-edit chains | Track fuzzy/exact per edit — bug magnet |
| BOM and line ending restored at the very end, not inside `apply_edits` | Lets `apply_edits` stay a pure string function with no encoding awareness | Make `apply_edits` accept bytes — couples it to I/O |
| `replace_all` semantics handled in `tool_edit`, not `apply_edits` | Keeps `apply_edits` contract clean (one occurrence required by default) | Pass `replace_all` down — bigger surface |
| Use stdlib `difflib` for previews | Zero dep cost; "good enough" diff is good enough | Pull in `diff` lib equivalent (`diff-match-patch`) |
| Place `edit_diff` in `core/lib/` so `tool_edit` imports stdlib-style | Without this, the algorithm has nowhere to live | Inline into `tool_edit.py` — exceeds 300-LoC budget |

## Out of Scope

- Three-way merge / conflict resolution.
- Patch application (unified-diff input). The model produces explicit `old`/`new` pairs, not patches.
- Whitespace-insensitive structural matching (AST-level).

## Open Questions

- [ ] Should `apply_edits` return the *list of effective edits* (with their post-fuzzy positions) so callers can render position-accurate diffs? Defer until a real renderer needs it.
- [ ] Do we honor a `.editorconfig` discovered upward from the file? Pi does not. Defer.
