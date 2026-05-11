# Design: Search Tool Atoms (`grep`, `find`, `ls`)

**Status**: DRAFT
**Created**: 2026-05-01
**Last Updated**: 2026-05-09

## Overview

Three new tool atoms — `grep`, `find`, `ls` — implemented as deterministic stdlib + `pathspec` traversals behind the shared `FileOperations` read seam. They respect `.gitignore` where search semantics require it and apply deterministic truncation rules. They sit alongside the existing `tool_read` / `tool_bash` / `tool_edit` / `tool_write` atoms and follow the issue #89 hybrid IO seam: read-only file tools consume `FileOperations`, while write tools consume `ResourceWriter`.

## Motivation

AgentM's `general_purpose` scenario currently has `read`, `bash`, `edit`, `write` — but no first-class search. Models route around the gap by shelling out (`bash` with `find . -name ...` / `grep -r`), which is fragile (no `.gitignore`, no truncation, no structured output) and floods the bash tool's audit trail. Pi proves that splitting search into typed tools with their own truncation rules dramatically improves the model's behavior on large repos.

## Design Details

### Shared utilities (Tier 1, consumed by all three atoms)

#### `core/lib/text_truncate.py`

Port of `truncate.ts`. Pure functions, no I/O.

```python
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_LENGTH = 500

@dataclass(frozen=True, slots=True)
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

def format_size(bytes_: int) -> str: ...
def truncate_head(content: str, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult: ...
def truncate_tail(content: str, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult: ...
def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]: ...
```

Byte counting uses `len(s.encode("utf-8"))`. Multi-byte tail truncation walks UTF-8 continuation bytes (`b & 0xC0 == 0x80`) — same algorithm as `truncate.ts:236-251`.

#### `core/lib/path_utils.py`

Port of `path-utils.ts`.

```python
def expand_path(p: str) -> str:
    """`~`, `~/...` expansion; strip leading `@`; normalize Unicode spaces."""

def resolve_to_cwd(p: str, cwd: str) -> str:
    """Absolute path resolution against `cwd`."""

def resolve_read_path(p: str, cwd: str) -> str:
    """As `resolve_to_cwd`, but on macOS also tries NFD-normalized and
    curly-quote variants for screenshot filenames. Falls through to the
    original path if no variant exists."""
```

The macOS variants are real-world fixes pi accumulated; we port them verbatim (`os.path.normpath` is not enough). Use `unicodedata.normalize("NFD", p)` for the NFD variant.

### Tool atom: `extensions/builtin/tool_grep.py`

**Parameters** (JSON schema):
```python
{
  "type": "object",
  "properties": {
    "pattern": {"type": "string", "description": "Search pattern (regex by default; literal if `literal=true`)."},
    "path": {"type": "string", "description": "Directory or file (default: cwd)."},
    "glob": {"type": "string", "description": "Filter files by glob, e.g. '*.py'."},
    "ignore_case": {"type": "boolean", "default": false},
    "literal": {"type": "boolean", "default": false},
    "context": {"type": "integer", "description": "Lines of context before/after each match.", "default": 0},
    "limit": {"type": "integer", "description": "Max matches.", "default": 100}
  },
  "required": ["pattern"]
}
```

**Backend selection** (in order):

1. Use the active `FileOperations` (`config["file_ops"]` or `api.get_operations().file`) for all path checks, directory listing, and file reads.
2. Compile the query with stdlib `re` (`re.escape(pattern)` when `literal=true`).
3. Traverse directories through `FileOperations.list_dir` / `is_dir`, honoring `.gitignore` via `pathspec` patterns loaded through `FileOperations.read_file`.

Fail with a clear error if `path` doesn't exist. Do not shell out from the atom: that would bypass the read seam scenario authors override for SSH/sandbox/in-memory filesystems.

**Output rules**:

- Sort by file path then line number.
- For each match: `<rel_path>:<line>: <line_text>` (truncate line to `GREP_MAX_LINE_LENGTH=500` via `truncate_line`).
- With `context > 0`: blocks separated by `--`; non-match context lines use `<rel_path>-<line>- <text>`.
- After collection, apply `truncate_head(output, max_lines=∞, max_bytes=DEFAULT_MAX_BYTES)`.
- Append `[Truncated: <reasons>]` line listing which limits were hit (match limit, byte limit, line truncation).

**Pluggable operations**:

`grep` does not define a bespoke per-tool Protocol. It uses the shared read-only `FileOperations` Protocol from `agentm.core.abi.operations` and accepts `config["file_ops"]` for tests/scenarios, defaulting to `api.get_operations().file`. SSH/sandbox/in-memory scenarios swap the same `FileOperations` bundle used by `read`, `find`, and `ls`.

### Tool atom: `extensions/builtin/tool_find.py`

**Parameters**:
```python
{
  "type": "object",
  "properties": {
    "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' or '**/*.spec.py'."},
    "path": {"type": "string", "description": "Directory to search (default: cwd)."},
    "limit": {"type": "integer", "default": 1000}
  },
  "required": ["pattern"]
}
```

**Backend selection**:

1. Use the active `FileOperations` (`config["file_ops"]` or `api.get_operations().file`) for existence checks, directory tests, directory listing, and `.gitignore` reads.
2. Use `pathspec` to apply the requested glob and ignore rules. Bare patterns (no slash) are expanded to match basenames anywhere in the tree.

**Output**:

- Relative paths to `search_path`, POSIX-separated. Trailing `/` for directories preserved (pi includes them — useful for the model).
- Sort lexicographically.
- Apply `truncate_head(..., max_bytes=DEFAULT_MAX_BYTES)`.
- Append `[Truncated: ...]` if limit hit.

**Pluggable operations**:

`find` uses the shared `FileOperations` read seam, not a bespoke per-tool Protocol. `config["file_ops"]` may supply a scenario/test implementation; otherwise the atom uses `api.get_operations().file`.

### Tool atom: `extensions/builtin/tool_ls.py`

**Parameters**:
```python
{
  "type": "object",
  "properties": {
    "path": {"type": "string", "default": "."},
    "limit": {"type": "integer", "default": 500}
  }
}
```

**Backend**: active `FileOperations`. No external binary. Sort case-insensitive. Append `/` to directory entries using `FileOperations.is_dir`.

Truncate via `truncate_head(...)` after the entry-count limit is applied.

**Pluggable operations**:

`ls` uses the shared `FileOperations` read seam, not a bespoke per-tool Protocol. It lists with `FileOperations.list_dir` and checks path existence/directory status with `FileOperations.access` / `is_dir`.

### Cross-cutting rules

- `from __future__ import annotations` mandatory.
- Imports limited to: stdlib, `pathspec`, `agentm.core.abi.*`, `agentm.core.lib.text_truncate`, `agentm.core.lib.path_utils`, `agentm.core.abi.operations` (for `FileOperations`), `agentm.harness.extension`, and `agentm.extensions` (for `ExtensionManifest`). Atom-to-atom imports forbidden ([extension-as-scenario.md §11.1](extension-as-scenario.md#111-hard-rules)).
- Each atom's `MANIFEST.registers` includes exactly one `tool:<name>` tag.
- Cancellation: every `execute` accepts `signal: asyncio.Event`. Traversal checks the signal between awaited read/list operations.
- Errors: raised as `Exception` with a single sentence. The kernel wraps them into a tool-error result.
- No direct subprocess calls from these atoms; shelling out would bypass `FileOperations`.
- No `pathlib.Path` operations on user-controlled paths beyond `resolve_to_cwd`. Direct filesystem IO routes through `FileOperations`.

### Default-scenario integration

Update `extensions/scenarios/general_purpose.yaml` (existing file) to load the three new atoms:

```yaml
extensions:
  - module: agentm.extensions.builtin.tool_read
  - module: agentm.extensions.builtin.tool_bash
  - module: agentm.extensions.builtin.tool_edit
  - module: agentm.extensions.builtin.tool_write
  - module: agentm.extensions.builtin.tool_grep
  - module: agentm.extensions.builtin.tool_find
  - module: agentm.extensions.builtin.tool_ls
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: "..."
```

The `rca.yaml` and `trajectory_analysis.yaml` scenarios opt in by adding the lines they need. `plan_mode.yaml` should *also* gain `grep`/`find`/`ls` since planning often requires read-only exploration.

## Interface Definition

See per-atom blocks above. Public Python surface:

```python
# core/lib/text_truncate.py
class TruncationResult: ...
def truncate_head(...) -> TruncationResult: ...
def truncate_tail(...) -> TruncationResult: ...
def truncate_line(...) -> tuple[str, bool]: ...
def format_size(...) -> str: ...

# core/lib/path_utils.py
def expand_path(p: str) -> str: ...
def resolve_to_cwd(p: str, cwd: str) -> str: ...
def resolve_read_path(p: str, cwd: str) -> str: ...

# extensions/builtin/{tool_grep,tool_find,tool_ls}.py
MANIFEST: ExtensionManifest
def install(api: ExtensionAPI, config: dict[str, Any]) -> None: ...
```

## Acceptance Scenarios

1. `grep(pattern="def foo")` on a repo with `.gitignore`-listed `dist/` returns no matches under `dist/`.
2. `grep(pattern="x", limit=5)` returning more than 5 hits emits a truncation notice and stops traversal once the match limit is reached.
3. `grep` uses `re` + `pathspec` through `FileOperations` and never requires an external search binary to be installed.
4. `find(pattern="**/*.py")` returns POSIX paths; `node_modules/` and `.git/` entries are excluded.
5. `ls(path=cwd)` returns sorted entries with `/` suffix on directories.
6. A tool call to a non-existent path returns a clear `Path not found: <path>` error, not a stack trace.
7. Output crossing the 50KB byte budget is head-truncated with `[Truncated: 50.0KB limit reached]` appended.
8. The same atom, given a custom `FileOperations` impl pointing at an SSH host or sandbox, returns matches from that remote/sandbox FS without any code change.

## Related Concepts

- [extension-as-scenario.md](extension-as-scenario.md) §7.1 — tool atom catalog (this design extends it)
- [pluggable-architecture.md](pluggable-architecture.md) §3.2 — issue #89 hybrid IO seam: read-only file atoms reuse shared `FileOperations`; write atoms use `ResourceWriter`
- [edit-diff.md](edit-diff.md) — sibling tool-foundation module landing in the same wave
- [tool-event-narrowing.md](tool-event-narrowing.md) — these tools each contribute a typed `ToolCallEvent`/`ToolResultEvent` variant

## Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Always use `FileOperations` + stdlib/pathspec traversal | Keeps every read behind the same scenario-overridable seam | Shell out to external search binaries (faster locally but bypasses SSH/sandbox FileOperations overrides) |
| **Do not** auto-download binaries | Pi's `ensureTool` violates least-surprise; user owns their binaries | Auto-fetch — security and disk-quota issues |
| `pathspec` for `.gitignore` parsing | Battle-tested, handles all edge cases | Hand-roll — bugs in `**`, negation, anchored patterns |
| Shared `FileOperations` instead of per-tool Protocols | One read seam for `read`/`grep`/`find`/`ls`; matches issue #89 Option C | Bespoke per-tool search/list protocols — more seams for scenario authors |
| 500-char per-line truncation in grep, 50KB total | Pi defaults; bias toward more useful output for the model | Bigger limits — context budget pressure |
| Each atom is one `.py` file ≤ 300 LoC | [§11.1](extension-as-scenario.md#111-hard-rules) hard rule | Subpackage per tool — violates contract |

## Out of Scope

- Semantic / embedding-based search.
- `grep --replace` (covered by `tool_edit`).
- Watching files for changes (`fswatch`-style notifications).
- Cross-host search aggregation.

## Open Questions

- [ ] Should `tool_ls` honor `.gitignore` by default? Pi does **not**; a literal `ls` shows everything. Recommendation: keep parity — do not filter.
- [ ] Should grep return JSON instead of text for easier programmatic consumption? Defer until a real consumer asks.
