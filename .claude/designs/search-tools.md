# Design: Search Tool Atoms (`grep`, `find`, `ls`)

**Status**: DRAFT
**Created**: 2026-05-01
**Last Updated**: 2026-05-01

## Overview

Three new tool atoms — `grep`, `find`, `ls` — backed by `ripgrep` / `fd` (with safe fallbacks), respecting `.gitignore`, with deterministic truncation rules. Sit alongside the existing `tool_read` / `tool_bash` / `tool_edit` / `tool_write` atoms. Pi-mono references: [`tools/grep.ts`](/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/tools/grep.ts), [`tools/find.ts`](/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/tools/find.ts), [`tools/ls.ts`](/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/tools/ls.ts), plus shared utilities [`tools/truncate.ts`](/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/tools/truncate.ts) and [`tools/path-utils.ts`](/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/tools/path-utils.ts).

## Motivation

AgentM's `general_purpose` scenario currently has `read`, `bash`, `edit`, `write` — but no first-class search. Models route around the gap by shelling out (`bash` with `find . -name ...` / `grep -r`), which is fragile (no `.gitignore`, no truncation, no structured output) and floods the bash tool's audit trail. Pi proves that splitting search into typed tools with their own truncation rules dramatically improves the model's behavior on large repos.

## Design Details

### Shared utilities (Tier 1, consumed by all three atoms)

#### `core/text_truncate.py`

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

#### `core/path_utils.py`

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

1. `ripgrep` (`rg`) if `shutil.which("rg")` returns a path. Spawn via `asyncio.create_subprocess_exec`. Use `--json --line-number --color=never --hidden`. Parse JSON-Lines from stdout for `match` events; collect `{file_path, line_number, line_text}`.
2. **Fallback**: stdlib `re` + `os.walk` traversal. `.gitignore` honored via `pathspec` (also used by `skills.md`). Slower; documented as the fallback.

Fail with a clear error if `path` doesn't exist. **Do not** auto-download `rg` — pi's `ensureTool` is a project-specific concession we explicitly reject; the user installs `ripgrep` via their package manager.

**Output rules**:

- Sort by file path then line number.
- For each match: `<rel_path>:<line>: <line_text>` (truncate line to `GREP_MAX_LINE_LENGTH=500` via `truncate_line`).
- With `context > 0`: blocks separated by `--`; non-match context lines use `<rel_path>-<line>- <text>`.
- After collection, apply `truncate_head(output, max_lines=∞, max_bytes=DEFAULT_MAX_BYTES)`.
- Append `[Truncated: <reasons>]` line listing which limits were hit (match limit, byte limit, line truncation).

**Pluggable operations** (parallel to `FileOperations`/`BashOperations`):

```python
class GrepOperations(Protocol):
    async def is_directory(self, path: str) -> bool: ...
    async def read_file(self, path: str) -> str: ...
```

Default impl uses stdlib + `core/operations.py` `LocalFileOperations`. SSH-backed scenarios swap in a remote impl. Atom config: `config["ops"] = GrepOperations()` (defaults to local).

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

1. `fd` if available. `--glob --color=never --hidden --no-require-git --max-results=<limit>`. If pattern contains `/`, also pass `--full-path` and prepend `**/` if the user pattern doesn't already start with `**/` or `/`. (Verbatim port of `find.ts:228-249`.)
2. **Fallback**: stdlib `pathspec.PathSpec` + `os.walk(followlinks=True)`. Build spec from each `.gitignore` encountered. Match `pattern` against `Path.relative_to(search_path).as_posix()`.

**Output**:

- Relative paths to `search_path`, POSIX-separated. Trailing `/` for directories preserved (pi includes them — useful for the model).
- Sort lexicographically.
- Apply `truncate_head(..., max_bytes=DEFAULT_MAX_BYTES)`.
- Append `[Truncated: ...]` if limit hit.

**Pluggable operations**:

```python
class FindOperations(Protocol):
    async def exists(self, path: str) -> bool: ...
    async def glob(self, pattern: str, cwd: str, *, ignore: list[str], limit: int) -> list[str]: ...
```

If `config["ops"]` is supplied with a non-default `glob`, the atom uses it instead of `fd` (matches pi's branching).

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

**Backend**: pure stdlib (`os.scandir`). No external binary. Sort case-insensitive. Append `/` to directory entries.

Truncate via `truncate_head(...)` after the entry-count limit is applied.

**Pluggable operations**:

```python
class LsOperations(Protocol):
    async def exists(self, path: str) -> bool: ...
    async def stat(self, path: str) -> os.stat_result: ...
    async def listdir(self, path: str) -> list[str]: ...
```

### Cross-cutting rules

- `from __future__ import annotations` mandatory.
- Imports limited to: stdlib, `agentm.core.kernel.*`, `agentm.core.text_truncate`, `agentm.core.path_utils`, `agentm.core.operations` (for `FileOperations` reuse), `agentm.harness.extension`, `agentm.harness.events`, `agentm.extensions` (for `ExtensionManifest`). Atom-to-atom imports forbidden ([extension-as-scenario.md §11.1](extension-as-scenario.md#111-hard-rules)).
- Each atom's `MANIFEST.registers` includes exactly one `tool:<name>` tag.
- Cancellation: every `execute` accepts `signal: asyncio.Event`. The subprocess implementations call `proc.terminate()` when set.
- Errors: raised as `Exception` with a single sentence. The kernel wraps them into a tool-error result.
- No `subprocess.run` (sync); all subprocess work goes through `asyncio.create_subprocess_exec`.
- No `pathlib.Path` operations on user-controlled paths beyond `resolve_to_cwd`. Direct filesystem IO routes through the configured `Operations` impl when one is provided.

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
# core/text_truncate.py
class TruncationResult: ...
def truncate_head(...) -> TruncationResult: ...
def truncate_tail(...) -> TruncationResult: ...
def truncate_line(...) -> tuple[str, bool]: ...
def format_size(...) -> str: ...

# core/path_utils.py
def expand_path(p: str) -> str: ...
def resolve_to_cwd(p: str, cwd: str) -> str: ...
def resolve_read_path(p: str, cwd: str) -> str: ...

# extensions/builtin/{tool_grep,tool_find,tool_ls}.py
MANIFEST: ExtensionManifest
def install(api: ExtensionAPI, config: dict[str, Any]) -> None: ...
```

## Acceptance Scenarios

1. `grep(pattern="def foo")` on a repo with `.gitignore`-listed `dist/` returns no matches under `dist/`.
2. `grep(pattern="x", limit=5)` returning more than 5 hits emits truncation notice and stops `rg` early.
3. `grep` falls back to `re` + `pathspec` when `rg` is absent and produces equivalent output (within line-truncation rules).
4. `find(pattern="**/*.py")` returns POSIX paths; `node_modules/` and `.git/` entries are excluded.
5. `ls(path=cwd)` returns sorted entries with `/` suffix on directories.
6. A tool call to a non-existent path returns a clear `Path not found: <path>` error, not a stack trace.
7. Output crossing the 50KB byte budget is head-truncated with `[Truncated: 50.0KB limit reached]` appended.
8. The same atom, given a custom `GrepOperations` impl pointing at an SSH host, returns matches from the remote FS without any code change.

## Related Concepts

- [extension-as-scenario.md](extension-as-scenario.md) §7.1 — tool atom catalog (this design extends it)
- [pluggable-architecture.md](pluggable-architecture.md) §3.2 — Operations port pattern reused for `Grep/Find/LsOperations`
- [edit-diff.md](edit-diff.md) — sibling tool-foundation module landing in the same wave
- [tool-event-narrowing.md](tool-event-narrowing.md) — these tools each contribute a typed `ToolCallEvent`/`ToolResultEvent` variant

## Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Use `ripgrep`/`fd` if present, fall back to stdlib | Best of both: speed when available, no hard dep | Always stdlib (slow on big repos) / always require `rg`/`fd` (UX friction) |
| **Do not** auto-download binaries | Pi's `ensureTool` violates least-surprise; user owns their binaries | Auto-fetch — security and disk-quota issues |
| `pathspec` for `.gitignore` parsing | Battle-tested, handles all edge cases | Hand-roll — bugs in `**`, negation, anchored patterns |
| Per-tool `Operations` Protocol (mirrors `FileOperations`) | Same SSH-swap story works for search tools | Single `SearchOperations` superset — too coarse |
| 500-char per-line truncation in grep, 50KB total | Pi defaults; bias toward more useful output for the model | Bigger limits — context budget pressure |
| Each atom is one `.py` file ≤ 300 LoC | [§11.1](extension-as-scenario.md#111-hard-rules) hard rule | Subpackage per tool — violates contract |

## Out of Scope

- Semantic / embedding-based search.
- `grep --replace` (covered by `tool_edit`).
- Watching files for changes (`fswatch`-style notifications).
- Cross-host search aggregation.

## Open Questions

- [ ] Add `pathspec` to `pyproject.toml` deps now, or only when grep/find fallback path lands? (Recommendation: now — both Tier 1 features need it.)
- [ ] Should `tool_ls` honor `.gitignore` by default? Pi does **not**; a literal `ls` shows everything. Recommendation: keep parity — do not filter.
- [ ] Should grep return JSON instead of text for easier programmatic consumption? Defer until a real consumer asks.
