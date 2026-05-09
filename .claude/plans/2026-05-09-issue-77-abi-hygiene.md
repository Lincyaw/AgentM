# Plan: Issue #77 — ABI hygiene cleanup

Date: 2026-05-09
Issue: Lincyaw/AgentM#77
Branch: `issue-77-abi-hygiene`

Four independent cleanups to `core/abi/` and `core/_internal/`:

## A6 — Move `FunctionTool` out of `core/abi`

`FunctionTool` is a concrete adapter; `core/abi/*` is meant to be
Protocol/dataclass-of-data only.

- Extract the dataclass from `src/agentm/core/abi/tool.py` (lines 137–168) to
  a new module `src/agentm/core/_internal/tools.py`.
- Re-export `FunctionTool` from `agentm.core.abi.__init__` and
  `agentm.harness.__init__` so call sites that use either entry point keep
  working without a shim.
- Update direct imports of `agentm.core.abi.tool` to import from
  `agentm.core.abi` instead.

## A9 — Typed `ToolErrorEvent`

Replace the synthesized English error strings in `core/abi/loop.py` with a
typed event:

```python
@dataclass(slots=True, frozen=True)
class ToolErrorEvent(Event):
    CHANNEL = "tool_error"
    kind: Literal["execution_failed", "unknown_tool", "blocked"]
    tool_name: str
    reason: str
    exception: BaseException | None = None
    result: ToolResult  # mutable container — atoms fill in content
```

Loop emits `ToolErrorEvent` (carrying a freshly-constructed empty
`ToolResult(is_error=True)`); a default builtin atom
(`tool_error_messages`) subscribes and writes the human-readable text into
`event.result.content`. Today's exact strings ship as the default so there
is no behavior change for users who run with the default scenario.

## A14 — Drop `details`/`extras` aliasing from `ToolResult`

- Remove the `__init__` shuffling, the `details` property/setter, and the
  `init=False` flag.
- `ToolResult` becomes a normal `@dataclass(slots=True)` with
  `content`/`is_error`/`extras` as canonical fields.
- Update every `.details` and `details=` reader/constructor call site to
  the canonical `extras` name (no compatibility shim).

## A15 — Promote magic numbers to settings

- `_MAX_NAME_LENGTH=64` and `_MAX_DESCRIPTION_LENGTH=1024` in
  `core/_internal/skills.py` → kwargs on `load_skills(...)` with current
  defaults; thread them through `_load_skills_from_dir` and
  `_parse_skill_file`.
- `_TOOL_RESULT_MAX_CHARS=2000` in `core/_internal/compaction/utils.py` →
  field on `CompactionSettings` with default 2000. `serialize_conversation`
  takes the value as a kwarg.

## Verification gate

```bash
uv run ruff check src/
uv run mypy src/
uv run pytest --tb=short
grep -rn "Tool execution error\|Unknown tool:\|Tool call blocked" src/agentm/core/   # empty
grep -rn '\.details\b' src/agentm/   # no ToolResult-related access
uv run python -c "from agentm.extensions.validate import validate_builtin; \
    issues = [i for i in validate_builtin() if i.module_path.endswith('tool_error_messages')]; \
    assert issues == [], issues"
```
