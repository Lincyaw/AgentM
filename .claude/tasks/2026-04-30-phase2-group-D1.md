# Task: Phase 2 Group D1 — Tool Atoms

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §1, §1.1, §7.1, §10b.6
**Architecture**: [pluggable-architecture.md](../designs/pluggable-architecture.md) §3.2 (acceptance scenario 2 — bash over SSH), §6 (acceptance scenario 7 — scenario composition)
**Agent**: implementer (sonnet)
**Status**: BLOCKED on Group A0 (`core/operations.py` must land first)

## Why this proves the central architectural claim

> **Scenarios are compositions of atoms.** Each tool here is a single-responsibility extension; many scenarios reuse it.

After Group D1, `tool_read` exists exactly once. `general_purpose`, `rca`, `trajectory_analysis` (Group D2) all compose it. There is no per-scenario duplication.

## Scope

**Seven** atomic tool extensions, each one Python module under `src/agentm/extensions/builtin/`. Every module exports `install(api, config)` that registers exactly one logical capability.

| Module | Tool(s) registered | Notes |
|---|---|---|
| `tool_read.py` | `read(path: str, offset: int = 0, limit: int = 2000) -> str` | `await config["file_ops"].read_file(path)`; decode utf-8 with `errors="replace"`; slice by line. Default `file_ops = LocalFileOperations()`. |
| `tool_bash.py` | `bash(cmd: str, timeout: float = 120.0) -> str` | `await config["bash_ops"].exec(cmd, cwd=api.cwd, timeout=timeout)`; format `{exit_code, stdout, stderr, timed_out}` into a string. Default `bash_ops = LocalBashOperations()`. |
| `tool_edit.py` | `edit(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str` | Read via `file_ops.read_file`, do string replace, write via `file_ops.write_file`. Fail if `old_string` not unique unless `replace_all`. |
| `tool_write.py` | `write(path: str, content: str) -> str` | `await config["file_ops"].write_file(path, content.encode("utf-8"))`. |
| `tool_hypothesis_store.py` | `add_hypothesis`, `update_hypothesis`, `list_hypotheses` | Owns an in-memory dataclass store captured in closure. Each mutation also calls `api.session.append_entry("hypothesis", asdict(hyp))`. Used by RCA scenario. |
| `tool_trajectory_loader.py` | `load_trajectory(path)`, `summarize_trajectory()`, `find_event(predicate_str)`, `compare_trajectories(a, b)` | Reads JSONL files written by `trajectory` atom. State (loaded trajectories) lives in closure. |
| `tool_submit_plan.py` | `submit_plan(plan: str)` | `api.session.append_entry("plan", {"text": plan})`; `api.events.emit("plan_submitted", {"plan": plan})`; returns `ToolResult(content=[TextContent("plan submitted")], extras={"plan_submitted": True})`. Used by plan_mode scenario. |

### Required shape

```python
def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    file_ops = config.get("file_ops") or LocalFileOperations()

    async def execute(args: dict, signal, on_update) -> ToolResult:
        data = await file_ops.read_file(args["path"])
        return ToolResult(content=[TextContent(text=data.decode("utf-8", errors="replace"))])

    api.register_tool(FunctionTool(
        name="read",
        description="Read a file from disk.",
        parameters={"type": "object", "properties": {...}, "required": ["path"]},
        execute=execute,
    ))
```

## HARD constraints

- Imports allowed: stdlib + `agentm.core.kernel.*` + `agentm.core.operations` + `agentm.harness.extension`.
- **Forbidden in tool bodies**: `subprocess`, `pathlib`, `open(`, `os.system`. Grep test in CI: `grep -nE 'subprocess|pathlib|open\(|os\.system' src/agentm/extensions/builtin/tool_*.py` must return empty.
- No `langchain*`. No `agentm.scenarios.*`, `agentm.tools.*` (legacy). No `agentm.harness.middleware`, `agentm.harness.runtime`.
- `from __future__ import annotations` at top of every file.
- Each module's docstring references the relevant `extension-as-scenario.md` §7.1 row.

## Tests

`tests/unit/extensions/builtin/<atom>/test_*.py` — one folder per atom.

### Per atom

1. **install smoke**: `AgentSession.create(extensions=[(<module>, {})], provider=fake_provider)` → session's tool list contains the expected tool name.
2. **execute**: invoke the registered tool's `execute` directly with valid args; assert the right `FileOperations`/`BashOperations` call(s) happened (use a recording fake) and the `ToolResult` content is correct.
3. **error path**: file not found / bash non-zero exit / replace string not unique — tool returns a `ToolResult(is_error=True, content=...)`, never raises.

### Cross-atom (one test in `tests/unit/extensions/builtin/test_tool_atom_composition.py`)

- Load `tool_read` + `tool_write` together; write then read; assert roundtrip.
- Load `tool_hypothesis_store` alone; add 3 hypotheses; assert `list_hypotheses` returns all 3 and `api.session.get_active_branch()` contains 3 `hypothesis` entries.

## Quality gates

```bash
uv run ruff check src/agentm/extensions/builtin/tool_*.py tests/unit/extensions/builtin/
uv run mypy src/agentm/extensions/builtin/tool_*.py
uv run pytest tests/unit/extensions/ tests/unit/kernel/ tests/unit/harness_v2/ tests/unit/llm/ tests/unit/core/operations/ -q
# Layer purity:
! grep -nE 'subprocess|pathlib|open\(|os\.system' src/agentm/extensions/builtin/tool_*.py
```

All pass.

## Reference (READ-ONLY, will be deleted in Phase 2.5)

- `src/agentm/scenarios/<name>/` — system prompt + tool wiring (for behavior parity)
- `src/agentm/tools/<name>/` — legacy tool implementations (for behavior parity only — re-implement against `FunctionTool` + Operations ports)
- `src/agentm/core/trajectory.py` — for `tool_trajectory_loader` to know the file format

## Report format (≤300 words)

1. Files created (absolute paths) — one per atom.
2. For each tool atom, the JSON schema you wrote for `parameters`.
3. Test counts per atom + cross-atom test results.
4. Layer purity grep result (must be empty).
5. Anything that surfaced a missing API surface that should feed back to `extension-as-scenario.md` §10b.
