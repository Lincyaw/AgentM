# 2026-05-09 — Issue #76: De-scenario the compaction engine

Parent: #73. Touches review items A2, A7, A8 + B14.

## Goal

Strip three pockets of scenario-specific knowledge from the compaction kernel:

1. Hardcoded tool names `"read"/"write"/"edit"` in `core/_internal/compaction/utils.py` —
   replace with a tool-registry-based metadata lookup (`tool.metadata["file_op"]`).
2. Hardcoded English prompt bodies (`"AI coding assistant"` persona, update / turn-
   prefix summarization templates) — move to a new builtin atom
   `extensions/builtin/compaction_prompts.py`. Kernel keeps zero literal English.
3. `entry.type == "..."` string dispatch in `compaction.get_message_from_entry`
   and `harness/session_manager.build_session_context` — replaced by an
   `ENTRY_MATERIALIZERS` registry registered by atoms at install time.

## Approach

### Tool metadata

- Tool atoms (`tool_read`, `tool_write`, `tool_edit`) attach
  `metadata={"file_op": "read"|"write"|"edit"}` to the `FunctionTool` they
  register. `FunctionTool.metadata` already exists.
- `extract_file_ops_from_message` accepts an optional `tools` parameter
  (`list[Tool] | None`); for each `ToolCallBlock`, it looks up the tool by
  name and reads its `metadata.get("file_op")`. No string literal tool names
  remain in the kernel.

### Prompt externalization

- Extend `PromptTemplatesService` with two registry methods:
  `register_prompt(name, body)` and `get_prompt(name) -> str | None`. Backed
  by an in-memory dict on the default impl. Atoms register prompts at
  install time; engine code reads via the service.
- New atom `extensions/builtin/compaction_prompts.py` registers three
  prompt names:
  - `compaction.summarization_system`
  - `compaction.update_summarization`
  - `compaction.turn_prefix_summarization`
- The atom also registers the three default `EntryMaterializer`s for
  `"message"`, `"branch_summary"`, `"compaction"`. Single atom, single file —
  the issue allowed splitting but the §11 single-file contract is easier to
  uphold with one atom owning both pieces of "compaction defaults".
- `compaction.compact` and friends accept optional prompt strings as
  parameters; callers (the `llm_compaction` atom, branch summarization)
  resolve via `api.prompt_templates.get_prompt(...)` and pass them in.
- When the prompts atom is *not* installed:
  - `extract_file_ops_from_message` with no tool registry yields empty file
    ops (degrades silently).
  - `get_message_from_entry` / `build_session_context` consult the registry;
    if empty, they emit a `DiagnosticEvent` (level=warning,
    source="compaction") and fall back to no-op materialization (returns
    `None`). The harness already has a bus, so the materializer signature
    accepts an optional `EventBus`-like thunk, but we keep it side-effect
    free and surface the diagnostic from the harness call site.

### EntryMaterializer registry

- `core/abi/session.py` adds:
  ```python
  class EntryMaterializer(Protocol):
      def to_message(self, entry: SessionEntry) -> AgentMessage | None: ...
  ENTRY_MATERIALIZERS: dict[str, EntryMaterializer] = {}
  ```
- Mutable module-level dict — atoms populate at install time. Built-in
  defaults registered by `compaction_prompts` atom.
- `core/_internal/compaction/compaction.get_message_from_entry` consults
  the registry. No `entry.type == "..."` string equality remains in core.
- `harness/session_manager.build_session_context` consults the same
  registry.

## Verification gate

- `uv run ruff check src/`
- `uv run mypy src/`
- `uv run pytest --tb=short`
- `grep -n '"read"\|"write"\|"edit"' src/agentm/core/_internal/compaction/utils.py` → empty
- `grep -rn "AI coding assistant\|coding assistant" src/agentm/core/` → empty
- `grep -rn 'entry.type == "' src/agentm/core/ src/agentm/harness/` → empty
- `python -c "from agentm.extensions.validate import ..."` accepts the new atom

## Tests

New unit tests covering:

- Compaction with `compaction_prompts` atom installed runs end-to-end against
  a stub summarizer and produces a summary that contains the registered
  template body.
- Compaction with the atom **not** installed emits a diagnostic and yields
  empty materialization (no crash).
- File-op extraction with tool registry containing `metadata={"file_op":"read"}`
  picks the path up; without it, it is empty.
