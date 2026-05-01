# Plan: Pi-Mono Feature Migration — Skills, Templates, Search, Edit-Diff, Typed Tool Events

**Created**: 2026-05-01
**Designs**:
- [skills.md](../designs/skills.md)
- [prompt-templates.md](../designs/prompt-templates.md)
- [search-tools.md](../designs/search-tools.md)
- [edit-diff.md](../designs/edit-diff.md)
- [tool-event-narrowing.md](../designs/tool-event-narrowing.md)

**Architecture**: [pluggable-architecture.md](../designs/pluggable-architecture.md), [extension-as-scenario.md](../designs/extension-as-scenario.md)
**Predecessors**: Phase 2 (built-in atoms) + Phase 2.5 (legacy sweep) — landed.
**Reference codebase**: pi-mono at `/tmp/pi-analysis/pi-mono/packages/coding-agent/`.

---

## Goal

Port a curated set of pi-mono features into AgentM as new atoms and core modules — *without* importing pi's tree/branch session model or heavyweight LLM compaction, which are flagged for a separate user decision (Tier 3 below).

Three landing waves:

- **Tier 1 + Tier 1.5** — direct ports of features AgentM lacks: skills, prompt templates, search tools, `resources_discover` event hook.
- **Tier 2** — adapt-and-enhance work on existing atoms (`tool_read`/`tool_edit`/`tool_write`/`tool_bash`) plus the typed-tool-event migration.
- **Tier 3** — three open architectural questions; **no tasks scheduled**; document only.

## Why this is parallelizable

Same shape as the Phase 2 catalog work: every atom is a single `.py` file under `extensions/builtin/`, gated by `validate_builtin()` ([extension-as-scenario.md §11](../designs/extension-as-scenario.md#11-single-file-self-contained-extension-contract)). Different atoms touch different files. The shared modules (`core/text_truncate.py`, `core/path_utils.py`, `core/edit_diff.py`, `core/frontmatter.py`, `core/skills.py`, `core/prompt_templates.py`) are leaves — no atom-to-atom imports, only atoms importing core.

## Group breakdown — Tier 1 + 1.5

| Group | Artifacts | Risk | Depends on |
|---|---|---|---|
| **α — Soft resources** | `core/skills.py` · `core/prompt_templates.py` · `core/frontmatter.py` (thin wrapper around `python-frontmatter` package) · `extensions/builtin/skill_loader.py` · `extensions/builtin/prompt_templates.py` · `harness/events.py::ResourcesDiscoverEvent` · `pyproject.toml`: `uv add python-frontmatter` · §11.1 import allow-list addition for `agentm.core.{skills,prompt_templates,frontmatter}` · `agentm.harness.events` already allowed | Medium | none (Phase 2 baseline) |
| **β — Search tools** | `core/text_truncate.py` · `core/path_utils.py` · `extensions/builtin/tool_grep.py` · `extensions/builtin/tool_find.py` · `extensions/builtin/tool_ls.py` · §11.1 import allow-list addition for `agentm.core.{text_truncate,path_utils}` · `pyproject.toml`: `uv add pathspec` | Medium | none |
| **γ — Tool foundations** | `core/edit_diff.py` · §11.1 import allow-list addition for `agentm.core.edit_diff` | Low | none |

α, β, γ are **independent** and run in parallel — three implementer agents.

### Notes per group

#### Group α (soft resources)

- `uv add python-frontmatter` first. Then write `core/frontmatter.py` as a thin wrapper exposing `parse_frontmatter(text) -> tuple[dict, str]` (so the rest of the code base does not depend on the third-party API directly). Update `harness/resource_loader.py::_split_frontmatter` to delegate to the wrapper (one-line refactor).
- `skill_loader` and `prompt_templates` both subscribe to `resources_discover` and `session_ready`. The `resources_discover` event is *emitted by the consumer*, not by `AgentSession` — i.e., `skill_loader` emits it during its own discovery pass; same for `prompt_templates`. (See [skills.md](../designs/skills.md) "resources_discover event".)
- Slash-command runner ordering update in `harness/session.py::AgentSession.prompt`: when text starts with `/` and no code command matches, run `input` event handlers (which `prompt_templates` hooks) **before** falling through to the agent loop. Document this in the group α task.
- Validator allow-list (`agentm.extensions.validate`): add `agentm.core.skills`, `agentm.core.prompt_templates`, `agentm.core.frontmatter` to the allowed `agentm.core.*` imports list.

#### Group β (search tools)

- Add `pathspec` dependency in `pyproject.toml` via `uv add pathspec`.
- `core/text_truncate.py` and `core/path_utils.py` are pure modules — no I/O, no `Operations`. Stdlib only.
- The three atoms each consume `text_truncate` + `path_utils`; their `Operations` Protocols (`GrepOperations`, `FindOperations`, `LsOperations`) are local to the atom file. Default impls use stdlib + `agentm.core.operations.LocalFileOperations` where it makes sense (e.g., `tool_grep` reads file contents via `LocalFileOperations` for context-line rendering).
- Add `agentm.core.text_truncate` and `agentm.core.path_utils` to validator allow-list.
- Atom budget: each tool ≤ 300 LoC. If approaching, request a re-split before merging.

#### Group γ (tool foundations)

- `core/edit_diff.py` is pure-function, stdlib-only. Implementer writes the algorithm and stops there — `tool_edit` upgrade is deferred to Tier 2 (next wave).
- Add `agentm.core.edit_diff` to validator allow-list.

### Composition gate (mandatory before Tier 2 starts)

After α + β + γ all land, one reviewer agent writes `tests/integration/pi_mono_tier1_composition.py`:

- Loads `{tool_read, tool_bash, tool_edit, tool_write, tool_grep, tool_find, tool_ls, skill_loader, prompt_templates, system_prompt}` together.
- Sets up a temp-dir project with: one `.gitignore` excluding `dist/`, one `SKILL.md` under `.agentm/skills/refactor/`, one prompt template at `.agentm/prompts/explain.md`.
- Drives a session via a fake provider that issues each tool exactly once.
- Asserts:
  - `<available_skills>` block present in system prompt (verify via `BeforeAgentStartEvent` capture).
  - `/explain $@` user-text is expanded by `prompt_templates`.
  - `grep` results omit anything under `dist/`.
  - `find` returns POSIX paths sorted lexicographically.
  - `ls` returns sorted entries with `/` suffix.
  - All tool result events are captured (no double-block from policy atoms — note: `permission` and `dedup` are *not* loaded in this test; that is a separate Phase 2 composition test).
  - Session-tree (linear) integrity via `SessionManager.get_messages()`.

Plus the standard cross-checks:

- `uv run pytest`
- `uv run ruff check src/agentm/`
- `uv run mypy src/agentm/`
- Layer purity grep: `grep -rE 'subprocess|pathlib\.Path|open\(' src/agentm/extensions/builtin/tool_{grep,find,ls}.py` returns only allow-listed forms (i.e., subprocess used only via `asyncio.create_subprocess_exec`; `pathlib` only inside `Operations` impl bodies).
- Atom-to-atom import grep: `grep -rE 'from agentm\.extensions\.builtin' src/agentm/extensions/builtin/` returns empty.
- No `langchain` imports anywhere under `src/agentm/`.

---

## Group breakdown — Tier 2 (after Tier 1 lands)

| Group | Artifacts | Risk | Depends on |
|---|---|---|---|
| **δ — Read/Edit/Write/Bash enhancements** | Upgrade `tool_read.py` (image MIME detection, line-range slicing), `tool_edit.py` (use `core/edit_diff`, `replace_all` arg, diff preview), `tool_write.py` (atomic write via temp + rename through `FileOperations`), `tool_bash.py` (more rigorous timeout/abort: kill-tree on cancel, `signal.alarm` on POSIX where applicable, `psutil`-free pure asyncio) | Medium | Tier 1 γ (`core/edit_diff.py`) |
| **ε — Tool-event narrowing** | Per-tool `*ToolCallEvent` / `*ToolResultEvent` subclasses + `is_*_tool_call/result` helpers in `core/kernel/events.py` (or new `core/kernel/tool_events.py`); coordinated migration of all atoms that branch on `event.tool_name`; tests updated. **Single coordinated PR** (Phase 2.0b-style) — landing piecemeal causes long-lived split states. | High | Tier 1 β (so `grep`/`find`/`ls` events are migrated in the same PR), δ (so `EditToolDetails` shape is final) |

### Notes — δ

- Each atom retains its single-file shape (≤ 300 LoC). If the bash upgrade pushes over budget, split the timeout/abort logic into `core/operations.py`'s `LocalBashOperations` (it already lives there).
- Atomic write contract: `LocalFileOperations.write_file` writes to `<path>.tmp.<pid>.<rand>` and renames. Add unit test for crash safety: write with kill mid-flight leaves no partial file at the destination.
- Image read: `tool_read` returns `ImageContent(mime_type, data_base64)` for `.png`/`.jpg`/`.gif`/`.webp`. Detection by extension + magic-byte check (stdlib `imghdr` is deprecated in 3.12; use `mimetypes.guess_type` + a small magic-byte sniff function).

### Notes — ε

- Migration scope per [tool-event-narrowing.md](../designs/tool-event-narrowing.md). All atoms that currently branch on `event.tool_name` are updated in one PR.
- Tests under `tests/unit/extensions/builtin/*/` are updated to construct typed events where they currently construct generic ones. This is a mechanical sweep.
- The `validate_builtin()` validator gains a check: `MANIFEST.registers` referencing `tool:<name>` for a known tool name should match a corresponding subclass (informational only — fails as warning, not error).

### Composition gate (Tier 2)

A second reviewer pass — extends `pi_mono_tier1_composition.py` (now becomes `pi_mono_full_composition.py`):

- Adds `permission`, `dedup`, `file_mutation_queue`, `trajectory`, `tool_result_budget` to the loadout.
- Asserts that:
  - All policy atoms successfully use `is_*_tool_call` style narrowing (verified by importing helpers in their module bodies — caught by `validate_builtin()`).
  - Bash timeout actually fires and produces a typed `BashToolResultEvent` with `details.timed_out=True`.
  - Edit on a CRLF file preserves line endings (tests `core/edit_diff.py` end-to-end through `tool_edit`).
  - Trajectory recorder receives typed events and serializes them losslessly through JSONL round-trip.

Plus standard `pytest` / `ruff` / `mypy` / layer-purity gates.

---

## Tier 3 — open architectural questions (DECISION REQUIRED, NO TASKS SCHEDULED)

The following are deliberately not planned. They require user buy-in before scheduling. Each entry documents the gap and what's at stake. Per the user's "full-fidelity port" steer, the recommended action when scheduling is to **mirror pi-mono's structure** rather than redesign.

### T3.1 Tree/branch session model

**Pi**: `SessionManager` (1425 LoC) supports forking at any entry, navigating the tree, branch summarization. Multi-agent orchestration (sub-agent spawns, parallel exploration) benefits significantly.
**AgentM today**: linear (single active branch). `SessionManager.fork_at` exists in the Protocol but `JsonlSessionManager` raises `NotImplementedError`.
**Cost**: medium-large refactor of `harness/session_manager.py` and any atom that calls `append`/`get_messages`. Touches the on-disk JSONL format (parent-id chains become non-linear).
**Recommendation if approved**: stage as a separate Phase 3 plan with its own designs (`session-tree.md`, `branch-summarization.md`). Mirror pi's `SessionManager` shape and JSONL entry-id chaining.

**Decision required from user**: schedule, defer, or drop?

### T3.2 Heavyweight LLM compaction

**Pi**: `compaction/compaction.ts` (839 LoC) + `branch-summarization.ts` (355 LoC) — replaces a chunk of history with an LLM-generated summary; supports custom instructions and integrates with branch summary for tree mode.
**AgentM today**: `extensions/builtin/micro_compact.py` — sliding window, rule-based.
**Coexistence**: pi's variant lands as a *separate atom* (`extensions/builtin/llm_compact.py`); both can be loaded in different scenarios.
**Cost**: ≈ one new atom + one design doc. Lower than T3.1.
**Recommendation if approved**: schedule after T3.1's outcome is known (because pi-mono integrates compaction with branch summarization; if we never have branches, the integration shrinks).

**Decision required from user**: schedule, defer, or drop?

### T3.3 Provider registration / model registry expansion

**Pi**: `model-resolver.ts`, OAuth flow, provider catalog.
**AgentM today**: `llm/anthropic.py` only; provider is a single `(module, config)` tuple on `AgentSessionConfig`.
**Cost**: model-resolver alone is small; OAuth is a sizable surface.
**Recommendation**: out of scope for this migration. Re-raise when adding a second provider.

**Decision required from user**: defer to a separate plan; no action this wave.

---

## Tier 1.5 attachment — `resources_discover` event

Already folded into Group α (`harness/events.py::ResourcesDiscoverEvent` and per-consumer emission). Not a separate group — small enough to ride along.

---

## Verification gates per group

Identical for every group:

```bash
uv run pytest tests/unit/extensions/builtin/<atom>/ -v
uv run pytest tests/unit/core/<module>/ -v        # for core/* modules
uv run ruff check src/agentm/<changed-files>
uv run mypy src/agentm/<changed-files>

# Layer purity
grep -rE 'from agentm\.extensions\.builtin' src/agentm/extensions/builtin/   # must be empty
grep -rE 'import langchain' src/agentm/                                        # must be empty
grep -rE 'from agentm\.harness\.session ' src/agentm/extensions/              # must be empty (atoms never import the orchestrator)

# Validator gate
uv run pytest tests/unit/extensions/test_extension_contract.py -v
```

---

## Pacing recommendation

1. **Tier 1 parallel batch**: α + β + γ — 3 implementer agents in parallel.
2. **Tier 1 reviewer composition test** — single reviewer.
3. **Tier 2 wave 1** (δ): one implementer per atom subgroup (read/edit/write/bash can be split if useful).
4. **Tier 2 wave 2** (ε): single implementer for the coordinated narrowing migration.
5. **Tier 2 reviewer composition test** — single reviewer extends Tier 1's test.
6. **Tier 3 design discussions** — user decides scheduling for T3.1, T3.2; T3.3 stays deferred.

---

## Deferred (NOT in this plan)

- Tier 3 items — explicitly require user decision before scheduling.
- pi-mono's interactive TUI / mode-layer features (`modes/interactive/*`).
- pi-mono's `model-resolver` and OAuth flow.
- pi-mono's MCP integration (intentional non-goal per AgentM philosophy memo).

---

## Gaps in the migration scope I noticed (informational)

While reading pi-mono I noticed three additional features the user might consider — none are in this plan, all flagged for visibility:

1. **`session-cwd.ts`** — pi tracks per-session CWD changes (the `cd` bash command updates the session's effective cwd). AgentM `BashOperations.exec` accepts `cwd` per call but no session-level tracking exists. Useful when an agent does `cd subdir && ls`. ~50 LoC.
2. **`exec.ts` layered abort** — pi has soft-timeout-warning → hard-kill → kill-tree. AgentM's bash atom currently has only a single timeout. Mentioned in Tier 2 δ; calling it out as its own story in case the user wants it sooner.
3. **`event-bus.ts` 33-line minimal port** — the AgentM `EventBus` already covers this; mentioning only because pi's spec is a useful reference if questions arise about handler ordering.

End of plan.
