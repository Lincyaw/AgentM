# Task: Phase 2 Group B — Context-Shaping Extensions

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §7
**Agent**: implementer (sonnet)
**Status**: READY (depends on Phase 2.0 — `BeforeSendToLlmEvent`, `BeforeCompactEvent`, `AfterCompactEvent` already defined)

## Scope

Five **atomic** context-shaping extensions. Each is a single-responsibility module. The `plan_mode` scenario is now expressed in YAML (Group D2) composing `tool_submit_plan` (Group D1) + `system_prompt` + `permission` — it is NOT a code module here.

| File | Atom | Behavior |
|---|---|---|
| `turn_reminder.py` | `turn_reminder` | (renamed from `system_reminder`) Inject a reminder string into the system prompt every N turns. Subscribe `before_agent_start`; check `len(session.get_messages())` to count turns; return `{"system": original + "\n\n" + reminder}` when due. Config: `{"reminder": str, "every_n_turns": int}`. |
| `system_prompt.py` | `system_prompt` | (NEW atom) Subscribe `before_agent_start`; return `{"system": config["prompt"] + "\n\n" + (event.system or "")}`. The single most-reused atom — used by every scenario. Config: `{"prompt": str}`. |
| `file_mutation_queue.py` | `file_mutation_queue` | Serialize concurrent edits across `edit` / `write` tools. Implemented as a **tool wrapper**: subscribe `agent_start` (after all scenarios have registered their tools), look up `edit`/`write` (or any name in `config["tools"]`) in `api.tools`, and replace each with a wrapping `Tool` whose `execute` acquires an `asyncio.Lock` keyed by the absolute path argument. **Locked load-order rule (design §10b.4)**: extensions are processed in declaration order; `file_mutation_queue` MUST appear AFTER scenarios that register `edit`/`write`. The extension fails fast (raises `ExtensionLoadError`) on `agent_start` if it cannot find the named tools. Config: `{"tools": ["edit", "write"]}`. |
| `micro_compact.py` | `micro_compact` | Compact context when usage approaches the model's context window. Subscribe `before_send_to_llm`; if estimated token count > `config["threshold_pct"]` (default 0.85) of `model.context_window`: emit `BeforeCompactEvent(messages=..., reason="auto_overflow")` on `api.events`, perform default compaction (keep last `config["keep_last"]` messages; summarize earlier into one synthetic message), append a `compaction` entry via `api.session.append_entry("compaction", details)` (see §10b.7 — `ReadonlySession.append_entry` returns the new id; store it in details as parent for the next entry). Emit `AfterCompactEvent` after. Replace `event.messages` in place so the kernel sends compacted context. Config: `{"threshold_pct": 0.85, "keep_last": 8}`. |
| `trajectory.py` | `trajectory` | Record every kernel + harness event to a list, expose for replay/analysis. Subscribe to all major channels (`agent_start`, `agent_end`, `turn_start`, `turn_end`, `context`, `before_send_to_llm`, `tool_call`, `tool_result`, `before_compact`, `after_compact`, `child_session_start`, `child_session_end`, `cost_budget_exceeded`, `plan_submitted`). Append to an in-memory list with a timestamp + channel name. Persist on `agent_end` to a JSONL file at `config["path"]` (default `./trajectory.jsonl`). Use `dataclasses.asdict` with `default=str` fallback. Config: `{"path": str, "channels": [...] | None}` (None = all). |

## Locked decision: expand ReadonlySession

Per `extension-as-scenario.md §10b.7`, this group MUST add a single new method to `ReadonlySession` in `src/agentm/harness/extension.py`:

```python
def append_entry(self, type: str, payload: Any, parent_id: str | None = None) -> str:
    """Append a custom entry to the active branch. Returns the new entry id."""
```

Implementation delegates to the underlying `SessionManager.append`. Update `_ReadonlySessionImpl` accordingly. Add a test at `tests/unit/harness_v2/test_readonly_session_append_entry.py` covering: (a) returned id is non-empty, (b) entry shows up in `get_active_branch()`, (c) parent_id linkage works for chained entries.

## Tests

Per-extension tests in `tests/unit/extensions/builtin/<name>/test_*.py`:

- `turn_reminder`: a session with 5 messages and `every_n_turns=3` → reminder appears on turns 3, 6, 9. Verify by inspecting the system prompt seen by the StreamFn (fake provider records the system arg).
- `system_prompt`: with `config={"prompt": "X"}`, the StreamFn sees a system arg starting with `"X\n\n"`. Loading two `system_prompt` atoms in sequence stacks them in declaration order.
- `file_mutation_queue`: register two fake tools `edit` and `write` that both target path `/tmp/x`; fire concurrent calls; verify they execute serially (counter + `asyncio.sleep(0.01)` interleaving detection). Also test the fast-fail path: load `file_mutation_queue` BEFORE any scenario registers `edit`/`write` → expect `ExtensionLoadError` at `agent_start`.
- `micro_compact`: build a long message list whose estimated tokens exceeds 85% of a small `context_window=1000`; verify `BeforeCompactEvent` fires, a `compaction` entry is appended (returned id is non-empty), and the StreamFn sees a shorter messages list.
- `trajectory`: run a full session smoke; assert the JSONL file contains one record per fired event, with the right channel names.

## Layout

```
src/agentm/extensions/builtin/turn_reminder.py
src/agentm/extensions/builtin/system_prompt.py
src/agentm/extensions/builtin/file_mutation_queue.py
src/agentm/extensions/builtin/micro_compact.py
src/agentm/extensions/builtin/trajectory.py

tests/unit/extensions/builtin/turn_reminder/test_turn_reminder.py
tests/unit/extensions/builtin/system_prompt/test_system_prompt.py
tests/unit/extensions/builtin/file_mutation_queue/test_file_mutation_queue.py
tests/unit/extensions/builtin/micro_compact/test_micro_compact.py
tests/unit/extensions/builtin/trajectory/test_trajectory.py
```

## HARD constraints

Same as Group A. No legacy harness imports. No langchain. `from __future__ import annotations` everywhere. Module docstrings reference §7.

## Quality gates

```bash
uv run ruff check src/agentm/extensions/builtin/ tests/unit/extensions/builtin/ src/agentm/harness/extension.py
uv run mypy src/agentm/extensions/builtin/{turn_reminder,system_prompt,file_mutation_queue,micro_compact,trajectory}.py src/agentm/harness/extension.py
uv run pytest tests/unit/extensions/ tests/unit/kernel/ tests/unit/harness_v2/ tests/unit/llm/ -q
```

`ReadonlySession.append_entry` is a Protocol surface change — `tests/unit/harness_v2/` must stay green.

## Reference

Legacy (READ-ONLY, will be deleted in Phase 2.5): `src/agentm/harness/{system_reminder,micro_compact}.py`, `src/agentm/core/trajectory.py`, `src/agentm/harness/permission.py` (for plan-mode behavior parity). Do not import.

## Report format (≤300 words)

1. Files added.
2. Test counts.
3. The exact `ReadonlySession.append_entry` signature you shipped + diff snippet.
4. Confirmation that `file_mutation_queue` fast-fails when load order is wrong (test passes).
5. Note: plan-mode is now a Group D2 scenario YAML composing `tool_submit_plan` (D1) + `system_prompt` (this group) + `permission` (Group A). No code lives here.
6. Deviations.
