# Task: Phase 2 Group A — Policy Gate Extensions

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §7
**Agent**: implementer (sonnet)
**Status**: READY

## Scope

Five extensions, each one Python module under `src/agentm/extensions/builtin/`. All use the same shape: subscribe to `tool_call` / `tool_result` events on the `ExtensionAPI`. They're grouped because they're independent and stylistically similar — one agent can produce the whole group consistently.

| File | Extension | Behavior |
|---|---|---|
| `permission.py` | `permission` | `on("tool_call")` returns `{"block": True, "reason": ...}` for tools not on the allowlist. Config: `{"allow": ["read", "grep"], "deny": ["bash"]}`; `deny` wins ties. If neither set: pass through (no-op). |
| `tool_filter.py` | `tool_filter` | Filter tools at registration time: removes named tools from the registered list before the loop sees them. Config: `{"allow": [...]}` or `{"deny": [...]}`. Implemented by subscribing to `agent_start` and slicing the tools list — OR by mutating the session-tools list directly in `install()` (preferred — runs once). |
| `dedup.py` | `dedup` | Skip recently-repeated tool calls. Track `(tool_name, json.dumps(args, sort_keys=True))` for the last N calls (config: `window=10`). On match: `on("tool_call")` returns `{"block": True, "reason": "duplicate of recent call"}`. Reset on `agent_start`. |
| `cost_budget.py` | `cost_budget` | Track token cost per session. Subscribe `before_send_to_llm` to estimate input cost (use a simple `len(json.dumps(messages)) // 4` heuristic since we don't have a tokenizer; document this). Subscribe `turn_end` to read `message.usage` and accumulate. **Overflow mechanism (locked, design §10b.8)**: emit a custom `cost_budget_exceeded` event with payload `{used: float, limit: float, currency: str}`. `AgentSession.prompt` subscribes to this channel and, on first emission, terminates with `agent_end(stop_reason="budget")`. Do NOT raise exceptions across handler boundaries. Pricing table per provider in `_PRICING: dict[str, tuple[float, float]]` mapping `provider → (input_per_mtok, output_per_mtok)`. |
| `tool_result_budget.py` | `tool_result_budget` | Truncate huge tool outputs. `on("tool_result")` inspects content text length; if > `max_chars` (default 50000), returns a replacement `ToolResult` with content truncated and a marker note appended. Don't drop image content. |

## Per-extension `install` shape

```python
def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    ...  # all registrations
```

No async needed for any of these. Sync `install` returns `None`.

## Tests

Layout: `tests/unit/extensions/builtin/<name>/test_*.py`. Each extension gets:

1. A unit test of its handler logic in isolation (build a `ToolCallEvent` / `ToolResultEvent` / etc. directly, invoke the registered handler, assert return value).
2. An integration test using `AgentSession.create` with a fake StreamFn (reuse / adapt `tests/unit/harness_v2/_fixtures/fake_provider.py` pattern). Set up the extension, fire a prompt, verify the policy fired correctly.

For `cost_budget`: add a test verifying that exceeding the limit emits `cost_budget_exceeded` with the right payload AND that an `AgentSession` whose prompt subscribes to the event terminates with `agent_end(stop_reason="budget")`. The wiring of the session subscription itself goes into `AgentSession.prompt`; coordinate with the `cost_budget_exceeded` channel name.

## Layout

```
src/agentm/extensions/__init__.py            # empty docstring module
src/agentm/extensions/builtin/__init__.py    # empty docstring module
src/agentm/extensions/builtin/permission.py
src/agentm/extensions/builtin/tool_filter.py
src/agentm/extensions/builtin/dedup.py
src/agentm/extensions/builtin/cost_budget.py
src/agentm/extensions/builtin/tool_result_budget.py

tests/unit/extensions/__init__.py
tests/unit/extensions/builtin/__init__.py
tests/unit/extensions/builtin/permission/test_permission.py
tests/unit/extensions/builtin/tool_filter/test_tool_filter.py
tests/unit/extensions/builtin/dedup/test_dedup.py
tests/unit/extensions/builtin/cost_budget/test_cost_budget.py
tests/unit/extensions/builtin/tool_result_budget/test_tool_result_budget.py
```

(Or a flatter layout if simpler — judgement call. Just keep tests grouped per extension so failures are easy to diagnose.)

## HARD constraints

- Imports allowed: stdlib + `agentm.core.kernel.*` + `agentm.harness.{extension,events,session_manager,resource_loader,session}`.
- No `agentm.harness.middleware`, `agentm.harness.runtime`, `agentm.harness.scenario`, `agentm.harness.loops`, `agentm.harness.permission` (the legacy one), etc.
- No `langchain*`.
- No `agentm.scenarios.*`, `agentm.builder`, `agentm.agents.*`.
- `from __future__ import annotations` at top of every file.
- Each module's top-level docstring references the relevant `extension-as-scenario.md` row from §7.

## Quality gates

```bash
uv run ruff check src/agentm/extensions/ tests/unit/extensions/
uv run mypy src/agentm/extensions/builtin/
uv run pytest tests/unit/extensions/ tests/unit/kernel/ tests/unit/harness_v2/ tests/unit/llm/ -q
```

All must pass. Do NOT break existing 60 tests.

## Reference for behavior parity

Legacy implementations (READ-ONLY for behavior reference; do NOT import):
- `src/agentm/harness/permission.py`
- `src/agentm/harness/tool_filter.py`
- `src/agentm/harness/cost_budget.py`
- `src/agentm/harness/tool_result_budget.py`
- (no legacy `dedup.py` — search for `tool_dedup` in tools/ if present)

Read them to understand intended behavior; re-implement against the new ExtensionAPI shape.

## Report format (≤300 words)

1. Files added (absolute paths).
2. Test counts per extension; total green/red.
3. ruff / mypy clean: yes/no.
4. Confirmation that `cost_budget` uses the locked event-based mechanism (no exceptions across handlers).
5. Any deviations.
6. One sentence per extension: how a user activates it via `AgentSessionConfig.extensions`.
