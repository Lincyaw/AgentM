# Task: Phase 2.0b — Pre-parallel Gate (load-bearing kernel/harness contracts)

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §10b
**Agent**: implementer (sonnet) → reviewer (opus)
**Status**: COMPLETED 2026-04-30 (landed in Phase 1 working tree)

## Why this task exists

The Phase 1 reviewer pass surfaced four cross-cutting contracts that **multiple
Phase 2 groups depend on**. If they shipped piecemeal inside Group A / B / C / D1
they would either (a) collide on `harness/session.py` and `harness/extension.py`
during parallel work, or (b) leave a parallel group with a broken dependency
that lands first. They are batched here as a serial gate run before the parallel
batch (A0 → 0b → A∥B∥C∥D1 → D2 → cleanup).

## Deliverables

### 0b.1 — `ReadonlySession.append_entry`

Move `append_entry` from `ExtensionAPI` (where it was a no-op stub appending
to a never-drained list) onto `ReadonlySession`. Signature:

```python
def append_entry(
    self,
    type: str,
    payload: Any,
    parent_id: str | None = None,
) -> str:
    """Append a custom entry to the active branch and return its id.
    parent_id defaults to the current active leaf."""
```

Implementation in `harness/session.py::_SessionView` delegates to the
underlying `SessionManager.append`. Required by Group D1 (`tool_hypothesis_store`,
`tool_submit_plan`) and Group B (`micro_compact`, `agent_memory`).

### 0b.2 — Drain `_pending_user_messages` at top of `prompt`

`api.send_user_message(content)` was queuing into a list nothing read.
`AgentSession.prompt` now drains the queue (FIFO) before appending the
caller's message, so queued items act as turn-prefix context. Required by
Group C (`sub_agent.inject_instruction`).

### 0b.3 — Declare missing event dataclasses

In `harness/events.py`:

- `CostBudgetExceededEvent(used: float, limit: float, currency: str = "usd")`
- `PlanSubmittedEvent(plan_id: str, plan_text: str)`
- `SessionReadyEvent(cwd, session_id, tool_names, command_names, model)`

`BeforeCompactEvent` / `AfterCompactEvent` / `ChildSessionStartEvent` /
`ChildSessionEndEvent` were already declared in Phase 1.

Required so Groups A (cost_budget), D1 (tool_submit_plan), trajectory (Group B)
agree on schema instead of inventing channel payloads ad-hoc.

### 0b.4 — `cost_budget_exceeded` subscription in `AgentSession.create`

`AgentSession` subscribes once at create-time and latches an internal flag
on emission. The next `prompt` short-circuits with an `agent_end` event
carrying `stop_reason="budget"`. Pure event-bus signalling per §10b.8 — no
exceptions cross handler boundaries. The subscription is **owned by the
session, not by the cost_budget extension**, so Group A only needs to emit
the event; it does not have to edit `harness/session.py`.

### 0b.5 — `session_ready` event emitted at end of `create`

After every extension is loaded and the active provider is picked, but
before the first `prompt` runs, the session emits `session_ready` with a
`SessionReadyEvent` payload. This is the only timing point where extensions
are guaranteed to see the *final* tool list, command set, and active model.
`tool_filter` (Group A) and similar post-install scrubbers hook here instead
of trying to time their work inside `install`.

### 0b.6 — `prompt()` helper extraction

To keep `prompt()` under the §4 100-line dispatcher budget once budget /
drain / slash branches all live there, extract:

- `_maybe_dispatch_command(text) -> list[AgentMessage] | None`
- `_drain_pending_user_messages() -> None`
- `_build_user_message(text, images) -> UserMessage`
- `_append_message(msg) -> SessionEntry`

`prompt()` becomes a sequenced dispatcher that calls these. No policy lives
in `prompt()` itself.

## Verification

```bash
uv run ruff check src/agentm/harness/
uv run mypy src/agentm/harness/
uv run pytest tests/unit/harness_v2/ tests/unit/kernel/ tests/unit/llm/ -q
```

All three must be clean. Existing test
`tests/unit/harness_v2/test_session_smoke.py::test_prompt_dispatches_slash_command_without_calling_stream`
must be updated to use `_api.session.append_entry` (not `_api.append_entry`)
since the method moved to `ReadonlySession`.

## HARD constraints

- **Serial gate**: Phase 2 groups A / B / C / D1 do **not** start until 0b is
  green. D2 already depends on the others; A0 is the only group that can
  start in parallel with 0b (different files).
- The new `harness/events.py` symbols become **part of the contract** Phase 2
  extensions consume; do not add channel payloads ad-hoc inside extensions.
- `append_entry` is the **only** mutation `ReadonlySession` exposes;
  fork / navigate stay inside the harness.

## Report format (≤150 words)

1. Files touched (path : line range).
2. New event dataclass list.
3. Test results: harness_v2 + kernel + llm test counts, all green.
4. Confirmation that the new prompt() body is ≤100 lines.
5. Any follow-ups discovered (file under §10b in extension-as-scenario.md).
