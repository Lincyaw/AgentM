# Task: AgentM adapter (TurnEnd trigger + BeforeAgentStart inject)

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-cognitive-audit-v0.md)
**Design**: [design](../designs/llmharness-cognitive-audit.md)
**Assignee**: implementer

## Objective

Restructure `claude_code.py` into `adapters/claude_code.py` (preserving
behavior) and create `adapters/agentm.py` realizing the V0 audit hookup
described in design §4.5.

## Inputs

- Read: design §4 (architecture, layers, adapter interfaces) and §4.5
  (V0 realization sketch).
- Read: design §5 (trigger and audit flow inputs).
- Read: `scenarios/llmharness/src/llmharness/claude_code.py` (current
  P0 adapter) — must move with no behavior change.
- Read: `scenarios/llmharness/src/llmharness/store.py` —
  `HarnessStore.append_inbox`, `append_events`, `read_events`,
  `recent_alerts` (or equivalents). If `append_verdict` /
  `recent_alerts` don't exist yet, sketch minimal additions in this
  task; full implementation can chain into Phase 6 if larger work
  needed.
- Read: `designs/extension-as-scenario.md#11` for the
  `MANIFEST: ExtensionManifest + install(api, config)` single-file
  contract.
- Read: `designs/observability.md` and `designs/agent-loop.md` for the
  exact `TurnEndEvent` / `BeforeAgentStartEvent` channel constants
  and the mutability contract on `BeforeAgentStartEvent.system`.
- Read: `designs/sub-agent-lifecycle.md` for `api.spawn_child_session`
  signature.
- Search rca-autorl docs / external imports for any direct
  `from llmharness.claude_code import ...` callers; if none, drop the
  re-export shim.

## Outputs

- New: `scenarios/llmharness/src/llmharness/adapters/__init__.py`
  (empty or re-exports both adapters).
- Moved: `claude_code.py` → `adapters/claude_code.py` (with import
  paths updated within the package; no behavior change).
- New: `scenarios/llmharness/src/llmharness/adapters/agentm.py`:
  - Module-level `MANIFEST: ExtensionManifest` (name, api_version,
    affects, tier — tier 1, single-file contract).
  - `def install(api: ExtensionAPI, config: dict[str, Any]) -> None:`
    body following design §4.5 sketch.
  - `TurnEndEvent` handler: collect new turns; append to inbox;
    spawn child session with input dict
    `{"new_turns", "history_events_tail", "recent_alerts"}`; await
    verdict; persist via `store.append_verdict`; if `verdict.drift`
    write pending reminder via existing store API.
  - `BeforeAgentStartEvent` handler: pop pending reminder; if
    present, mutate `event.system` in place (append, with leading
    `\n\n[harness] `).
  - Helper to extract `recent_alerts` (last N verdicts with their
    `downstream_reaction`, default N=5).
  - Tail config keys for now: `K_history_events: int = 50`,
    `N_recent_alerts: int = 5` (sane defaults; document in
    docstring).
- Modified: `scenarios/llmharness/project-index.yaml` —
  - REQ-003 `code` updated to `src/llmharness/adapters/claude_code.py`
  - Add REQ-017 ("AgentM adapter — TurnEnd trigger + BeforeAgentStart
    inject", `priority: P0`, `status: implemented`,
    `code: [src/llmharness/adapters/agentm.py]`,
    `depends_on: [REQ-001, REQ-002, REQ-011, REQ-015]`).

## Acceptance Conditions

- [ ] `claude_code.py` move is behavior-preserving — REQ-006 smoke
      test still passes
- [ ] `adapters/agentm.py` registers handlers via `api.on(...)` for
      `TurnEndEvent.CHANNEL` and `BeforeAgentStartEvent.CHANNEL`
- [ ] `MANIFEST` declares the atom per single-file contract (passes
      AgentM's §11 validator)
- [ ] On `TurnEndEvent`, child session spawns with scenario name
      `harness_monitor` and the three input keys
- [ ] On `BeforeAgentStartEvent` with a pending reminder,
      `event.system` is mutated in place (appended, never replaced);
      reminder is consumed (popped, not just peeked)
- [ ] On `BeforeAgentStartEvent` with no pending reminder,
      `event.system` is unchanged
- [ ] `uv run mypy src/llmharness/adapters` strict-clean
- [ ] `uv run ruff check src/llmharness` clean

## Notes

- The exact `ExtensionAPI` accessor for "the turns of the session at
  TurnEnd time" needs confirmation — design §4.5 names a placeholder
  `collect_turns_from_session`. The implementer should grep the agentm
  source for the canonical accessor (`api.session_turns()`,
  `event.turns`, etc) and use whichever is the documented surface.
- Reminder injection format MUST be exactly `\n\n[harness] {text}`
  (design §5.3). The leading double newline is intentional for system
  prompt readability.
- Do NOT use `api.send_user_message` — design §4.5 explicitly chose
  `system` mutation as quieter and more advisor-shaped.
- Do NOT add throttling logic in V0 — design §5.1 mandates trigger on
  every TurnEndEvent. The diagnostic agent itself is responsible for
  staying silent.
