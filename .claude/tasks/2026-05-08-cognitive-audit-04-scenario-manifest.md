# Task: harness_monitor scenario manifest (V0 three-axis audit)

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-cognitive-audit-v0.md)
**Design**: [design](../designs/llmharness-cognitive-audit.md)
**Assignee**: implementer

## Objective

Author (or enrich) `<AgentM-root>/scenarios/harness_monitor/manifest.yaml`
so its system prompt encodes the §5.2 audit flow and registers
`cards_list` / `cards_get` as tools the diagnostic agent can call.

## Inputs

- Read: design §3 (three axes: backward continuity, forward
  fulfillment, content correctness — what each axis judges, examples,
  cards covered).
- Read: design §5.2 (audit flow inside the child session — 9 numbered
  steps including self-monitor, three-axis, cards consultation,
  silence gates, same-type suppression).
- Read: design §5.3 reminder format `[harness] {free text}` and §2
  authority constraints (advisor not controller, low coercion,
  default silent).
- Read: `scenarios/llmharness/project-index.yaml` REQ-011 description
  (mentions a P0 single-call manifest at `../harness_monitor/manifest.yaml`).
- Search the working tree for any pre-existing `harness_monitor`
  manifest (none found by planner; confirm before authoring fresh).
- Read one of `scenarios/plan_mode/manifest.yaml`,
  `scenarios/rca/manifest.yaml`, or
  `scenarios/trajectory_analysis/manifest.yaml` for AgentM scenario
  manifest shape (system prompt + tools + atoms list).

## Outputs

- New (or rewritten): `scenarios/harness_monitor/manifest.yaml`. Keys
  to populate (cross-check against an existing scenario manifest):
  - Scenario `name: harness_monitor`.
  - Atoms: include the standard kernel/loop atoms PLUS the
    cards-tools atom (Phase 2's `cards_list`/`cards_get` exposed via
    a small `cards_tools` atom — implementer adds a thin wrapper
    atom file under `src/llmharness/atoms/cards_tools.py` if needed,
    or registers them directly in the manifest's `tools` block per
    AgentM convention).
  - System prompt body: instructions encoding §5.2:
    1. Inputs description: receives `new_turns`, `history_events_tail`,
       `recent_alerts` (each with `downstream_reaction`).
    2. Step-by-step audit flow (self-monitor first; three-axis check
       only if not self-silenced; cards consultation when helpful;
       silence gates with concrete-reason articulation; same-type
       suppression two-stage — structural pre-filter + LLM re-drift
       judgment).
    3. Output JSON contract aligned with `Verdict` (drift,
       reminder/text, cited_cards, matched_event_ids; design §6.2
       fields).
    4. Tone convention: observational, not imperative; reminder
       format `[harness] {free text}`.
    5. NO external retrieval; trajectory-internal material only
       (axis 3).
- Modified: `scenarios/llmharness/project-index.yaml` —
  - Update REQ-011 description to reflect three-axis audit role
    (still P0, still implemented, but now an enrichment).
  - Add REQ-018 ("harness_monitor enriched with three-axis +
    self-monitor + silence gates + cards tools", `priority: P0`,
    `status: implemented`,
    `code: [../harness_monitor/manifest.yaml]`,
    `depends_on: [REQ-011, REQ-016]`).

## Acceptance Conditions

- [ ] `agentm --scenario harness_monitor` resolves the manifest with
      no error (smoke check)
- [ ] System prompt mentions all three axes by their names
      (backward continuity / forward fulfillment / content correctness)
- [ ] System prompt mandates the self-monitor step BEFORE any
      three-axis check
- [ ] System prompt mandates `[harness] ...` prefix for emitted
      reminders
- [ ] `cards_list` and `cards_get` are accessible from the
      diagnostic agent (verify via a manual run with a known small
      input)
- [ ] `validate_index.py project-index.yaml` clean

## Notes

- The diagnostic agent is itself an AgentM session — it has full
  observability via `<cwd>/.agentm/observability/<child_id>.jsonl`
  for free (design §7). No persistence code in this manifest.
- Card retrieval choices show up in the child's trajectory — they
  are the V1 training-data signal. The prompt should encourage but
  not require cards lookup.
- Do NOT prepend the 42 cards as static prompt content (design §4.4
  rejects this). Cards live as tools.
- The diagnostic agent must NOT spawn its own children (no
  recursive audit). If AgentM has a way to declare "no
  spawn_child_session for this scenario", apply it; otherwise rely
  on the prompt to forbid recursive dispatch.
