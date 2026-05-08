# Plan: llmharness Cognitive Audit V0 — soft NL advisor for T3+T4 drift

**Date**: 2026-05-08
**Status**: DRAFT
**Target design**: [llmharness-cognitive-audit](../designs/llmharness-cognitive-audit.md)

## Requirements Restatement

V0 ships an agent-agnostic cognitive audit that supervises a running AgentM
main agent for semantic-level (T3+T4) drift. The audit is a **soft NL
advisor** (never a controller): on every `TurnEndEvent` of the main session,
a child AgentM session runs the diagnostic scenario, performs a three-axis
check (backward continuity, forward fulfillment, hyperedge content
correctness) over trajectory-internal material, and — only when it can
articulate a concrete reason — writes a single `[harness] {free text}`
reminder that the next `BeforeAgentStartEvent` injects into the system
prompt.

V0 scope is **AgentM-on-AgentM only**. The Claude Code adapter remains on
the existing P0 rule-based detector. Schema changes must stay additive so
rca-autorl's import path keeps working.

Concrete deliverables:

1. Schema: opt-in `Verdict.cited_cards: list[str]` and
   `Verdict.downstream_reaction: str | None`, both default-empty / None,
   round-trippable JSON, no breakage of existing P0 consumers
   (worker.py, agentm_bridge.py, store.py).
2. `cards.py`: new module that loads YAML cards from
   `references/papers/cards/<class>/*.yaml`, exposes `cards_list()` and
   `cards_get(card_id)`, and carries an in-code `axis_hint: dict[str, int|None]`
   curated mapping (NOT a YAML schema field).
3. Adapter restructure: `claude_code.py` → `adapters/claude_code.py`
   (preserve behavior); new `adapters/agentm.py` registering
   `TurnEndEvent` → spawn child + persist verdict, and
   `BeforeAgentStartEvent` → inject pending reminder via `system` mutation.
4. `scenarios/harness_monitor/manifest.yaml`: create on-disk scenario
   whose system prompt encodes the §5.2 audit flow (self-monitor first,
   three-axis check, silence gates, same-type suppression, reminder
   format) and registers `cards_list` / `cards_get` as tools.
5. `project-index.yaml` updated for every code change (per llmharness
   CLAUDE.md MANDATORY rule), with new REQ entries (status:
   `implemented`; tests are deferred to a later pass).

## Prerequisites

- Design doc accepted at `.claude/designs/llmharness-cognitive-audit.md`
  (revised 2026-05-08).
- Existing P0 modules in `scenarios/llmharness/src/llmharness/` — schema,
  store, worker, agentm_bridge, claude_code, summarizer, detector — all
  `status: implemented`.
- AgentM `ExtensionAPI` exposing `api.on(channel)`, `api.spawn_child_session`,
  and the events `TurnEndEvent` and `BeforeAgentStartEvent` with mutable
  `system` field per `designs/sub-agent-lifecycle.md` and
  `designs/observability.md`.
- AgentM scenario loader supporting `agentm --scenario harness_monitor`
  via `<AgentM-root>/scenarios/harness_monitor/manifest.yaml` (per
  `scenarios/` layout in CLAUDE.md).

## Implementation Phases

### Phase 1: Schema additions (additive, non-breaking)

- Add `cited_cards: list[str] = field(default_factory=list)` and
  `downstream_reaction: str | None = None` to `Verdict`.
- Both fields opt-in. Existing constructions
  (`Verdict(drift=False, ...)` from `detector.py`, `agentm_bridge.py`)
  must continue to work unchanged.
- Verify `asdict(verdict)` JSON shape stays backward-compatible:
  consumers that don't read the new fields keep working; consumers that
  serialize a Verdict can round-trip with no key mismatch.
- Task: [task](../tasks/2026-05-08-cognitive-audit-01-schema.md)
- Size: S

### Phase 2: cards.py module

- New `src/llmharness/cards.py` with:
  - `CardSummary` and `CardFull` dataclasses (frozen).
  - `cards_list() -> list[CardSummary]` (id, name, axis_hint,
    one_line_mechanism — derived from `defect.mechanism` first sentence).
  - `cards_get(card_id) -> CardFull` (full YAML payload).
  - `_AXIS_HINT: dict[str, int | None]` curated module-level constant
    mapping each known card_id (AFC-XXXX) to axis 1/2/3 or `None`/`"?"`
    for "unclassified". Coverage: every card present in
    `references/papers/cards/<class>/*.yaml` listed by AFC ID.
  - YAML loader walks `references/papers/cards/<class>/*.yaml`
    relative to package install (compute via `importlib.resources` or
    a `_CARDS_ROOT = Path(__file__).resolve().parents[3] / "references" / "papers" / "cards"`).
- No I/O on import (lazy load on first `cards_list` call; cache result).
- Task: [task](../tasks/2026-05-08-cognitive-audit-02-cards-module.md)
- Size: M

### Phase 3: adapters/ subpackage + AgentM adapter

- Create `src/llmharness/adapters/__init__.py`.
- Move `src/llmharness/claude_code.py` → `src/llmharness/adapters/claude_code.py`.
  Preserve behavior; add a re-export shim at the old path **only if**
  rca-autorl imports `llmharness.claude_code` directly (verify via grep
  of rca-autorl docs / requirement REQ-003 description). If no external
  import, no shim — pure move.
- New `src/llmharness/adapters/agentm.py` implementing the §4.5 sketch:
  - `install(api, config)` registering two handlers.
  - `TurnEndEvent` handler:
    - Collects new turns from session (via API surface available at
      handler time — implementer to confirm exact accessor).
    - Appends to `HarnessStore.inbox`.
    - Calls `api.spawn_child_session(scenario="harness_monitor",
      input={...})` with `new_turns`, `history_events_tail`,
      `recent_alerts` (each carrying `downstream_reaction`).
    - Parses child verdict; persists via `store.append_verdict`.
    - If `verdict.drift`, writes pending reminder via existing
      `store.write_pending_reminder` (or equivalent).
  - `BeforeAgentStartEvent` handler:
    - Pops pending reminder, mutates `event.system` in place to append
      `\n\n[harness] {pending.text}`.
- Manifest module-level `MANIFEST: ExtensionManifest` per
  `designs/extension-as-scenario.md` §11 single-file contract.
- Task: [task](../tasks/2026-05-08-cognitive-audit-03-agentm-adapter.md)
- Size: M

### Phase 4: harness_monitor scenario manifest

- Create `<AgentM-root>/scenarios/harness_monitor/manifest.yaml`. (Note:
  the design refers to it as already "shipping P0", but no manifest
  exists on disk — implementer to confirm whether agentm_bridge.py
  ever resolved it from a different path; if a P0 manifest exists in
  another tree, this phase **enriches** it rather than creating it.)
- System prompt encodes:
  1. Inputs contract: receives `new_turns`, `history_events_tail`,
     `recent_alerts` (with each alert's `downstream_reaction`).
  2. Self-monitor first step: read `recent_alerts`/`downstream_reaction`,
     judge whether prior reminders were heeded; if not → emit silent
     verdict and exit (no three-axis check).
  3. Three-axis check (axis 1 backward continuity, axis 2 forward
     fulfillment, axis 3 hyperedge content correctness) over
     trajectory-internal material only — no external retrieval.
  4. Silence gate (route B): emit only if able to articulate a
     concrete reason; card citation optional but encouraged.
  5. Same-type suppression: structural pre-filter (axis or
     `cited_cards` overlap) → LLM judgment over `recent_alerts` for
     duplicate vs re-drift.
  6. Reminder format: `[harness] {free text}` half-structured, free body.
  7. Output JSON contract aligned with `Verdict` (drift,
     reminder/text, cited_cards, matched_event_ids, optional
     downstream_reaction set by next firing).
- Tools registered: `cards_list`, `cards_get` (atom-form Python
  callables backed by Phase 2's module).
- Task: [task](../tasks/2026-05-08-cognitive-audit-04-scenario-manifest.md)
- Size: M

### Phase 5 (formerly 6): Integration + index.yaml registration

- Run `uv run ruff check src`, `uv run ruff format src`,
  `uv run mypy src/llmharness` (strict),
  `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`.
  (Tests deferred to a later pass — no `pytest` gate this round.)
- Run an end-to-end manual smoke per `bash scripts/tick_worker.sh`
  with `LLMHARNESS_PROVIDER=agentm` to confirm the AgentM adapter path
  spawns a child session and emits the expected trajectory shape.
- Update top-level `.claude/index.yaml`: append plan + task list to the
  existing `cognitive_audit` concept entry (which already has design
  registered).
- Task: [task](../tasks/2026-05-08-cognitive-audit-06-integration.md)
- Size: S

## Dependency Graph

- Phase 2 depends on Phase 1 (cards.py is independent of schema, but
  the adapter in Phase 3 wants both — safe to run 1 and 2 in parallel
  if two implementers are available).
- Phase 3 depends on Phase 1 (uses new Verdict fields) and Phase 2
  (adapter doesn't import cards.py directly, but the scenario in
  Phase 4 does — so Phase 4 transitively needs Phase 2).
- Phase 4 depends on Phase 2 (registers cards_* as tools) and Phase 3
  (adapter is what spawns the child session that runs this scenario).
- Phase 5 (integration) depends on Phases 1–4 — gating commands
  (ruff / mypy / validate_index / manual smoke) require all code
  + project-index updates to be in.

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Schema additions break rca-autorl serializer | HIGH | Field defaults are empty list / None so legacy consumers ignoring them stay correct; manual JSON round-trip check during Phase 1; regression lock-in test deferred to a later testing pass. |
| AgentM adapter API surface assumptions wrong (e.g. event accessor for "new turns") | MEDIUM | Phase 3 task flags "implementer to confirm exact accessor"; manual smoke in Phase 5 is the first end-to-end check this round. |
| harness_monitor manifest shape unknown — design says it exists but no on-disk file found in current tree | MEDIUM | Phase 4 task notes "create or enrich"; implementer must first grep for any P0 manifest under another path before authoring fresh. |
| `_AXIS_HINT` curation drifts from card YAMLs as new cards land | LOW | Phase 2 acceptance asserts every list result has a hint entry at implementation time; CI lint deferred to a later pass. |
| Reminder fatigue (V0 audit too noisy) — design's named failure mode | LOW for V0 acceptance | Manifest's silence-gate prompt is the defense; measurement deferred to V1 evaluation harness. |

## Acceptance Criteria

- [ ] `uv run mypy src/llmharness` clean (strict)
- [ ] `uv run ruff check src` clean
- [ ] `python .../validate_index.py project-index.yaml` reports 0 violations
- [ ] `project-index.yaml` has new requirements REQ-015..017 (schema,
      cards, AgentM adapter) plus REQ-018 (harness_monitor enriched
      scenario); affected entries have correct `code` paths;
      `status: implemented` for now (test gating deferred)
- [ ] `.claude/index.yaml` `cognitive_audit` entry lists this plan and
      its task files
- [ ] Manual smoke (`tick_worker.sh` with `LLMHARNESS_PROVIDER=agentm`)
      shows a child session firing per `TurnEndEvent` and a
      `[harness] ...` line appearing in the next prompt's `system`
- [ ] Reviewer confirms: schema is additive, no rca-autorl break;
      adapter contract matches design §4.5; harness_monitor manifest
      encodes §5.2 self-monitor + three-axis + silence gate +
      same-type suppression
- [ ] Existing P0 worker path (rule-based detector via Claude Code
      adapter) unchanged — REQ-006 smoke confirmable manually
