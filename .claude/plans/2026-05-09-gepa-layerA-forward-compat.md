# Plan: GEPA Layer A — MVP Forward-Compatibility Edits

**Date**: 2026-05-09
**Status**: DRAFT
**Target design**: [per-task-evolution-loop](../designs/per-task-evolution-loop.md) §6, §7, §11
**Methodological reference**: [GEPA summary §6.2 / §8 option C](../../knowledge/summary_gepa_reflective_evolution.md)

## Requirements Restatement

The MVP loop (`tool_query_traces` / `tool_eval_run` / `tool_propose_change`) is
real-LLM-validated on `format_fix`. Design commit `d02969f` ratified five
forward-compat tweaks at the doc level; the code still has the older shape.
Layer A is **interface + schema-only**, zero behavior change, all in
extensions/scenarios (no `agentm.core.*` touch). Goal: ship the contract that
Layer B (Phase 2) builds on, without breaking any existing scenario.

The five edits, mapped to design sections:

1. Activations file rename `decisions.jsonl` → `activations.jsonl` (§6, §7).
2. `target_atom: str` → `target: ChangeSpec` with MVP guard
   `kind="atom_source"` only (§6).
3. Grader contract returns `μ_f = {score, dimensions, feedback_text,
   module_feedback}` instead of `{score, rationale}` / scalar float (§3.2).
4. Reserve `.agentm/decisions/<scenario>/candidates/` directory, write a
   degenerate single-entry record alongside each activation (§7).
5. Add `.agentm/decisions/<scenario>/budget.json` slot + `max_cost_usd` cap
   wired through `tool_eval_run` (§11 MVP cost ceiling).

## Prerequisites

- MVP merged on main (commit `c52f9e8` and prior). Real-LLM validation report
  at `.claude/tasks/2026-05-08-per-task-evolution-real-llm-validation.md`.
- Design commits `3a59e9e` (4-floor gate) and `d02969f` (forward-compat doc)
  on the worktree branch.
- Confirmed scope: no `agentm.core.*` modifications. The single existing core
  hook (`AgentSessionConfig.atom_source_overrides`) is sufficient.

## Implementation Phases

Layer A items are mostly orthogonal. One day's work in two waves.

### Wave A1 (parallel-safe — schema/file edits only)

These four touch disjoint files; can be implemented concurrently.

- **A-1**: rename `decisions.jsonl` writer → `activations.jsonl`.
  Task: [task](../tasks/2026-05-09-A1-rename-activations-jsonl.md). Size: S.
- **A-3**: grader contract → `μ_f` (TypedDict) and `tool_eval_run`
  aggregates `feedback_text` / `module_feedback`.
  Task: [task](../tasks/2026-05-09-A3-grader-mu-f-contract.md). Size: M.
- **A-4**: write skeleton `candidates/<id>.json` per activation (degenerate
  single-entry pool member).
  Task: [task](../tasks/2026-05-09-A4-candidates-skeleton.md). Size: S.
- **A-5**: `budget.json` slot + `max_cost_usd` cap accumulator in
  `tool_eval_run`.
  Task: [task](../tasks/2026-05-09-A5-budget-json-slot.md). Size: S.

### Wave A2 (sequenced — depends on A-3 for the new shape)

- **A-2**: `target_atom: str` → `target: ChangeSpec`. Hard switch (no
  positional adapter — the only callers today are `format_fix` and the
  validation report; both tuner prompts get migrated).
  Task: [task](../tasks/2026-05-09-A2-changespec-signature.md). Size: M.
  Depends on **A-1** (uses the renamed activations log) and is paired with
  the migration patch for `format_fix`'s tuner prompt.

## Dependency Graph

