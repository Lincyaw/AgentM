# Task: Per-Task Evolution Loop — MVP Implementation

**Date**: 2026-05-08
**Design**: [`per-task-evolution-loop.md`](../designs/per-task-evolution-loop.md)
**Branch**: `worktree-agent-a7f8dbc01c9ee5710` (fresh worktree off `main`)

## Summary of changes

Constitution layer (minimal, additive only):

- `core-manifest.yaml` — adds `.agentm/decisions/**` to `constitution.paths`.
- `src/agentm/harness/session_config.py` — adds `task_class`,
  `eval_run_id`, `eval_task_id`, `atom_source_overrides` fields to
  `AgentSessionConfig`.
- `src/agentm/harness/events.py` — `SessionReadyEvent` carries
  `task_class`, `eval_run_id`, `eval_task_id` so observability can
  populate `task_meta`.
- `src/agentm/harness/session.py` — three changes:
  1. Adopt scenario-declared `task_class` unless the caller set one.
  2. Apply `atom_source_overrides` after extension load — each override
     is staged into `<cwd>/.agentm/eval-sandbox/<session_id>/`, the
     atom's `LoadedAtom.file_path` is redirected there, and the
     existing `reload_atom` path runs (path classifies as `unmanaged` →
     no git mutation).
  3. Tear down the sandbox on `shutdown`.
- `src/agentm/harness/atom_reloader.py` — two fixes (small):
  - `list_atoms` uses `is_file()` instead of `exists()`.
  - `_activate_atom_install` now treats `agentm._scenarios.*` synthetic
    modules the same way as `_agentm_user_atom__*` (rebuild
    `sys.modules` entry from the on-disk file before running install).
    Without this, scenario-local atoms could not be reloaded.
- `src/agentm/extensions/loader.py` — new `load_scenario_with_meta`
  surfaces the manifest's top-level `task_class` and `promotion`
  fields. `load_scenario` is preserved as a thin wrapper.
- `src/agentm/extensions/builtin/observability.py` — `task_meta` now
  carries `task_class`, `eval_run_id`, `task_id` (legacy `type` /
  `difficulty` / `external_id` slots retained).

Three new tier-1 atoms in `src/agentm/extensions/builtin/`:

- `tool_query_traces.py` — filters `.agentm/observability/*.jsonl` by
  `task_class` + optional fingerprint exact-match. Returns lightweight
  summaries; agent reads full traces via the existing `read` tool.
- `tool_eval_run.py` — spawns one child `AgentSession` per
  (task × sample), applies `atom_source_overrides` to each child,
  runs the eval grader from `<eval_dir>/grader.py`, aggregates per-task
  mean/stddev + overall primary score + guard metrics + holdout score,
  writes a summary record to `.agentm/eval_runs/<eval_run_id>.jsonl`.
  Supports `samples_per_task`, `holdout_only`, `smoke`.
- `tool_propose_change.py` — required-arg validation, tier-2
  deferral (`pending_human_approval`), four-part promotion gate
  (relative threshold, statistical sanity Δ > 2σ, guard tolerance
  ±X%, decision-mode skip), then `api.reload_atom` + decision record
  appended to `.agentm/decisions/<scenario>/decisions.jsonl` (the path
  is now constitution-protected; this atom is the sole mediated
  writer).

Format-fix scenario (`contrib/scenarios/format_fix/`):

- `manifest.yaml`, `tool_normalize_json.py` (deliberately weak v1 —
  passes only task 01).
- `eval/tasks/01..08.yaml` — 8 tasks (06/07 marked `holdout: true`).
- `eval/grader.py` — deterministic deep-equal grader, tolerates
  markdown fences and surrounding commentary.
- `eval/README.md` — eval-set contract.
- `tuner/manifest.yaml`, `tuner/prompt.md` — meta-scenario stacking
  the three new atoms with promotion config (threshold 0.05, guard
  0.10, stop_after_no_improvement 3).

Tests:

- `tests/integration/test_per_task_evolution.py` — 6 fail-stop tests
  (P3 evidence rejection, tier-2 deferral, decisions write-protect at
  predicate + `tool_write` end-to-end, sandbox tree-cleanliness, full
  propose_change → reload activation flow).
- `tests/integration/test_per_task_evolution_e2e.py` — placeholder
  for real-LLM E2E (skipped without API key); intended for the parent
  agent to wire and run manually for the report.
- `pyproject.toml` — registers `slow` and `requires_api_key` markers.

## Decisions made

### §6.3 — eval execution mechanism: temp-dir source overrides (preferred)

I went with the *preferred* path from the design: a per-session
sandbox under `.agentm/eval-sandbox/<session_id>/`. Each override is
written to a sandbox file; the atom's `LoadedAtom.file_path` is
redirected there before the reload runs. Because the sandbox path is
outside any managed glob, the `GitBackedResourceWriter` classifies it
as `unmanaged` and writes without a git commit — so the source-of-truth
tree stays bit-identical (verified by the
`test_atom_source_override_leaves_tree_clean` test using `git status
--porcelain` before/after, ignoring `.agentm/` and `__pycache__`).

The shadow-worktree alternative would have required new git plumbing
in `harness.resource_writer`. The temp-dir path reuses existing
mechanism cleanly and was the smaller change.

### Decisions writes bypass `ResourceWriter`

The design says "use a direct file-append (not `ResourceWriter`) since
it's not a managed source artifact". I followed that — `.agentm/decisions/`
is constitution-protected against `tool_edit`/`tool_write` (which both
go through `ResourceWriter`), but `tool_propose_change` opens the file
directly with append mode. The mediation comes from being the sole
caller, not from an FS-layer block.

### Stop condition surfaced in prompt only

The design mandates the tuner's `stop_after_no_improvement` is in the
manifest. The atom-side enforcement of "stop after N no-improvements"
would require iteration tracking outside any single tool call — the
tuner is an LLM agent and respects the stop condition because the
prompt instructs it to. The config field is parsed and passed through
to `tool_propose_change.config['promotion']` for future use but isn't
acted on by code today.

## Test results

```
$ uv run pytest tests/integration/test_per_task_evolution.py -v
tests/integration/test_per_task_evolution.py::test_propose_change_rejects_without_evidence PASSED
tests/integration/test_per_task_evolution.py::test_tier2_activate_is_deferred PASSED
tests/integration/test_per_task_evolution.py::test_decisions_path_is_constitution PASSED
tests/integration/test_per_task_evolution.py::test_tool_write_rejects_decisions_path PASSED
tests/integration/test_per_task_evolution.py::test_atom_source_override_leaves_tree_clean PASSED
tests/integration/test_per_task_evolution.py::test_end_to_end_loop_activates_known_good_replacement PASSED
6 passed
```

Full suite: `97 passed, 14 deselected`. `mypy src/` and `ruff check src/`
both clean.

## Constitution-layer changes and rationale

Two non-trivial constitution-layer edits:

1. **`AgentSessionConfig` field additions** (`harness/session_config.py`,
   `harness/events.py`, `harness/session.py`). The design's §4.1
   demands `task_class` populated on `session.fingerprint.task_meta`,
   and §6.3 needs an API surface for sub-session source overrides.
   Both are additive (default `None`), so no backwards-compat break.
   Plumbing through `SessionReadyEvent` (rather than another channel)
   keeps the observability writer's path simple.
2. **`AtomReloader` synthetic-module handling**. Pre-existing code
   only handled `_agentm_user_atom__*` synthetics. Scenario-local
   atoms (`agentm._scenarios.*`) had the same need but no support;
   reload would fail with `ModuleNotFoundError`. This is a one-line
   `or` clause and arguably a bug fix that this task surfaced rather
   than a constitution change — it's still inside the constitution
   tree, hence flagged here.

Neither edit changes any ABI signature; both retain backwards
compatibility.

## Open issues / TODOs for the parent agent

- **Real-LLM E2E (Path H)** is a stub. The placeholder file exists at
  `tests/integration/test_per_task_evolution_e2e.py` with the right
  markers and skip logic but no actual provider wiring. Per the
  task brief, this is intentional — leave for the parent agent's
  manual run.
- **Tuner stop-after-no-improvement** is prompt-side only. If a tighter
  guarantee is wanted, the next iteration could add a per-iteration
  counter to the decision log and have `tool_propose_change` refuse
  further activations after N consecutive misses.
- **Cross-task atom transfer** — design §13 leaves this open; not
  attempted here.
- **Indexer integration** — `tool_eval_run`'s `eval_runs/*.jsonl` is a
  parallel append-only log to `.agentm/observability/`; the catalog
  indexer doesn't consume it yet. Phase-2 work.
- **`tool_query_traces` cost field** — currently records 0.0 unless
  the LLM provider emits `cost_usd` on `llm.request.end`. Most stub
  providers don't; real providers do. Acceptable for MVP but worth a
  note in the report.

## Files touched

```
core-manifest.yaml
pyproject.toml
src/agentm/harness/session_config.py
src/agentm/harness/events.py
src/agentm/harness/session.py
src/agentm/harness/atom_reloader.py
src/agentm/extensions/loader.py
src/agentm/extensions/builtin/observability.py
src/agentm/extensions/builtin/tool_query_traces.py             [new]
src/agentm/extensions/builtin/tool_eval_run.py                 [new]
src/agentm/extensions/builtin/tool_propose_change.py           [new]
contrib/scenarios/format_fix/manifest.yaml                     [new]
contrib/scenarios/format_fix/tool_normalize_json.py            [new]
contrib/scenarios/format_fix/eval/README.md                    [new]
contrib/scenarios/format_fix/eval/grader.py                    [new]
contrib/scenarios/format_fix/eval/tasks/{01..08}.yaml          [new]
contrib/scenarios/format_fix/tuner/manifest.yaml               [new]
contrib/scenarios/format_fix/tuner/prompt.md                   [new]
tests/integration/test_per_task_evolution.py                   [new]
tests/integration/test_per_task_evolution_e2e.py               [new, skipped]
.claude/designs/per-task-evolution-loop.md                     [from PR #67]
.claude/index.yaml                                             [from PR #67]
```
