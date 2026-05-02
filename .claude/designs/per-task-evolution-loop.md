# Design: Per-Task Evolution Loop

**Status**: PROPOSED
**Created**: 2026-05-02
**Builds on**: [evolution-substrate.md](evolution-substrate.md), [self-modifiable-architecture.md](self-modifiable-architecture.md), [observability.md](observability.md), [extension-as-scenario.md](extension-as-scenario.md)
**Sister docs**: evolution-substrate (catalog substrate), self-modifiable-architecture (transactional reload)

---

## 1. First Principle

> **The optimization target is a task class, not a user. A production agent runs the task; a separate tuner agent (a meta-scenario) reads observability traces, proposes mutations to atoms / prompts / scenario composition, validates against a pinned eval set, and activates winners.**

This is the operational shape of self-modification we want — not "agent gradually improves itself across diverse user requests" (statistical noise dominates), but "for task class X, find the configuration that performs best on a fixed input distribution".

The reframe has three load-bearing consequences:

1. **JSONL is the source of truth.** Each production run already writes a complete OTel-flavored trace at `.agentm/observability/<trace_id>.jsonl`. The tuner's input is N such files, filtered by `(task_class, fingerprint)`. The catalog substrate (`.agentm/catalog/atoms/<hash>/metrics.jsonl`) becomes a *secondary index* — useful for cross-scenario browsing, not load-bearing for decisions. (See [evolution-substrate.md §3](evolution-substrate.md) for catalog layout; this doc demotes it.)

2. **No predefined stats API.** `compare()` / `find_best()` were Phase 2 of evolution-substrate.md. Per-task-class evolution does not need them — the tuner is itself an LLM agent that reads raw traces and decides what matters for *this* task class. "What counts as success on RCA" is policy that lives in the tuner's system prompt, not in a stats library. As models improve, analysis improves automatically; a fixed `compare()` cannot benefit from that.

3. **Confounding is killed by an eval set, not by sample size.** Production traces have variable user inputs — a `compare(v1, v2)` over them confounds atom effect with task difficulty. A pinned eval set (representative tasks per scenario) gives a fixed input distribution; running both versions on it isolates atom effect. This is the missing primitive — without it, no honest comparison is possible at the volumes a single user generates.

---

## 2. Layer Position

The loop sits entirely in the autonomy layer; the constitution is unchanged.

```
constitution    core + reload primitive + ResourceWriter + observability sink
   ↑
autonomy        atoms + production scenarios
                eval/             ← NEW: pinned task suite per scenario
                tuner/            ← NEW: meta-scenario for evolution
                tool_query_traces, tool_eval_run, tool_propose_change ← NEW atoms
   ↑
discovery       fs scan + reload
```

Eval set, tuner scenario, and the three new tool atoms are autonomy-layer additions. They reuse existing constitution mechanism — `reload_atom` for atom mutation, observability JSONL for inputs, ResourceWriter for git-backed write-with-rollback.

---

## 3. The Eval Set

The missing primitive. Every scenario that wants per-task evolution declares one:

```
scenarios/<name>/eval/
├── README.md                  # purpose: what this suite is meant to measure
├── tasks/
│   ├── 01_simple_disk_full.yaml
│   ├── 02_memory_leak.yaml
│   └── ...                    # 10–50 representative tasks
└── grader.md                  # rubric used by an LLM grader
                               # (or grader.py for deterministic checks)
```

### 3.1 Task YAML schema

```yaml
# scenarios/rca/eval/tasks/01_simple_disk_full.yaml
id: rca_disk_full
description: User reports disk-full alert; root cause is log-rotation regression.
task_class: rca
input:
  user_message: |
    The /var partition on app-server-3 is full. The on-call alert fired at
    14:32 UTC. Find the root cause.
  fixtures:
    - data/eval-fixtures/disk_full/observability.parquet
    - data/eval-fixtures/disk_full/syslog.parquet
expected:
  root_cause_keywords: ["log rotation", "logrotate", "/var/log/app"]
  must_call_tools: ["sql"]                    # at least one query
  forbidden_tools: ["bash"]                   # plan-mode constraint check
budget:
  max_turns: 25
  max_cost_usd: 0.50
```

`expected` is intentionally loose — keyword-presence and tool-presence checks. Strict golden trajectories are too brittle for LLM-driven scenarios; the grader does the semantic work.

### 3.2 Grader

Each task is graded after the production scenario terminates. Two modes:

- **Rubric mode** (default): `grader.md` is a prompt template; an LLM call given `(task input, agent's final output, full trajectory)` returns `{ score: 0–1, dimensions: {...}, rationale: str }`. Result is appended as a `task.grade` event to the eval-run trace.
- **Programmatic mode** (opt-in): `grader.py` exports `grade(task_yaml, trace) -> GradeResult`. Used when there is a deterministic check (e.g. trajectory_judger). Falls back to rubric on raise.

### 3.3 Why the eval lives next to the scenario

Eval is part of the scenario's contract — when scenario YAML or its atoms change, the eval may also need to change. Keeping them in one tree means `git log scenarios/rca/` tells the full story. Same locality argument as `tests/` next to source.

---

## 4. Trace Filtering & Tuner Input

### 4.1 task_class on session.fingerprint

`session.fingerprint` (per [evolution-substrate.md §4](evolution-substrate.md)) already has a `task_meta` slot but nothing populates it. Two writers fill it:

- **Production**: scenario manifest declares `task_class: <name>`. Observability emits it on every session's `session.fingerprint` record.
- **Eval-run**: `tool_eval_run` writes `task_class` + `eval_run_id` + per-task `task_id` so eval traces are joinable back to the suite they came from.

```jsonc
{
  "kind": "session.fingerprint",
  "trace_id": "...",
  "fingerprint": { "atoms": {...}, "scenario": "rca@..." },
  "task_meta": {
    "task_class": "rca",
    "eval_run_id": null,        // null for production; <id> for eval
    "task_id": null,            // null for production; <task.id> for eval
    "external_id": null
  }
}
```

### 4.2 tool_query_traces

Tier-1 atom registering one tool:

```python
tool_query_traces(
    task_class: str,
    fingerprint: dict | None = None,           # exact-match; None = any
    n: int = 50,                                # most-recent N
    include_events: list[str] | None = None,    # default: agent_end, tool_call,
                                                #          tool_result, llm.request.end
) -> list[TraceSummary]
```

Returns lightweight summaries `(trace_id, timestamp, terminal stop_reason, total_cost, total_turns, eval_run_id?)`. The agent then reads specific traces via the existing `read` tool — query is the index, read is the fetch. Pattern matches how skills work today (compact index + on-demand body).

### 4.3 Why not give the tuner the catalog?

The catalog's `metrics.jsonl` aggregates with a fixed schema (`task.completion_rate`, `tokens_per_task`, …). For per-task tuning, the relevant signal often is *not* in that schema — it is "which tool-error pattern dominated v1's failures", "did v2 reach the conclusion in fewer turns but with a higher refusal rate", etc. The tuner reads raw events and decides; the catalog's pre-aggregation throws away the structure it needs.

The catalog stays useful for *cross-class* browsing ("show me every version of `tool_read` across all scenarios") — out of scope for this loop.

---

## 5. The Tuner Scenario

A meta-scenario lives next to its target:

```
scenarios/rca/
├── manifest.yaml            # production scenario (existing)
├── eval/                    # § 3
└── tuner/
    ├── manifest.yaml        # the meta-scenario
    └── prompt.md            # tuner's system prompt
```

### 5.1 tuner/manifest.yaml

```yaml
name: rca_tuner
description: Evolves the rca scenario by reading traces and proposing atom changes.
extensions:
  - module: agentm.extensions.builtin.tool_read
  - module: agentm.extensions.builtin.tool_grep
  - module: agentm.extensions.builtin.tool_query_traces
  - module: agentm.extensions.builtin.tool_eval_run
    config:
      target_scenario: rca
      eval_dir: scenarios/rca/eval
  - module: agentm.extensions.builtin.tool_propose_change
    config:
      target_scenario: rca
      promotion:
        threshold_relative: 0.05    # +5% on primary metric required
        guard_tolerance: 0.10       # ±10% on guards
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: !include prompt.md
```

### 5.2 prompt.md (sketch)

The system prompt encodes the tuner's mission AND the success criterion for the task class:

```
You are the RCA scenario tuner. Your job is to make the rca scenario better
at finding root causes given parquet observability data.

Quality signal for this task class:
  primary:   root_cause_hit_rate (from grader)
  secondary: cost_per_task, turns_to_completion
  guard:     refusal_rate, tool_error_rate

Loop:
  1. tool_query_traces(task_class="rca", n=50) → recent production traces
  2. Read 5–10 representative traces; identify failure patterns
  3. Form hypothesis: "atom X is causing Y" (e.g. "tool_read truncates parquet
     at the wrong boundary, losing the error window")
  4. tool_eval_run(fingerprint=current) → baseline eval scores
  5. Propose a mutation: read the atom's source via get_source_at, design a
     focused change, call tool_propose_change(...) with the new source
  6. tool_eval_run(fingerprint=proposed) → proposed eval scores
  7. If proposed dominates baseline on primary AND no guard regression →
     ACTIVATE. Else → record reason, try a different angle next iteration.

You are not making one-shot edits. Each loop iteration changes ONE atom; do
not propose multi-atom mutations. If a proposed change underperforms, record
why and try a different angle next iteration.
```

### 5.3 Why a scenario, not a hardcoded mode

Three reasons:

- **Mechanism reuse**: every primitive (tool registration, system prompt, sub-agent dispatch, observability) already works for scenarios.
- **Per-task customization**: each task class has its own quality signal. A hardcoded mode would force a single fitness-function shape; a prompt is free-form.
- **Tuner is itself versioned**: `scenarios/rca/tuner/` sits in the autonomy layer, so tuner improvements go through the same git-backed write path as everything else. The tuner can even tune its own prompt — meta-meta if you want it.

---

## 6. tool_propose_change Contract

```python
tool_propose_change(
    target_atom: str,                    # e.g. "tool_read"
    new_source: str,                     # full source for the atom file
    rationale: str,                      # human/agent-readable why
    eval_run_baseline: str,              # eval_run_id under current fingerprint
    eval_run_proposed: str,              # eval_run_id under proposed fingerprint
    decision: Literal["activate", "rollback", "exploratory"],
) -> ProposeChangeResult
```

Calling semantics by `decision`:

- `activate`: applies the change via existing `api.reload_atom`. Requires `eval_run_proposed` to dominate `eval_run_baseline` on the scenario's primary metric AND no guard regression. Atomic — rollback on install failure (reuses existing transactional reload).
- `rollback`: re-activates the previous version (the baseline fingerprint's atom hash). Used when an earlier `activate` regressed in subsequent production traffic.
- `exploratory`: applies the change but flags the decision record `exploratory: true`. For when eval is inconclusive but the tuner wants production signal. Production guard metrics get extra-strict regression watching for exploratory atoms.

All three write to `.agentm/decisions/<scenario>/decisions.jsonl` (one file per target scenario, easier to grep than per-atom-version files).

### 6.1 Why eval_run_baseline AND eval_run_proposed are required

Forces the tuner to actually run eval before proposing. Without both, a proposal cannot be filed — structural enforcement of "evidence-driven, not error-driven". This is the equivalent of B's "comparison_id required" but adapted to the eval-set world.

### 6.2 Tier-2 atoms still gate

If `target_atom.tier == 2` (`permission`, `cost_budget`, `tool_filter`, `llm_compaction`), `decision="activate"` is rejected with `pending_human_approval`. The decision record is still written; activation is deferred. For per-task tuning, tier-2 mutations need the user (or a future automated policy) — there is no scenario-level approval channel.

### 6.3 Eval execution mechanism

`tool_eval_run` must execute the eval suite under a *proposed* fingerprint without permanently mutating the working tree. Two implementation strategies, decided at the Plan stage:

- **Sub-session with source overrides**: extend `AgentSession.create` to accept `atom_source_overrides: dict[str, bytes]`; eval spawns one child session per task with those overrides applied. No filesystem mutation. Cleanest, but adds API surface.
- **Shadow worktree**: the new source is committed to a transient git ref (`refs/agentm/eval/<run_id>`); eval session is configured with that ref as its source root. Reuses `git-backed-versioning.md` machinery. No new API surface, but more git state to clean up.

Both are viable; the choice belongs in the implementation Plan, not this design.

---

## 7. Decision Records

`.agentm/decisions/<scenario>/decisions.jsonl`:

```jsonc
{"at":"2026-05-10T...","kind":"activate","scenario":"rca","atom":"tool_read",
 "from_sha":"abc123","to_sha":"def456",
 "evidence":{"baseline_run":"er_001","proposed_run":"er_002",
             "primary_metric":"root_cause_hit_rate",
             "baseline":0.62,"proposed":0.78},
 "rationale":"v2 reads parquet via row-group boundaries instead of byte offset; ...",
 "by":"rca_tuner@<tuner_session_id>"}
```

Append-only. The `.agentm/decisions/` directory is constitution-write-protected against agent's `tool_edit`/`tool_write` (covered by `is_constitution_path` after a one-line manifest update). Only `tool_propose_change` writes here, mediated.

---

## 8. Promotion Policy

Minimal MVP policy:

- **activate** requires: primary metric improvement ≥ `threshold_relative` AND every guard metric within `± guard_tolerance` of baseline.
- Both knobs declared per-scenario in `tuner/manifest.yaml::promotion`. Defaults `threshold_relative=0.05`, `guard_tolerance=0.10`.
- `decision="exploratory"` overrides the gate but tags the decision record so production regression watching is stricter.

Policy is evaluated by `tool_propose_change` itself (reads the eval results, applies the threshold). Putting it in the atom keeps it auditable — the activation gate is one function, callable from tests, greppable.

Post-MVP: Goodhart guard (auto-rollback if production traffic shows guard regression after activation), multi-objective Pareto frontier reasoning, etc. Defer until activations actually happen regularly.

---

## 9. Worked Example: RCA Tuning Loop

1. User runs `agentm tune rca --iterations 3`. Harness loads `scenarios/rca/tuner/manifest.yaml`.
2. Tuner agent calls `tool_query_traces(task_class="rca", n=50)`. Receives 50 trace summaries.
3. Tuner reads 8 traces, notices: 60% failed at parquet read step (`truncated_at_boundary` error). Hypothesizes `tool_read` truncates parquet wrongly.
4. Tuner calls `tool_eval_run(fingerprint=current)`. Result: `er_baseline`, `root_cause_hit_rate=0.62`.
5. Tuner reads `tool_read.py` source via `read`, designs new version using row-group boundaries.
6. Tuner calls `tool_eval_run(fingerprint={tool_read: <proposed_sha>})`. Result: `er_proposed`, `root_cause_hit_rate=0.78`.
7. Tuner calls `tool_propose_change(target_atom="tool_read", new_source=..., rationale=..., eval_run_baseline="er_baseline", eval_run_proposed="er_proposed", decision="activate")`.
8. `tool_propose_change` checks threshold (0.78 / 0.62 = 1.26 ≥ 1.05 ✓), guard metrics (refusal_rate Δ within ±10% ✓), tier (1 ✓). Calls `api.reload_atom`. Decision record written.
9. Next production RCA session loads new `tool_read`. Observability records both old and new fingerprint runs in JSONL — future tuning iterations have richer data.

---

## 10. Acceptance Scenarios

| # | Scenario | Expected |
|---|----------|----------|
| P1 | Tuner runs against a scenario with no `eval/` directory | `tool_eval_run` rejects with clear error; tuner cannot propose |
| P2 | `task_class` on production session.fingerprint is null | Production session warns at startup; trace still recorded but `tool_query_traces` filtered by `task_class` will not see it |
| P3 | `tool_propose_change` called without `eval_run_proposed` | Rejected; "evidence missing" |
| P4 | Eval run on a fingerprint that requires a constitution-layer change | Rejected by ResourceWriter; tuner sees the rejection and skips |
| P5 | Activate succeeds; subsequent production traffic shows primary regression beyond guard | (Phase 2) auto-rollback; today the tuner sees it next iteration via `tool_query_traces` |
| P6 | Two tuners (RCA + plan_mode) propose simultaneous mutations to a *shared* atom (`tool_read`) | Writes serialize through ResourceWriter's git layer; the second proposal sees the first's change in its baseline fingerprint and must re-eval |
| P7 | Eval run hits `max_turns` budget on a task | Task graded as failed at that task; eval result still comparable across versions |
| P8 | Tuner tries to mutate `permission` (tier-2) | Decision recorded as `pending_human_approval`; reload not applied |
| P9 | Production trace has `task_class="rca"`; `tool_query_traces` filter returns it | Confirmed (round-trip test) |
| P10 | After 10 activated mutations, `git log scenarios/rca/` shows full mutation history with rationales as commit messages | Confirmed (ResourceWriter's commit-with-rationale path) |

---

## 11. Phasing

### MVP

- Eval set scaffolding: directory layout, task YAML schema, rubric grader.
- `task_class` populated on `session.fingerprint` (production + eval).
- Three new tier-1 atoms: `tool_query_traces`, `tool_eval_run`, `tool_propose_change`.
- One worked tuner: `scenarios/rca/tuner/`.
- Decision records under `.agentm/decisions/`.
- Threshold-based promotion policy.

### Phase 2

- Production-traffic guard regression watching → auto-rollback.
- Programmatic graders (replacing the rubric LLM grader for deterministic checks).
- Cross-task transfer: when `tool_read` evolves under RCA, surface the change as a candidate for `plan_mode`'s own eval before activation.
- Catalog `metrics.jsonl` gains schema fields the tuner has shown it actually needs (data-driven, not pre-defined).

### Out of scope (not Phase 2 either)

- Multi-objective Pareto reasoning (single primary metric is enough until activations happen regularly).
- Replay cache (eval set obviates it; replay value reappears only at huge scale).
- Automated tier-2 approval (security policy, deferred indefinitely).

---

## 12. Relationship to Existing Designs

| Design | Relationship |
|---|---|
| [evolution-substrate.md](evolution-substrate.md) | This doc **demotes** §6 (`compare`/`find_best`) to optional — the tuner does it via LLM analysis on raw JSONL. Catalog directory remains useful as a cross-scenario browsing index but is not on the decision path. §7 (statistical hard problems) is replaced by the eval-set's controlled-input approach. |
| [self-modifiable-architecture.md](self-modifiable-architecture.md) | Unchanged — `reload_atom` / `freeze_current` / tier system / transactional rollback are all reused. `tool_propose_change` is a thin policy wrapper over `reload_atom`. |
| [observability.md](observability.md) | `task_class` becomes a populated field on `session.fingerprint`. No new event types. |
| [extension-as-scenario.md](extension-as-scenario.md) | The tuner is a normal scenario (meta only by convention). `eval/` is a new directory adjacent to `manifest.yaml`. Three new tier-1 atoms register normally. |
| [git-backed-versioning.md](git-backed-versioning.md) | Activation = `reload_atom` = ResourceWriter commit. `git log scenarios/rca/` IS the activation history; `decisions.jsonl` is the structured mirror. |

---

## 13. Open Questions

- **Cross-task atom transfer**: when the RCA tuner evolves `tool_read`, should `plan_mode` automatically inherit it, eval it, and decide independently? MVP says no (each scenario manages its own atom versions via fingerprint). Phase 2 may relax.
- **Tuner concurrency**: two tuners on different scenarios both want to mutate a shared atom. ResourceWriter serializes writes, but the eval results are computed on different baselines — the second tuner's "baseline" already includes the first's change. Does this need explicit locking? MVP says no (the race is observable in `decisions.jsonl`; fix lazily).
- **Eval set drift**: eval tasks themselves may need to evolve as the user's real workload evolves. There is no anti-Goodhart on the eval set itself. Out of scope; user's responsibility.
- **Cost of eval runs**: an eval suite of 50 tasks × 2 fingerprints per tuning iteration × several iterations per cycle is expensive. Quantify before MVP ships. May need a "smoke" eval (3–5 tasks) for fast gating, with the full suite reserved for the activation decision.
- **Eval execution mechanism** (§6.3): sub-session source overrides vs. shadow worktree. Plan-stage decision.
