# Design: Per-Task Evolution Loop

**Status**: PROPOSED
**Created**: 2026-05-02
**Builds on**: [evolution-substrate.md](evolution-substrate.md), [self-modifiable-architecture.md](self-modifiable-architecture.md), [observability.md](observability.md), [extension-as-scenario.md](extension-as-scenario.md)
**Sister docs**: evolution-substrate (catalog substrate), self-modifiable-architecture (transactional reload)
**Methodological reference**: [GEPA (Agrawal et al., 2026)](https://arxiv.org/abs/2507.19457) — reflective prompt evolution with Pareto-based illumination. See [`../../knowledge/summary_gepa_reflective_evolution.md`](../../knowledge/summary_gepa_reflective_evolution.md) for the mapping. MVP is single-incumbent hill climbing; Phase 2 (§11) is the GEPA-shaped reorganization with candidate pool, reflection atom, and generalized `ChangeSpec`.

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

The loop is autonomy-layer in **policy** but requires two narrow constitution
additions — both mechanism, neither encodes scenario policy:

1. `.agentm/decisions/**` added to `is_constitution_path` (one-line manifest update) so only `tool_propose_change` can write the audit log.
2. `AgentSessionConfig.atom_source_overrides: dict[str, str]` field for §6.3 hypothesis testing (lets a sub-session run with a proposed atom source without mutating the working tree).

No new event types, no new ExtensionAPI surface beyond the three atoms.

```
constitution    core + reload primitive + ResourceWriter + observability sink
                + atom_source_overrides hook (§6.3)
                + .agentm/decisions/** write-protect
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
holdout: false                  # ← if true: scored separately, NOT in primary gate
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

`expected` is intentionally loose — keyword-presence and tool-presence checks. Strict golden trajectories are too brittle for LLM-driven scenarios; the grader does the semantic work. The `expected.*` schema is grader-consumed (not framework-fixed): `format_fix` uses `value: <dict>`; `rca` uses `root_cause_keywords: [...]`; rubric graders may use free-text. The framework only routes the dict through to `grade()`.

**Holdout discipline.** Mark 10–25% of tasks as `holdout: true`. Holdout tasks are scored separately from the primary metric and are **not** part of the activation gate (§8) — they are the structural Goodhart guard. A tuner that overfits non-holdout tasks will show a primary improvement *and* a holdout regression; the gate catches the latter without trusting the tuner's prompt.

### 3.2 Grader

Each task is graded after the production scenario terminates. The grader is a **feedback function** `μ_f` (after [GEPA](https://arxiv.org/abs/2507.19457) §3.2), not a scalar metric — it returns score *and* natural-language diagnosis so a future reflection step (§5.4) can do credit assignment without re-deriving "why" from raw traces:

```python
GradeResult = TypedDict("GradeResult", {
    "score": float,                          # in [0,1]
    "dimensions": dict[str, float],          # optional sub-scores
    "feedback_text": str,                    # ← required: why this score
    "module_feedback": dict[str, str],       # ← optional: per-atom attribution
})
```

`feedback_text` is the diagnostic signal that survives compaction into the next mutation prompt; `module_feedback` lets the reflection step round-robin through modules the grader fingered (e.g. "atom `tool_read` truncated input"). Programmatic graders fill these from compiler errors / rubric mismatches; rubric graders get them naturally from the LLM's own output.

Two modes:

- **Rubric mode** (default): `grader.md` is a prompt template; an LLM call given `(task input, agent's final output, full trajectory)` returns `GradeResult`. Result is appended as a `task.grade` event to the eval-run trace.
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
  5. Propose a mutation: read the atom's source via the `read` tool (atoms
     live at contrib/scenarios/<target>/<atom>.py), design a focused change,
     prepare the new source as a string
  6. tool_eval_run(eval_dir=..., atom_source_overrides={"tool_read": <new_src>})
     → proposed eval scores
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
- **Tuner is itself versioned**: `scenarios/rca/tuner/` sits in the autonomy layer, so tuner improvements go through the same git-backed write path as everything else. The tuner can even tune its own prompt — meta-meta if you want it. ([GEPA](https://arxiv.org/abs/2507.19457) explicitly notes this path: "search-tree nodes can be extended to the code describing the *whole* AI system" — Phase 2 generalizes `target` from atom-only to any scenario file (§6 ChangeSpec, §11), making meta-tuning a special case of the general mechanism rather than a separate construct.)

---

## 6. tool_propose_change Contract

```python
class ChangeSpec(TypedDict):
    kind: Literal["atom_source",          # MVP: only this kind is accepted
                  "system_prompt",        # Phase 2: scenario system_prompt text
                  "manifest_extensions",  # Phase 2: add/remove atoms in extension list
                  "manifest_field",       # Phase 2: edit other manifest scalar fields
                  "scenario_compose"]     # Phase 2: structural changes (control flow)
    path: str          # target file relative to contrib/scenarios/<target_scenario>/
    new_content: str   # full replacement OR structured patch (see §6.5)
    target_atom: str | None  # name (only for kind="atom_source"), else null

tool_propose_change(
    target: ChangeSpec,                  # see above
    rationale: str,                      # human/agent-readable why
    eval_run_baseline: str,              # eval_run_id under current fingerprint
    eval_run_proposed: str,              # eval_run_id under proposed fingerprint
    decision: Literal["activate", "rollback", "exploratory"],
) -> ProposeChangeResult
```

**MVP scope**: only `kind="atom_source"` is accepted; other kinds reserved and rejected with `not_yet_implemented`. The signature is forward-compatible — see §11 Phase 2 for the rest.

Calling semantics by `decision`:

- `activate`: applies the change via existing `api.reload_atom` (in-session) or `ResourceWriter` + git commit (cross-session, §6.4). Requires `eval_run_proposed` to pass the §8 four-floor gate against `eval_run_baseline`. Atomic — rollback on install failure (reuses existing transactional reload).
- `rollback`: re-activates the previous version (the baseline fingerprint's atom hash). Used when an earlier `activate` regressed in subsequent production traffic.
- `exploratory`: applies the change but flags the activation record `exploratory: true`. For when eval is inconclusive but the tuner wants production signal. Production guard metrics get extra-strict regression watching for exploratory activations.

All three append to **`.agentm/decisions/<scenario>/activations.jsonl`** (renamed from `decisions.jsonl`: this file records *what is live in production*, distinct from the candidate pool tree introduced in §11 Phase 2 which records *the search frontier*). One file per target scenario, easier to grep than per-atom-version files.

### 6.1 Why eval_run_baseline AND eval_run_proposed are required

Forces the tuner to actually run eval before proposing. Without both, a proposal cannot be filed — structural enforcement of "evidence-driven, not error-driven". This is the equivalent of B's "comparison_id required" but adapted to the eval-set world.

### 6.2 Tier-2 atoms still gate

If `target_atom.tier == 2` (`permission`, `cost_budget`, `tool_filter`, `llm_compaction`), `decision="activate"` is rejected with `pending_human_approval`. The decision record is still written; activation is deferred. For per-task tuning, tier-2 mutations need the user (or a future automated policy) — there is no scenario-level approval channel.

### 6.3 Eval execution mechanism

`tool_eval_run` must execute the eval suite under a *proposed* fingerprint
without permanently mutating the working tree. **Decision: sub-session source
overrides.** `AgentSessionConfig` carries `atom_source_overrides: dict[str, str]`
(atom name → new source text); eval spawns one child session per task with those
overrides applied. The override is realized by the same synthetic-module pattern
already used by `AtomReloader._AGENT_ATOM_MODULE_PREFIX` — proven mechanism, no
new git ref to clean up.

Shadow-worktree-via-`refs/agentm/eval/<run_id>` was considered and rejected for
MVP: it adds cross-process git state to clean up and serializes poorly with
concurrent tuners. Kept as a Phase-2 option if remote eval execution is ever
needed.

### 6.4 Cross-session atom resolution

The tuner is a separate scenario from its target — it does **NOT** load the
production atoms. `target_atom` resolution therefore takes one of two paths:

1. **In-session** (rare; only when tuner happens to compose the same atom):
   `api.list_atoms()` finds it; `api.reload_atom()` swaps it. Transactional,
   rollback on install failure.
2. **Cross-session** (the normal case): `tool_propose_change` walks
   `contrib/scenarios/<target_scenario>/` parsing each `.py` for a
   `MANIFEST.name` match, then writes the file directly via `ResourceWriter`.
   The next production session loads the new version on startup. "Activation
   = git commit" per [git-backed-versioning.md](git-backed-versioning.md).

The two paths produce different identifiers — record both:

- `kind: "in_session_reload" | "file_commit"` distinguishes them.
- `from_sha` / `to_sha` are the **file's** post-commit git SHAs for `file_commit`
  (with `from_sha=null` for new files); for `in_session_reload` they are the
  in-memory atom hashes.
- `atom_in_session: bool` is recorded so operators can see which path produced
  the activation without re-deriving it from `kind`.

`target_scenario` in `tuner/manifest.yaml::tool_propose_change.config` names the
scenario whose `contrib/scenarios/<name>/` is searched for `target_atom` and
where decisions land (`.agentm/decisions/<target_scenario>/`). The tuner does
not load the target's manifest — only its filesystem layout matters.

---

## 7. Decision Records

`.agentm/decisions/<scenario>/` layout:

```
.agentm/decisions/<scenario>/
├── activations.jsonl         # one record per propose_change call (deployment log)
├── candidates/<id>.json      # ← Phase 2: one record per pool member (search frontier)
├── tree.jsonl                # ← Phase 2: parent → child edges in the search tree
└── budget.json               # ← Phase 2: rollouts_used, usd_used vs budgets
```

MVP writes only `activations.jsonl`. The `candidates/` + `tree.jsonl` schema is reserved for Phase 2's Pareto pool (§11) — laid out now so adding it is a non-breaking schema extension. A single-incumbent MVP is the degenerate case `|candidates| = 1`.

Example `activations.jsonl` entry:

```jsonc
{"at":"2026-05-10T...","kind":"activate","scenario":"rca","atom":"tool_read",
 "from_sha":"abc123","to_sha":"def456",
 "atom_in_session":false,                          // §6.4: cross-session activation
 "reload_kind":"file_commit",                      // "file_commit" | "in_session_reload"
 "gate_bypass":[],                                 // [] | ["threshold"] | ["noise_floor"] | ["threshold","noise_floor"]
 "evidence":{"baseline_run":"er_001","proposed_run":"er_002",
             "primary_metric":"root_cause_hit_rate",
             "baseline":0.62,"proposed":0.78,
             "delta":0.16,
             "noise_threshold_2sigma":0.08,        // §8: independent statistical floor
             "holdout_baseline":0.60,
             "holdout_proposed":0.58},
 "rationale":"v2 reads parquet via row-group boundaries instead of byte offset; ...",
 "by":"rca_tuner@<tuner_session_id>"}
```

Append-only. The `.agentm/decisions/` directory is constitution-write-protected against agent's `tool_edit`/`tool_write` (covered by `is_constitution_path` after a one-line manifest update). Only `tool_propose_change` writes here, mediated.

---

## 8. Promotion Policy

> **Scope note (MVP vs Phase 2)**: in MVP this gate is the *only* filter — it
> decides both "is this candidate worth keeping" and "should production switch
> to it". In Phase 2 (§11, GEPA-style Pareto pool), these split: any candidate
> winning on at least one task enters the pool unconditionally; the four-floor
> gate becomes the **deployment gate** that picks which pool member is live for
> end users. The same four floors, narrower scope. Read the rest of this section
> with that future split in mind — nothing here changes meaning, just the set
> of decisions it governs.

Minimal MVP policy. `activate` requires **all four** to hold simultaneously:

1. **Economic floor**: `delta >= threshold_relative * baseline` on primary
   (default `threshold_relative=0.05`).
2. **Statistical floor (independent)**: `delta > 2 · sqrt(stderr_b² + stderr_p²)`
   on primary. The 2σ check is **independent of `threshold_relative`** —
   even if `threshold_relative=0.0`, an underpowered eval cannot activate.
   Stderr is computed from per-task scores across the eval suite (Bernoulli
   under deterministic graders; sample variance under rubric graders).
3. **Guard envelope**: every declared guard metric within `±guard_tolerance`
   of baseline (default `0.10`).
4. **Holdout non-regression**: `holdout_score_proposed >= holdout_score_baseline - guard_tolerance`.
   Holdout tasks (§3 `holdout: true`) are **not** part of the primary signal —
   they are the structural Goodhart guard. Tuner cannot "see" the holdout score
   improving as a target; it can only fail by regressing on it.

`decision="exploratory"` skips floors 1 and 3 only — **2σ and holdout
non-regression still apply**. This is so "I want production signal on an
underpowered eval" remains a legitimate channel without becoming a hole through
which random changes get activated. The decision record's `gate_bypass` field
lists exactly which floors were waived (`["threshold"]` for normal exploratory;
ill-formed if it lists `noise_floor` or `holdout`).

`decision="rollback"` is always allowed — rollback is the safe direction by
construction.

Why the noise floor is non-negotiable: with `samples_per_task=1` on 8 tasks,
a +12.5% delta can correspond to a single coin flip. Real-LLM bring-up of
`format_fix` produced exactly this case; the 2σ check correctly rejected it
(see `.claude/tasks/2026-05-08-per-task-evolution-real-llm-validation.md`,
Run 2). The `exploratory` channel exists precisely to handle "I know the eval
is underpowered but want to ship it anyway" without dressing it up as evidence.

Policy is evaluated by `tool_propose_change` itself (reads the eval results,
applies the four floors). Putting it in the atom keeps it auditable — the
activation gate is one function, callable from tests, greppable.

Post-MVP: Goodhart guard via production traffic (auto-rollback if guard
regresses after activation), multi-objective Pareto frontier reasoning,
structural enforcement of `stop_after_no_improvement` (today the tuner's
honor system). Defer until activations actually happen regularly.

---

## 9. Worked Example: RCA Tuning Loop

1. User runs `agentm tune rca --iterations 3`. Harness loads `scenarios/rca/tuner/manifest.yaml`.
2. Tuner agent calls `tool_query_traces(task_class="rca", n=50)`. Receives 50 trace summaries.
3. Tuner reads 8 traces, notices: 60% failed at parquet read step (`truncated_at_boundary` error). Hypothesizes `tool_read` truncates parquet wrongly.
4. Tuner calls `tool_eval_run(eval_dir="contrib/scenarios/rca/eval")`. Result: `er_baseline`, `root_cause_hit_rate=0.62`, `stderr=0.04`, `holdout=0.60`.
5. Tuner reads `contrib/scenarios/rca/tool_read.py` source via the `read` tool, designs new version using row-group boundaries.
6. Tuner calls `tool_eval_run(eval_dir=..., atom_source_overrides={"tool_read": <new_source_text>})`. Result: `er_proposed`, `root_cause_hit_rate=0.78`, `stderr=0.04`, `holdout=0.58`.
7. Tuner calls `tool_propose_change(target={"kind":"atom_source", "path":"tool_read.py", "new_content":<text>, "target_atom":"tool_read"}, rationale=..., eval_run_baseline="er_baseline", eval_run_proposed="er_proposed", decision="activate")`.
8. `tool_propose_change` checks all four floors: threshold (delta=0.16, 0.16/0.62=26% ≥ 5% ✓), 2σ (0.16 > 2·√(0.04²+0.04²)=0.113 ✓), guard envelope (refusal_rate Δ within ±10% ✓), holdout (0.58 ≥ 0.60 − 0.10 = 0.50 ✓), tier (1 ✓). §6.4 cross-session path: writes `contrib/scenarios/rca/tool_read.py` via `ResourceWriter` (auto-commit). Decision record written with `kind:"file_commit", atom_in_session:false, gate_bypass:[]`.
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
| P6 | Two tuners (RCA + plan_mode) propose simultaneous mutations to a *shared* atom (`tool_read`) | Writes serialize through ResourceWriter's git layer. MVP: detection is honor-system — tuner prompts must call `tool_query_traces` (or re-eval) before each iteration to refresh baseline. Phase 2: `tool_propose_change` accepts `baseline_fingerprint` and validates `git rev-parse HEAD -- <atom_file>` is unchanged before activating; otherwise rejects with `stale_baseline` requiring re-eval. |
| P7 | Eval run hits `max_turns` budget on a task | Task graded as failed at that task; eval result still comparable across versions |
| P8 | Tuner tries to mutate `permission` (tier-2) | Decision recorded as `pending_human_approval`; reload not applied |
| P9 | Production trace has `task_class="rca"`; `tool_query_traces` filter returns it | Confirmed (round-trip test) |
| P10 | After 10 activated mutations, `git log scenarios/rca/` shows full mutation history with rationales as commit messages | Confirmed (ResourceWriter's commit-with-rationale path) |

---

## 11. Phasing

### MVP

- Eval set scaffolding: directory layout, task YAML schema (with `holdout` flag), rubric grader.
- `task_class` populated on `session.fingerprint` (production + eval).
- `AgentSessionConfig.atom_source_overrides` field (§6.3, sub-session hypothesis testing).
- `.agentm/decisions/**` write-protect (one-line `core-manifest.yaml` update).
- Three new tier-1 atoms: `tool_query_traces`, `tool_eval_run`, `tool_propose_change`.
- One worked tuner — start with `format_fix` as the toy lab, then `scenarios/rca/tuner/`.
- Decision records under `.agentm/decisions/<scenario>/`.
- Four-floor promotion gate (§8): threshold + 2σ + guards + holdout non-regression.
- Per-eval-run cost ceiling: `max_cost_usd` knob in `tuner/manifest.yaml::tool_eval_run.config` (closes §13's "cost of eval runs" — it's a budget field, not a research question).

### Phase 2 — GEPA-shaped reorganization

The MVP is single-incumbent hill climbing. Phase 2 reshapes the loop around
[GEPA](https://arxiv.org/abs/2507.19457)'s evidence that **MAP-Elites-style
illumination over a Pareto candidate pool** dominates greedy promotion (35×
fewer rollouts than GRPO, +13% over MIPROv2, escapes local optima that
greedy gets stuck in). Layered so each item is independently shippable:

1. **Pareto candidate pool**. `.agentm/decisions/<scenario>/candidates/<id>.json` records each pool member with `{parent_id, change_spec, per_task_scores, holdout_scores, rollouts_consumed}`. Inclusion criterion loosens to "wins on ≥1 task vs all retained candidates"; strictly dominated members are pruned. The four-floor gate (§8) becomes the *deployment* gate, not the *inclusion* gate. New atom `tool_query_candidates(scenario) → ParetoFrontier` so the tuner can sample parents stochastically (weighted by per-task wins).

2. **Reflection atom (`tool_reflect`)**. Separates diagnosis from mutation: `tool_reflect(failures: list[TraceSummary], target_module: str) → {diagnosis: str, proposed_mutation: ChangeSpec}` reads `μ_f.feedback_text` (§3.2) verbatim and emits a structured diagnosis + change. Mutation prompt template lives at `scenarios/<name>/eval/reflection_template.md` (per-scenario; itself mutable).

3. **Generalized `ChangeSpec`** (§6). Lift the MVP restriction `kind="atom_source"` only — accept `system_prompt`, `manifest_extensions`, `manifest_field`, and (most ambitious) `scenario_compose`. Each kind needs its own validator. The `scenario_compose` path is what enables ARC-AGI-style whole-system evolution: a candidate can change *what tools the scenario exposes* and *which atoms compose it*, not just rewrite one atom.

4. **System Aware Merge (crossover)**. New decision channel `decision="merge"` taking `parents: list[candidate_id]`. Tuner reads two non-dominated candidates (e.g. one improves tasks 1,3,5 by editing the prompt; another improves 2,4,6 by editing the atom), generates a unified `ChangeSpec` that takes lessons from both. Same gate semantics as `activate`. Lets us escape the MVP's "one mutation kills another" serial choice.

5. **Per-module credit assignment**. `μ_f.module_feedback` (§3.2) drives a round-robin module-selection policy in the tuner: instead of "pick any atom to mutate", the tuner biases toward modules the grader actually fingered. Maps to GEPA §3.2 directly.

6. **Rollout budget tracking**. `.agentm/decisions/<scenario>/budget.json` keeps `{rollouts_used, rollouts_budget, usd_used, usd_budget}`. Per-tuning-session caps replace the per-eval-run cost ceiling from MVP. Lets us reproduce GEPA's "X% of GRPO's budget" comparisons on our own scenarios.

7. **Production-traffic guard regression watching → auto-rollback** (kept from old Phase 2).

8. **Cross-task transfer**: when `tool_read` evolves under RCA, surface the change as a candidate for `plan_mode`'s own eval before activation. Phase-2-shaped: the candidate enters `plan_mode`'s pool with a parent pointer to RCA's pool — eval there decides if it deploys.

9. **Structural `stop_after_no_improvement`**: today the tuner's honor system; Phase 2 wraps `tool_propose_change` with a counter that refuses further `activate` calls after N consecutive rejections, forcing the tuner to either change strategy (jump to a different Pareto member) or escalate.

10. **`baseline_fingerprint` validation** for concurrent tuners (§10 P6 — closes the stale-baseline race).

### Out of scope (not Phase 2 either)

- Weight-space optimization (we evolve `Π`, not `Θ` — keep this boundary; if a model needs fine-tuning it's a different system).
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
- **Tuner concurrency**: two tuners on different scenarios both want to mutate a shared atom. MVP detection is honor-system (tuner re-queries before each iteration); Phase 2 adds structural enforcement via `baseline_fingerprint` validation in `tool_propose_change` (see §10 P6).
- **Eval set drift**: eval tasks themselves may need to evolve as the user's real workload evolves. The `holdout` flag (§3) catches *within-suite* Goodhart but does not catch *suite-vs-production* drift. Out of scope for this design; user's responsibility, mitigated by occasional refresh of the eval suite from production traces.
- **Sample correlation under deterministic graders**: with a deterministic grader and a stable LLM, repeated samples on the same input are highly correlated; `samples_per_task > 1` does **not** divide stderr by √N. Stderr formula in `tool_eval_run` assumes IID across (task, sample) — accurate for rubric graders, optimistic for programmatic ones. Treat each task as one Bernoulli trial under deterministic graders; scale by adding more *tasks*, not more samples. Worth a more careful treatment before scaling to non-deterministic graders.
