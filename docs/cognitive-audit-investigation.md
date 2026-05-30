# Cognitive Audit Harness: Investigation Report

Status: **in-progress** (2026-05-30)
Branch: `eval/rca-replay-fork`

## 1. Research Question

Can a cognitive-audit harness (extractor + auditor sidecar) improve a
main agent's RCA accuracy by surfacing reasoning gaps mid-investigation?

## 2. Experimental Setup

### 2.1 Dataset

**ops-lite-fixed-100**: 100 pre-recorded Doubao agent trajectories on
the ops-lite RCA benchmark. Each case is a dual-injection fault (2
services). Ground truth (GT) is the `(service, fault_kind)` multiset
from `injection.json`.

### 2.2 Baseline

`agentm-ab100-baseline-0525-0847` in `eval.db`:
- Model: Doubao-Seed-2.0-pro
- exact_match: 6/100 (6%)
- any_service_hit: 76/100, all_service_hit: 17/100
- avg precision: 0.203, avg recall: 0.145

### 2.3 Replay-Fork Method

Counterfactual evaluation without re-running the control:

1. Replay the recorded trajectory through the harness pipeline
2. When the auditor surfaces a reminder, **fork**: truncate the
   trajectory at that turn, inject the reminder, and let the agent
   continue freely (new LLM calls)
3. Judge the forked continuation's final answer against GT
4. Compare with the recorded control answer (no re-run needed)

`max_surfaces_per_node=1` (greedy spine): fork at the first surface
point only, then continue the forked agent to completion.

### 2.4 Judge

Deterministic multiset match (`EvaluationResult.exact_match`): the
agent's `(service, fault_kind)` set must exactly equal GT. Softer
metrics: `precision`, `recall`, `any_service_hit`, `all_service_hit`.

### 2.5 Infrastructure

- **litellm proxy** (`101.96.215.245:8088`): Doubao-Seed-2.0-pro,
  ~30M TPM, no per-model quota. Used for full-scale runs.
- **Ark direct**: `ark.cn-beijing.volces.com/api/v3`, 5-hour sliding
  quota per model. Used for single-case validation when litellm was
  down.

## 3. Experiments and Results

### 3.1 With-Extractor (full pipeline)

`runs/doubao-ab100/`

The complete harness pipeline: extractor builds a logic-flow graph
every 5 turns, auditor reads the graph and decides whether to surface
a reminder.

| Metric (fired-only, n=50) | Control | Intervene | Delta |
|---|---|---|---|
| exact_match | 6 | 4 | **-2** |
| any_service_hit | 42 | 41 | -1 |
| avg precision | 0.180 | 0.210 | +0.030 |
| avg recall | 0.160 | 0.160 | +0.000 |
| helped / harmed | | | 0 / 2 |

Notes:
- 35/100 cases had extractor `spawn_error` (litellm connection drops
  at concurrency 30); these never ran the auditor.
- 12/100 errored entirely (connection errors during continuation).

### 3.2 No-Extractor (ablation: auditor reads raw trajectory)

`runs/doubao-noext-ab100/`

Skip the extractor entirely. Feed the raw conversation trajectory
directly to the auditor via a new `auditor_trajectory.md` prompt
variant.

| Metric (fired-only, n=99) | Control | Intervene | Delta |
|---|---|---|---|
| exact_match | 6 | 4 | **-2** |
| any_service_hit | 75 | 77 | +2 |
| avg precision | 0.205 | 0.204 | -0.002 |
| avg recall | 0.146 | 0.131 | -0.015 |
| helped / harmed | | | 0 / 2 |

Key finding: **removing the extractor did not improve auditor
quality.** The GT service mention rate in reminders was identical
(32% in both conditions). The auditor produced more diverse themes
(perf_hint, error_hint) but no better direction.

### 3.3 Upper-Bound v1 (fixed reflection, before submission)

`runs/doubao-upperbound-ab100/`

Bypass the auditor entirely. Inject a fixed behavioral reflection
prompt just **before** the agent calls `submit_investigation`.

| Metric (n=100) | Control | Intervene | Delta |
|---|---|---|---|
| exact_match | 6 | 6 | **+0** |
| helped / harmed | | | 0 / 0 |
| answer changed | | | 3/100 |

Finding: agent mostly re-submitted the identical answer (97/100
unchanged by precision+recall). The fork point was before the
submission tool call, so the agent hadn't "seen" its own answer yet
-- it simply re-submitted.

### 3.4 Upper-Bound v2 (fixed reflection, after submission)

`runs/doubao-upperbound-v2-ab100/`

Same fixed reflection prompt, but injected **after** the full
trajectory (including the submission and its tool result). The agent
sees its own final answer before the reflection prompt.

| Metric (n=100) | Control | Intervene | Delta |
|---|---|---|---|
| exact_match | 6 | 5 | **-1** |
| helped / harmed | | | 0 / 1 |
| answer changed (services) | | | 34/100 |

Detailed breakdown of the 34 changed cases:

| Outcome | Count | Description |
|---|---|---|
| Found more GT services | 3 | Agent discovered a previously missed root cause |
| Pruned false positives | 2 | Agent removed an incorrect service |
| Added false positives | 25 | Agent added services not in GT |
| Lost GT services | 4 | Agent dropped a correct service |

The 66 unchanged cases were NOT "ignored reflection": trace analysis
shows the agent did 3-6 extra queries per case, then re-confirmed its
original answer. It was "reflect -> investigate -> confirm", not
"ignore".

## 4. Four-Way Summary

```
                               with-ext     no-ext  ub-before   ub-after
fired                                50         99        100        100
errored                              12          0          0          0
ctrl exact_match                      6          6          6          6
iv exact_match                        4          4          6          5
delta                                -2         -2         +0         -1
helped (W->R)                         0          0          0          0
harmed (R->W)                         2          2          0          1
answer changed                      N/A        N/A       3/100     34/100
```

**helped = 0 across all 4 experiments (400 intervention attempts).**

## 5. Deep Analysis

### 5.1 Extractor Quality

Prompt asks for a branch-and-merge dependency graph with
consolidation. Actual output:

- **Pure append**: 0 merges, 0 deletes across 1101 node upserts
- **81% act nodes**: records what agent did, not what it concluded
  (only 6 hyp, 35 concl out of 1101 nodes)
- **Edges exist** and are passed to auditor (verified in code:
  `build_auditor_system_prompt` includes edges in the prompt)
- **Branching exists**: 92% of cases have fan-out > 1
- But branching is **action fan-out** (multiple queries from one
  finding), not **reasoning fan-out** (competing hypotheses)

Token cost: ~54K tokens/case (74% of harness total).

### 5.2 Auditor Quality

- **Firing rate**: 95% when extractor works (39/41 all-ok cases)
- **continuation_notes**: high quality -- tracks observations across
  firings, updates status, carries forward properly
- **Factually grounded**: every reminder references real agent actions
  and data
- **GT service in reminder**: 44% (22/50 fired cases)
- **Dominant theme**: coverage_nudge (69/92 messages) -- "you haven't
  investigated X yet"

When reminder mentions GT service: Δprec=+0.091, 23% improved.
When it doesn't: Δprec=-0.018, 0% improved.

The auditor does **reactive coverage nudging**, not **systematic
coverage tracking**. It picks the most "visible" uninvestigated lead,
which has only ~32% chance of being a GT service.

### 5.3 Does the Extractor Bottleneck the Auditor?

**No** -- the no-extractor ablation (3.2) proved this. Giving the
auditor the full raw trajectory instead of the graph produced
identical GT hit rates (32%) and identical outcomes (helped=0,
harmed=2). The auditor's limitation is intrinsic, not caused by poor
graph input.

### 5.4 Does the Agent Listen?

**Yes, thoroughly.** Across all experiments:

- **With-extractor/no-extractor**: agent follows auditor's specific
  direction (e.g., "investigate recommendation service") -- but the
  direction is often wrong, leading to harmed outcomes.
- **Upper-bound v2**: 34/100 agents changed their answer after
  reflection. The 66 unchanged agents also did 3-6 extra queries each
  -- they reflected, investigated, and re-confirmed.
- **The problem is not compliance but direction quality.**

### 5.5 Why Doesn't Intervention Help?

Three structural factors:

1. **Root cause vs cascading effect**: the agent finds real anomalies
   (reservation 4.7x slower, services with elevated latency) that are
   genuine but aren't the *injected* fault. Adding them dilutes
   precision without improving recall.

2. **Auditor is directionally blind**: it surfaces "you haven't
   checked X" based on coverage, not correctness likelihood. Only 32%
   of reminders point toward GT services.

3. **Reflection amplifies existing bias**: when told to "reconsider",
   the agent does more of what it already does -- finds more anomalies
   of the same type it already found. It doesn't fundamentally change
   its investigation strategy.

## 6. Methodology Notes

### 6.1 Ablation Framework

The `ForkStrategy` abstraction in `replay_fork/strategy.py` supports:

- `HarnessStrategy`: full extractor+auditor pipeline
  (`--skip-extractor` flag disables extractor)
- `FixedInjectionStrategy`: inject a fixed reminder at a computed turn
  point (configurable `TurnSelector`)

Pre-built turn selectors: `before_submission`, `after_submission`.

Adding a new ablation requires only a new `ForkStrategy` subclass or
a new `TurnSelector` -- no changes to the driver core.

### 6.2 Key Gotchas

- **Not-fired ≠ auditor silent**: 35/100 "silent" cases in
  with-extractor were actually extractor `spawn_error` (connection
  drops), not auditor choosing silence.
- **`judge_detail` empty for not-fired**: when auditor doesn't fire,
  `judge_detail={}`. Counting `jd.get("any_service_hit", False)` as
  False is correct behavior but can be confused with degradation.
- **Fork point semantics**: `messages[:fork_point]` excludes
  `fork_point`. For upper-bound, `before_submission` forks before the
  agent sees its answer (agent re-submits identically);
  `after_submission` includes the full trajectory (agent can reflect
  on its own answer).
- **Control response from eval.db**: `eval_metrics` are from the
  original eval pipeline judge; `response` is the raw JSON.

### 6.3 Token Economics

Per case (harness healthy):

| Component | Per Firing | Per Case | Growth |
|---|---|---|---|
| Extractor | ~8.7K tokens | ~54K (6.2 firings) | 2.0x first→last |
| Auditor | ~3.8K tokens | ~23K (6.2 firings) | 2.9x first→last |
| Harness total | | ~77K | |

Payload-fix (commit `4bdac485`): elided historical `source_turn_texts`
from extractor payload. Reduced max payload from 392KB to 70KB (max
input from ~25K to ~17K tokens at deepest firing).

## 7. Open Questions

1. **Stronger main agent**: would glm-5.1 or Kimi-K2 benefit from
   intervention? The 5 positive cases in ub-v2 suggest the mechanism
   works -- the bottleneck may be agent capability, not intervention
   design.

2. **More specific reflection**: instead of "did you check
   everything?", could a structured checklist (enumerate all services
   seen in traces, check each one) force systematic coverage?

3. **Multi-round reflection**: allow the agent multiple
   reflect-investigate-resubmit cycles instead of one shot.

4. **Stronger auditor model**: use a more capable model (e.g.,
   Claude/GPT-4) as auditor while keeping Doubao as main agent -- test
   if auditor intelligence is the bottleneck.

5. **Oracle auditor**: inject the GT directly as the reminder ("you
   missed service X") to measure the absolute ceiling of agent's
   ability to use correct information.

## 8. File Index

### Code

| Path | Description |
|---|---|
| `llmharness/audit/runner/runner.py` | HarnessRunner with `skip_extractor` |
| `llmharness/audit/auditor/prompts/auditor_trajectory.md` | No-extractor auditor prompt |
| `llmharness/audit/auditor/prompt.py` | `build_auditor_trajectory_prompt` |
| `llmharness/audit/auditor/extensions.py` | `compose_auditor_trajectory_extensions` |
| `llmharness/replay/fork_tree.py` | `skip_extractor` passthrough |
| `llmharness/replay/offline_driver.py` | `skip_extractor` passthrough |
| `agentm_rca/eval/replay_fork/strategy.py` | `ForkStrategy`, `FixedInjectionStrategy`, turn selectors |
| `agentm_rca/eval/replay_fork/driver.py` | `ReplayForkDriver` with strategy dispatch |
| `agentm_rca/eval/replay_fork/cli.py` | `--skip-extractor`, `--upper-bound` flags |

### Experiment Runs

| Directory | Config | Key Result |
|---|---|---|
| `runs/doubao-ab100/` | with-ext, concurrency 30 | helped=0 harmed=2 |
| `runs/doubao-noext-ab100/` | no-ext, concurrency 30 | helped=0 harmed=2 |
| `runs/doubao-upperbound-ab100/` | ub before submission | helped=0 harmed=0 |
| `runs/doubao-upperbound-v2-ab100/` | ub after submission | helped=0 harmed=1, 34% changed |
| `runs/glm51-baseline-20/` | glm-5.1 main agent, 20 cases | exact=5/20 (25%) |
| `runs/dsv4flash-baseline-20/` | deepseek-v4-flash, 20 cases | exact=3/20 (15%) |
