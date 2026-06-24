# RescueHarness Design (RCA domain)

Implementation design for *The Rescue Window* research memo (`doc.md`), scoped
to the RCA / diagnosis domain. This is the contract for the rewrite that
replaces the llmharness-policy-centric prototype with a **measurement-first**
harness.

Read `doc.md` first. Section numbers below (§) refer to it.

## 1. Positioning and the three architectural flips

The prototype walked a single critic-chosen surface point, injected one
reminder, ran one rollout, and emitted a helped/harmed table. That answers
"did this reminder help on N cases" — the prior reminder case-study — not the
rescue-window science (opportunity prevalence, window, opportunity–realizability
gap). The rewrite flips three things:

1. **Measurement first, critic demoted.** The spine is E1 oracle landscape →
   E2 window. Any critic (E4) is one `Critic` implementation evaluated with the
   *same* metrics — one column in the experiment matrix, not the entry point.
   llmharness is not a core dependency; it may plug in later as one critic.

2. **The `EvalUnit` row is the spine; rollout and analysis layers are fully
   separated.** Expensive fork+continue only writes append-only `EvalUnit` rows.
   Every metric (Q / Δ / G\* / window / η_C / report) is a **pure function over
   rows**. Re-deriving metrics never re-runs rollouts; adaptive-K is "append
   more rows, re-aggregate".

3. **RCA Snapshot Adapter = identity (conversation fork).** The RCA data plane
   is a read-only external DuckDB-over-S3 query surface. Forking the
   conversation prefix *is* an exact checkpoint: no filesystem/process/db state
   is mutated by the actor. This is an explicit contract, validated empirically
   in E0. (The coding domain is where a real snapshot adapter is required; it is
   out of scope here.)

## 2. Data spine: `EvalUnit` (doc §6.2, the `z`)

One row **per rollout** (K rollouts of one treatment = K rows). The row is the
only artifact the rollout layer produces and the only input the analysis layer
consumes.

```python
@dataclass(frozen=True)
class EvalUnit:
    # --- identity / replayable coordinates ---
    case_id: str            # RCA case (carries GT directory)
    repository_id: str      # split key: same source never crosses train/test
    trajectory_id: str      # source baseline trajectory
    actor_id: str           # fixed pi_A model fingerprint
    prefix_id: str          # sampling-point identity (clusters all branches+K)
    fork_point: ForkPoint   # turn_index / message_id selector
    progress: float         # normalized progress in [0,1] of this prefix (time axis)
    remaining_budget: dict  # B_t: remaining turns / tokens / tool quota

    # --- treatment ---
    treatment_id: str       # CONTINUE / PLACEBO / GENERIC / VERIFY@target / ORACLE_GROUNDED / ...
    content_level: ContentLevel
    action: ActionType
    intervention: Intervention   # existing schema: action/target/evidence/strength/valid_until
    rung: LadderRung             # which Rescuability-Ladder rung this row probes

    # --- rollout ---
    branch_seed: int        # K rollouts distinguished by seed
    fork_session_id: str

    # --- outcome (judge) ---
    binary_success: bool | None
    normalized_score: float | None   # continuous partial score = PRIMARY endpoint
    final_state_hash: str | None

    # --- behavior / cost (from trace) ---
    follow_through: bool | None      # did the actor act on the intervention
    stale: bool                      # injection-time state already changed (online only)
    duplicate: bool
    wasted_steps: int | None
    critic_latency_ms: int | None
    cost: dict                       # token / step / tool

    # --- analysis ---
    sampling_weight: float           # stratified-sampling weight to recover the true prefix distribution
```

`prefix_id` is the cluster key for §9.4 statistics: all treatments and all K
rollouts that share a prefix belong to one cluster and never split across
train/validation/test.

## 3. Module decomposition

The package is layered by design intent, not flattened. Package root:
`contrib/evals/rescue_window/src/rescue_window/`.

```
rescue_window/
  cli.py                    # subcommands: sample · run (E1) · aggregate
  model/                    # problem-definition data model (doc §3, §6.2, §7)
    schema.py               #   typed intervention DSL (§7.1): ActionType, ForkPoint, Intervention
    units.py                #   EvalUnit z (§6.2), ContentLevel (§7.2), LadderRung (§4.2), PrefixPoint, Treatment
    store.py                #   append-only EvalUnit row store (backbone)
  harness/                  # replayable benchmark machinery (doc §6, §7, §13.1)
    corpus.py               #   TrajectoryRef + GT pairing + split key (§6.1)
    sampler.py              #   PrefixSampler: stratified fork points (§6.4)
    treatments.py           #   TreatmentFactory + OracleBuilder (strong model + GT, §7 / §4.1)
    judge.py                #   wrap RcabenchJudge -> binary + continuous (§9.1)
    runner.py               #   Branch Runner: fork → inject → K rollouts; CONTINUE = baseline outcome (§6.5/§13.1)
    experiment.py           #   orchestrator: corpus×prefix×treatment×K → store (§8)
    provider_profiles.py    #   config.toml profile → provider (oracle/critic models)
  analysis/                 # measurement, pure functions over rows (doc §3.3, §4, §9, §10)
    aggregate.py            #   Q/Δ/G*/G^C/gap + bootstrap posteriors + adaptive-K (§3.2/§3.3/§9.3)
    window.py               #   Rescue Window: t_open/close/width/peak/area (§4.1)
    ladder.py               #   Rescuability Ladder: CHANNEL vs ACTOR rung gap (§4.2)
    report.py               #   rescue/harm, η_C, taxonomy, case cards (§9.1/§4.3/§10)
  critic/                   # bounded critic baselines (doc §3.4 / E4)
    critics.py              #   Critic Protocol + AbstainCritic / AlwaysVerifyCritic (llmharness-free)
```

Snapshot Adapter has no module: for the read-only RCA data plane it is identity
(conversation fork), validated in E0 (DESIGN §1). The `judge.py` grader wrap is
extracted from `scripts/export_reminder_case_study.py::_judge_payload`; the rest
is new.

## 4. Intervention space (doc §7) and the Oracle

`ContentLevel` is the doc §7.2 ladder. `OracleBuilder` generates oracle rows
from the case GT (`causal_graph_verified.json`).

| content_level | RCA generation | answers |
|---|---|---|
| `CONTINUE` | the baseline trajectory's own recorded outcome (it continued from every prefix on it); reference Q_t(CONTINUE), one sample per trajectory | natural-continuation baseline |
| `PLACEBO` | token-matched neutral text, no task info | disruption tax of a bare interruption |
| `GENERIC` | "your current analysis may have a problem, please re-check" | value of a bare alert |
| `TYPE` | action type only, e.g. "[VERIFY] re-verify one requirement before concluding" | are action primitives enough |
| `TYPE+TARGET` | action + concrete object, target derived from GT | marginal value of target selection |
| `EVIDENCE` | + prefix-visible evidence pointing toward GT | does grounded evidence raise adoption/recovery |
| `ORACLE_GROUNDED` | **primary oracle** — see below | best channel-respecting intervention |
| `ORACLE_DIAG` | raw full GT diagnosis injected directly | absolute upper bound (actor/state-recovery ceiling), separately labeled |

### 4.1 Oracle definition (locked)

**Primary oracle (`ORACLE_GROUNDED`) = strong model + GT, restricted to
information consistent with the then-visible prefix.** The OracleBuilder is a
strong-model call that receives:

- the visible prefix `h_t` (and only `h_t`),
- the ground truth (GT causal graph for the case),

and is instructed to emit a typed intervention (`action` + `target` +
`evidence`) that:

1. **steers toward the GT root cause**, but
2. **cites only prefix-visible evidence** (the `evidence` trajectory-event ids
   must exist in `h_t`), and
3. **respects the typed channel** (it is a VERIFY/REPLAN nudge, not the answer).

The "consistent with then-available information" constraint is the point: the
oracle is *not* a future-leaking GT dump. It is the best intervention a perfect
overseer who happened to know the truth could express through the legal channel
using only what is observable at `t`. This makes `ORACLE_GROUNDED` the
**channel-respecting upper bound** that defines `G*_t`.

`ORACLE_DIAG` (raw GT answer, future-leaking allowed) is kept as a *separate,
explicitly labeled* higher ceiling — it measures the actor/state-recovery limit
(doc §7.2 "oracle diagnosis/repair"), not the realizable channel.

No-future-leakage is enforced mechanically: `OracleBuilder` validates that every
`evidence.trajectory_event_ids[]` resolves inside the prefix; a violation drops
the oracle row rather than silently leaking.

### 4.2 Action set (RCA)

`CONTINUE / VERIFY / REPLAN_SCOPE / FINAL_AUDIT`.

`CHECKPOINT / REVERT_TO_BEST` are **not run in the RCA main study** (pure
read-only diagnosis makes their state semantics weak). They remain in the
`ActionType` enum, reserved for the coding domain.

`G*_t = max_u Δ_t(u)` ranges over
`{TYPE, TYPE+TARGET, EVIDENCE, ORACLE_GROUNDED} × {VERIFY, REPLAN_SCOPE}`.
`ORACLE_DIAG` is reported alongside as the separate ceiling.
`G^C_t = Δ_t(C(h_t))` uses the critic's chosen treatment.

## 5. Phase orchestration (E0→E4)

Each phase is a configuration of the same module chain — the payoff of the
z-row spine.

```
E0 Replay validation   corpus → sampler(small) → treatments(PLACEBO only) → runner → judge → aggregate
   produce: does a fork+prompt(PLACEBO) branch resume faithfully (reproduce baseline-like
            behavior, judge stable)? is the baseline-outcome CONTINUE reference sound?
            branching cost / variance / failure rate.
   Go: forks don't resume faithfully → fix fork/budget inheritance before proceeding (doc §E0 / P0).

E1 Oracle landscape    sampler(stratified) → treatments(full ladder + oracle) → runner(K, adaptive)
   → judge → store → aggregate(Q/Δ/G*) → report(opportunity prevalence + initial taxonomy)
   Go: oracle-actionable fraction >= ~5%? which result pattern (doc §8 E1 table)?

E2 Rescue window       sampler(dense 5–8 points/trajectory) → reuse E1 treatments/runner
   → window(t_open/close/width/peak/area, grouped by error type / phase / budget)

E3 Channel ladder      fix timing+target, sweep content_level only
   → minimal effective information + non-monotonicity (backseat-driving) detection

E4 Bounded critic      critics(baseline).decide → same runner/judge/store
   → aggregate(η_C / Gap / policy regret) → report(coverage–utility curve)
```

E5 (end-to-end online) and E6 (homogeneous stress test) are out of scope for the
first cut. The design leaves a clean seam: the online path reuses the same
`Intervention` DSL plus a live injector and is where `stale` / `valid_until` /
`duplicate` become meaningful. It is intentionally decoupled from any specific
critic implementation.

## 6. Metrics layer + adaptive-K loop

All metrics are pure functions `aggregate(rows)` / `report(rows)` (doc §9.1):
Q, Δ, G\*, G^C, Gap, `η_C = Σ[G^C]₊ / Σ[G*]₊`, Rescue/Harm/Net, coverage–utility,
regression regret, the window five-tuple, and the §10 taxonomy labels.

Adaptive-K (doc §9.3) is a loop over the store:

```
rows = []
queue = [(prefix, treatment, K0=3), ...]
while queue:
    new = run(queue); store.append(new); rows += new
    agg = aggregate(rows)
    # escalate only decision-relevant, undecided pairs:
    #   - oracle-best vs CONTINUE whose Δ posterior sits in [0.2, 0.8]
    #   - critic choice vs CONTINUE
    queue = controller(agg, K_max=20)
```

Primary results report `P(Δ_t(u) > ε | data)`, never a hard recoverable/
irrecoverable label (doc §9.3).

**Endpoint (locked):** continuous fpg score (`root_subject_f1` etc.) is the
primary endpoint feeding Q/Δ and the adaptive-K decision band; binary
`exact_match` is the high-level secondary outcome.

## 7. Statistical design (doc §9.4)

- Split by `repository_id`; the same `prefix_id` (hence all its branches and K
  rollouts) never crosses train/validation/test.
- Continuous score: hierarchical model with task/trajectory random effects, or
  task-level clustered bootstrap.
- Binary success: paired task-level bootstrap / McNemar.
- Pre-registered endpoints: oracle-opportunity prevalence, window width, η_C,
  selective net rescue. action × timing × error-type is exploratory with
  multiple-comparison control.

## 8. §10 outcome-based taxonomy (derived, not annotated)

Per-prefix labels are computed from the rows + ladder, never hand-judged:
`oracle-actionable (G*_t > ε)`, `bounded-actionable (G^C_t > ε)`,
`harm-sensitive`, `channel-limited`, `observability-limited`, `critic-limited`,
`latency-limited`, `irrecoverable under scope`. Human annotation is restricted
to *explaining* error type / evidence visibility / actor response — never to
deciding "should we have intervened".

## 9. Diff against the current tree

- **Reuse**: `schema.Intervention / ActionType / ForkPoint`; `runner.run_branch`
  fork+inject core; `agentm trace`; the RCA grader (extract
  `export_reminder_case_study.py::_judge_payload` into `judge.py`).
- **Remove coupling**: `llmharness_policy.py`; the single-surface logic in
  `policy_runner.py`; the cli `llmharness` subcommand. The export script's
  helped/harmed table is demoted to one case-card view inside `report.py`.
- **New**: `corpus / sampler / treatments / store / aggregate / window / ladder
  / critics / report` + the new cli surface.

## 10. Locked decisions

1. Oracle = strong model + GT, restricted to prefix-consistent information,
   channel-respecting (`ORACLE_GROUNDED`); raw GT dump kept as a separate
   `ORACLE_DIAG` ceiling. (§4.1)
2. Primary endpoint = continuous fpg score; binary exact_match secondary. (§6)
3. `CHECKPOINT / REVERT_TO_BEST` excluded from the RCA main study; reserved for
   the coding domain. (§4.2)
4. RCA Snapshot Adapter = identity (conversation fork), validated in E0. (§1)
