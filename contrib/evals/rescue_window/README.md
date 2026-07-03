# Rescue Window Evaluation Harness

Measurement-first harness for *The Rescue Window* research
(`doc.md` = research memo, `DESIGN.md` = implementation design).

Scenario-agnostic core; RCA is the first adapter.

## Prerequisites

```bash
# from repo root — installs rescue_window + rca_eval + agentm SDK
uv sync --extra eval

# actor model profile in $AGENTM_HOME/config.toml
# (default ~/.agentm/config.toml), e.g.:
# [models.litellm]
# provider = "openai"
# model = "doubao-seed-2-0-pro-260215"
# base_url = "http://localhost:8088/v1"
# api_key = "..."
```

## RCA Experiment Walkthrough

### 1. Prepare a corpus

A corpus manifest is a JSON/YAML/CSV file listing baseline trajectories to
fork from. Each row needs `trajectory_id` (or `session_id`), `case_id`, and
`data_dir` (path to GT case directory with `causal_graph_verified.json` etc.).

```json
{
  "trajectories": [
    {
      "session_id": "869b57cbc3474d5c9fb5535887ba662d",
      "case_id": "data_c2d0fb53",
      "data_dir": "/path/to/eval-data/v5bad-harness-drift1/data_c2d0fb53"
    }
  ]
}
```

**Finding baselines from ClickHouse:**

```bash
# list all rca:baseline sessions
agentm trace index | grep 'rca:baseline'

# check a session's final score (judge the recorded trajectory)
agentm trace tools --session <id> --tool submit_final_report
```

Pick sessions where the baseline failed (score < 0.5) — those are the
candidates for rescue.

### 2. Dry-run prefix sampling

```bash
uv run python -c "from rescue_window.cli import app; app()" -- sample \
  --corpus corpus.json \
  --adapter rca \
  --progress "0.2,0.4,0.6,0.8" \
  --min-turn 3
```

Prints sampled prefix points (turn indices at 20/40/60/80% progress plus
`pre_final` = one turn before `submit_final_report`). No LLM calls.

### 3. Run the oracle landscape

The oracle-landscape preset sweeps 6 action types at GT-level targeting to
answer "does rescue opportunity exist, and which action type works?":

| Condition | Action | Content |
|---|---|---|
| CONTINUE | CONTINUE | fork + resume, no injection (counterfactual baseline) |
| PLACEBO | CONTINUE | token-matched neutral text |
| GENERIC:TYPE_TARGET | GENERIC | "there may be an issue" + GT service name |
| VERIFY:TYPE_TARGET | VERIFY | "re-verify" + GT service name |
| ADVISE:TYPE_TARGET | ADVISE | risk area + correction direction + GT service |
| REPLAN:TYPE_TARGET | REPLAN | "pause and re-plan" + GT service |
| FINAL_AUDIT:TYPE_TARGET | FINAL_AUDIT | "audit before submitting" + GT service |
| ADVISE:ORACLE_DIAG | ADVISE | give the verified root cause directly (actor ceiling) |

```bash
uv run python -c "from rescue_window.cli import app; app()" -- run \
  --corpus corpus.json \
  --out results/oracle_landscape.jsonl \
  --adapter rca \
  --preset oracle-landscape \
  --actor-model litellm \
  --k 1 \
  --progress "0.2,0.4,0.6,0.8" \
  --min-turn 3 \
  --concurrency 50 \
  --max-turns 60
```

Key flags:

- `--preset oracle-landscape | content-ladder` — which condition set
- `--actor-model <profile>` — config.toml profile to override the baseline's
  stored provider (needed when the recorded endpoint is unreachable)
- `--k N` — rollouts per cell (start with 1 for exploration, 3+ for stats)
- `--concurrency N` — parallel rollouts (safe: store writes are locked)
- `--max-turns 60` — budget cap per rollout

The store is append-only JSONL; re-running with the same `--out` skips
already-completed `(prefix_id, treatment_id, seed)` cells automatically.

### 4. Run the content ladder

Fix action to VERIFY, sweep information from none to oracle — answers "what
is the minimal effective information?":

```bash
uv run python -c "from rescue_window.cli import app; app()" -- run \
  --corpus corpus.json \
  --out results/content_ladder.jsonl \
  --adapter rca \
  --preset content-ladder \
  --actor-model litellm \
  --k 1 \
  --concurrency 50
```

### 5. Aggregate results

```bash
uv run python -c "from rescue_window.cli import app; app()" -- aggregate \
  results/oracle_landscape.jsonl \
  --out-prefix results/report
```

Outputs:

- `results/report.json` — structured metrics (G\*, gap, window, taxonomy)
- `results/report.md` — human-readable summary with per-prefix cards

Key metrics in the report:

| Metric | Meaning |
|---|---|
| G\*\_t | Oracle opportunity: best achievable score at prefix t |
| G^C\_t | Bounded realization: best score from non-oracle conditions |
| Gap | G\* − G^C (what's left on the table) |
| η\_C | Capture ratio G^C / G\* |
| Rescue rate | fraction of prefixes where Δ > 0 over CONTINUE |
| Harm rate | fraction where Δ < 0 |
| t\_open, t\_close, width | Rescue window temporal boundaries |

## Architecture

```
rescue_window/
├── model/          # EvalUnit, ContentLevel, ActionType, Store
├── harness/        # corpus, sampler, treatments, runner, experiment
│   └── adapter.py  # ScenarioAdapter protocol (scenario-agnostic seam)
├── analysis/       # aggregate, window, ladder, report (pure functions over rows)
└── critic/         # Critic protocol + implementations (E4, not yet wired)
```

**RCA adapter** lives in `contrib/scenarios/rca/src/rca_eval/rescue_window_eval.py`:

- `judge()` — uses `RcabenchJudge` (same judge as the main eval pipeline)
- `ground_truth()` — uses `fpg.Scenario.graph.root_causes` (authoritative GT)
- `final_tool = "submit_final_report"`

## Adding a new scenario adapter

Implement the `ScenarioAdapter` protocol:

```python
class MyAdapter:
    name = "my_scenario"
    final_tool = "submit_answer"  # the tool that ends the task

    async def judge(self, messages: list[AgentMessage], ref: TrajectoryRef) -> ScoredOutcome:
        # Score the rollout's messages
        return ScoredOutcome(binary_success=..., normalized_score=..., detail={})

    def ground_truth(self, ref: TrajectoryRef) -> GroundTruth:
        # Load GT from ref.data_dir
        return GroundTruth(targets={"answer"}, summary="...", fault_kinds=set())
```

Register it in `harness/adapter.py`:

```python
_ADAPTERS = {
    "rca": "rca_eval.rescue_window_eval:RcaRescueAdapter",
    "my_scenario": "my_package:MyAdapter",
}
```

Then use `--adapter my_scenario` in the CLI.

## Run log

### 2026-06-24: oracle-landscape batch (8 failing RCA baselines)

**Corpus**: 8 cases from `eval-data/v5bad-harness-drift1/`, all baseline
failures (score < 0.5). Actor = Doubao (`doubao-seed-2-0-pro-260215`),
overridden via `--actor-model litellm` (original stored endpoint was
unreachable due to IP rotation).

```
eval-data/v5bad-harness-drift1/
├── data_c2d0fb53   (session 869b57cbc3474d5c9fb5535887ba662d)
├── data_1bd3e155   (session b9025631f672464e8d3e0e1a44d6f2d9)
├── data_0b806766   (session 5e7af2b223b5473ba967f5488e8e0d08)
├── data_79199624   (session 399fe99fae0a4dceb79d4a43d0323d81)
├── data_5cf18749   (session b6dc9622e82b47eaa9d305b119cc7a55)
├── data_8bc5efda   (session 152745ef0e9541b7859b4db9fa2d66e6)
├── data_f12354b8   (session 491f4e22c3ca4865aa529f7117716844)
└── data_a64c6de4   (session 48c63b38f706482f9bdfb7ff240f0963)
```

**Config**: `--preset oracle-landscape --k 1 --concurrency 50
--progress 0.2,0.4,0.6,0.8 --min-turn 3 --max-turns 60`

**Scale**: 8 cases × 36 prefixes × 8 conditions × K=1 = 288 rollouts.

**Store**: `batch_oracle_landscape.jsonl` (append-only, resumable).

**Results (288/288 completed, 0 failures)**:

| Metric | Value |
|---|---|
| Opportunity prevalence | 94.4% (34/36 prefixes have G\* > 0) |
| Mean G\* | 0.323 (oracle can lift score ~32pp on average) |
| Mean gap | 0.054 (TYPE_TARGET nearly saturates; oracle adds little) |
| Rescue window | 100% exist, mean width 0.456, mean area 0.146 |
| Harm-sensitive prefixes | 80.6% (some conditions hurt at most prefixes) |
| Binary rescue rate | 2.0% (low — continuous scores improve but rarely flip binary pass) |

**Action effectiveness** (which action type wins G\* most often):
- REPLAN and ADVISE:TYPE_TARGET dominate mid-trajectory prefixes
- FINAL_AUDIT strong at early prefixes (t3–t4)
- GENERIC sometimes competitive (surprising — bare alarm is enough)
- ORACLE_DIAG hits 1.0 on select prefixes (actor ceiling confirmed)
- channel_limited rare (most rescues don't need the oracle answer)

### 2026-06-24: oracle-landscape batch (30 hard RCA cases — ops-lite-allwrong-31)

**Corpus**: 30 cases from `ops-lite-allwrong-31` fixture (31 ALL_WRONG cases
from Doubao baseline; 1 excluded for missing testbed in GT). Fresh baselines
run with `session.prompt()` and OTEL export.

**Config**: `--preset oracle-landscape --actor-model litellm --k 1
--concurrency 30 --progress 0.2,0.4,0.6,0.8 --min-turn 3 --max-turns 60`

**Scale**: 30 cases × ~5 prefixes × 8 conditions × K=1 ≈ 1200 rollouts.

**Store**: `results/oracle_landscape_31hard.jsonl` (append-only, resumable).

**Results (30/30 cases completed, 1176 rollouts, 7 failures)**:

| Metric | Value |
|---|---|
| Opportunity prevalence | 90.5% (G\* > 0 at 90% of prefixes; 29/30 cases) |
| Mean G\* | 0.391 (oracle lifts score ~39pp on average) |
| Mean gap | 0.153 (bounded conditions capture ~61% of oracle) |
| Rescue window | 100% exist, mean width 0.559, mean area 0.241 |
| Harm-sensitive prefixes | 44.2% |
| Binary rescue rate | 0.2% (ALL_WRONG cases — continuous Δ > 0 but rarely flips binary) |

**Action hierarchy** (Δ over CONTINUE, N≈146 each):
- ORACLE_DIAG: +34.1pp (actor ceiling, 99.3% beat rate)
- REPLAN: +11.6pp (best low-bandwidth, net rescue +29.2%)
- GENERIC: +9.5pp (bare alarm surprisingly strong)
- ADVISE: +8.7pp
- FINAL_AUDIT: +8.7pp
- VERIFY: +8.2pp
- PLACEBO: +0.3pp (disruption tax ≈ 0)

Full analysis with findings, temporal structure, per-case G\*/gap, and
hypothesis validation in `doc.md` Appendix A.

### 2026-06-25: content-ladder batch (30 hard RCA cases)

Same corpus as above. Fixes action to VERIFY, sweeps information level.

**Config**: `--preset content-ladder --actor-model litellm --k 1
--concurrency 30 --max-turns 60`

**Scale**: 30 cases × 147 prefixes × 7 conditions × K=1 = 1029 rollouts.

**Store**: `results/content_ladder_31hard.jsonl`

**Results (1029/1029 completed, 0 failures)**:

Content information ladder (marginal value of each information step):

| Level | Mean Score | Δ(CONTINUE) | Marginal Δ |
|---|---|---|---|
| CONTINUE | 0.183 | baseline | — |
| PLACEBO | 0.198 | +0.015 | +0.015 (≈0) |
| GENERIC | 0.190 | +0.008 | -0.007 (≈0) |
| +TYPE | 0.180 | -0.003 | -0.010 (≈0) |
| **+TARGET** | **0.230** | **+0.048** | **+0.050** (jump!) |
| +EVIDENCE | 0.254 | +0.071 | +0.023 |
| ORACLE_DIAG | 0.502 | +0.319 | +0.248 (massive) |

Key finding: **target selection is the critical information dimension**.
Below TYPE_TARGET, interventions are ineffective. Action type (VERIFY vs
REPLAN vs ADVISE) matters far less than knowing *which service to check*.
Full analysis in `doc.md` Appendix A.5.

## Concurrency notes

- `--concurrency N` controls parallel rollouts via `asyncio.Semaphore`
- Store JSONL appends are serialized by `asyncio.Lock`
- `os.environ` mutation (for stored session env vars) is serialized during
  session creation only; rollout execution runs fully concurrent
- All 288+ rollouts are gathered globally (not per-case), so the full
  concurrency budget is used across cases
