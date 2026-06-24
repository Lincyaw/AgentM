# Rescue Window Evaluation Harness

Measurement-first harness for *The Rescue Window* research
(`doc.md` = research memo, `DESIGN.md` = implementation design).

Scenario-agnostic core; RCA is the first adapter.

## Prerequisites

```bash
# from repo root — installs rescue_window + rca_eval + agentm SDK
uv sync --extra eval

# actor model profile in ~/.agentm/config.toml, e.g.:
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

## Concurrency notes

- `--concurrency N` controls parallel rollouts via `asyncio.Semaphore`
- Store JSONL appends are serialized by `asyncio.Lock`
- `os.environ` mutation (for stored session env vars) is serialized during
  session creation only; rollout execution runs fully concurrent
- All 288+ rollouts are gathered globally (not per-case), so the full
  concurrency budget is used across cases
