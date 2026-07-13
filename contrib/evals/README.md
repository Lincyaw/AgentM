# Benchmark Evaluation

Unified evaluation framework for AgentM benchmarks. All benchmarks share a
common experiment lifecycle (auto exp_id, structured output, ClickHouse
linking, common result format).

## Quick start

```bash
# List available benchmarks
agentm eval list

# Run a benchmark (each has its own subcommands)
agentm eval sandbox batch --bench harbor --repo ~/harbor-datasets/terminal-bench -j 5
agentm eval aftraj-auditor run --model litellm-dsv4flash --limit 50
agentm eval tau2 run --model litellm-dsv4flash --domain airline --num-tasks 5

# List past experiments
agentm eval runs

# View results
agentm eval report <exp_id>

# Export trajectories from ClickHouse
agentm eval export <exp_id>
```

## Benchmarks

| Benchmark | CLI name | Scoring |
|---|---|---|
| Terminal Bench 1.0 | `sandbox --bench tb1` | F2P / P2P step scores |
| Harbor (TB 2.0) | `sandbox --bench harbor` | reward (0-1 float) |
| LHTB (Long-Horizon Terminal-Bench) | `sandbox --bench lhtb` | reward (0-1 float), solved ≥ 0.95 |
| SWE-bench Verified | `sandbox --bench swebench-verified` | Patch extraction |
| SWE-bench Pro | `sandbox --bench swebench-pro` | Patch extraction |
| AFTraj-2K auditor | `aftraj-auditor` | F1, ASS, FAR, StepAcc |
| AFTraj-2K grounding | `aftraj-grounding` | Symbol/dep/anaphor counts |
| tau2-bench | `tau2` | reward, pass rate |
| TELBench | `telbench` | Span-level precision/recall |
| Rescue window | `rescue-window` | Rescue rate |

## Sandbox benchmarks (TB1, Harbor, SWE-bench)

Docker-sandbox-based benchmarks run agents in ARL sandbox pods.

```bash
# List tasks
agentm eval sandbox list --bench tb1 --repo ~/longcli-bench/tasks_long_cli

# Mirror upstream images (Harbor/SWE-bench)
agentm eval sandbox mirror --bench harbor --repo ~/harbor-datasets/terminal-bench \
    --registry opspai --prefix tb2 --tag v1 -j 8

# Build task images (TB1)
agentm eval sandbox build --bench tb1 --repo ~/longcli-bench/tasks_long_cli --push

# Run batch evaluation
agentm eval sandbox batch --bench tb1 --repo ~/longcli-bench/tasks_long_cli \
    --model litellm -j 20

# pass@k (multiple independent attempts)
agentm eval sandbox batch --bench tb1 --repo ~/longcli-bench/tasks_long_cli \
    --model litellm -j 20 --attempts 8
```

### Private evaluator containers

The sandbox adapter supports private evaluator containers as an
adapter-level capability. Adapters implement `build_eval_image()` and
`evaluate_private_container()`:

```bash
agentm eval sandbox build --bench tb1 --repo ~/longcli-bench/tasks_long_cli \
    --eval-only --push
```

### TB2 image overlay

ARL sandbox cannot access github.com. TB2 test.sh scripts download uv +
python at eval time. Pre-install via overlay:

```bash
for task in $(ls ~/harbor-datasets/terminal-bench/); do
  docker build --build-arg BASE_IMAGE=opspai/tb2-${task}:v0 \
    -f contrib/evals/src/agentm_eval/adapters/sandbox/Dockerfile.tb2-overlay \
    -t opspai/tb2-${task}:v1 contrib/evals/src/agentm_eval/adapters/sandbox/
done
```

### Running SWE-bench Pro from Harbor

```bash
# Download dataset
harbor dataset download "terminal-bench@2.0" -o ../harbor-datasets/

# Run with docker.io mirror (no image prep needed)
agentm eval sandbox batch --bench harbor \
    --repo ~/AoyangSpace/harbor-datasets/swe-bench-pro \
    --registry pair-cn-guangzhou.cr.volces.com --source-images \
    --prefix swebenchpro \
    --model litellm-dsv4flash \
    --scenario terminal_bench:arl \
    --agent-timeout 7200 --eval-timeout 3000 -j 5
```

### Running LHTB (Long-Horizon Terminal-Bench)

[LHTB](https://github.com/zli12321/LHTB) is a 46-task long-horizon companion
to Terminal-Bench 2.0 in standard Harbor task layout. All tasks declare
prebuilt docker.io images in `task.toml`, so no image prep is needed —
run straight through the pull-through mirror:

```bash
# Clone (LFS blobs live only under solution/, not needed for evaluation)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/zli12321/LHTB.git

agentm eval sandbox batch --bench lhtb \
    --repo ~/AoyangSpace/LHTB/tasks \
    --registry pair-cn-guangzhou.cr.volces.com --source-images \
    --prefix lhtb \
    --model litellm-dsv4flash \
    --scenario terminal_bench:arl \
    --agent-timeout 5400 --eval-timeout 600 -j 5
```

The `lhtb` format is the Harbor adapter plus LHTB conventions: a task
counts as solved at reward ≥ 0.95 (upstream definition), and each task's
`[verifier] timeout_sec` from `task.toml` extends `--eval-timeout` when
larger. Upstream runs give agents a 90-minute budget per task, hence
`--agent-timeout 5400`.

Caveat: `sudoku-recovery` relies on Harbor's healthcheck to pre-start its
game daemon as root and declares an unprivileged `[agent] user`; the ARL
runner honors neither, so the agent starts the daemon itself with full
privileges and that task's anti-cheat guarantees are void here.

## Experiment output

Every run produces a structured output directory:

```
~/.agentm/eval_runs/{exp_id}/
  meta.json         # benchmark, model, params, status, timestamps
  results.jsonl     # one TaskResult per task
  report.txt        # aggregate summary
  artifacts/        # benchmark-specific files (scores, logs, etc.)
```

The `exp_id` is auto-generated as `{benchmark}-{model}-{YYYYMMDD}-{HHMMSS}-{uuid}`
and flows into ClickHouse as `eval_run_id` for trajectory linking.

## Adding a new benchmark

See the `eval-development` skill (`.claude/skills/eval-development/SKILL.md`)
for the adapter interface, step-by-step guide, and best practices.

## Registry: docker.io pull-through mirror

The ARL cluster cannot reach docker.io directly, but
`pair-cn-guangzhou.cr.volces.com` is a Docker Hub pull-through mirror:
**any docker.io image is pullable by swapping the prefix only** — path and
tag stay verbatim (`jefzda/sweap-images:tag` →
`pair-cn-guangzhou.cr.volces.com/jefzda/sweap-images:tag`).

- **Upstream images** (SWE-bench Pro, etc.): no mirror/retag step — run
  with `--registry pair-cn-guangzhou.cr.volces.com --source-images`.
- **Our own images** (tb1, tb2 overlays): `docker push
  docker.io/opspai/<name>:<tag>` once, then pull via the mirror.

## Historical results

### Doubao (doubao-seed-2-0-pro) on Terminal Bench 1.0

**Dataset**: 21 tasks from longcli-bench (xv6 OS, CS61A, CMU 15-445 DB, AP1400)

| Metric | pass@1 | pass@8 |
|---|---|---|
| Overall Pass Rate | 5.0% (1/20) | 19.0% (4/21) |
| Avg F2P Step | 32.5% | 30.0% |
| Avg P2P Step | 98.2% | 98.5% |

### Doubao on Terminal Bench 2.0

**Dataset**: 89 tasks from Harbor terminal-bench@2.0

| Metric | pass@1 |
|---|---|
| Overall Pass Rate | 16.9% (15/89) |
