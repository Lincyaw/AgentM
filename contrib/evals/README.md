# Benchmark Evaluation

Multi-format benchmark runner for evaluating AgentM on terminal agent tasks.

## Supported formats

| Format | Flag | Source | Scoring |
|---|---|---|---|
| Terminal Bench 1.0 | `--bench tb1` | Local repo (Dockerfile + task.yaml) | F2P / P2P step scores |
| Harbor (TB 2.0) | `--bench harbor` | Local repo (task.toml + instruction.md) | reward (0-1 float) |
| SWE-bench | `--bench swebench-verified` | HuggingFace dataset | Patch extraction (scored upstream) |

## Quick start

```bash
# List tasks
uv run python contrib/evals/bench.py list --bench tb1 --repo ~/longcli-bench/tasks_long_cli

# Mirror upstream images to our registry (Harbor/SWE-bench)
uv run python contrib/evals/bench.py mirror --bench harbor --repo ~/harbor-datasets/terminal-bench \
    --registry opspai --prefix tb2 --tag v1 -j 8

# Run batch evaluation
uv run python contrib/evals/bench.py batch --bench tb1 --repo ~/longcli-bench/tasks_long_cli \
    --model litellm --gateway http://<arl-gateway> --api-key <key> -j 20

# pass@k (multiple independent attempts)
uv run python contrib/evals/bench.py batch --bench tb1 --repo ~/longcli-bench/tasks_long_cli \
    --model litellm -j 20 --attempts 8 --results /tmp/pass8-results
```

## TB2 image overlay

ARL sandbox cannot access github.com. TB2 test.sh scripts download uv + python
at eval time. We pre-install them via a thin Docker overlay:

```bash
# Build overlay images (v1 tag = upstream v0 + uv/python pre-installed)
for task in $(ls ~/harbor-datasets/terminal-bench/); do
  docker build --build-arg BASE_IMAGE=opspai/tb2-${task}:v0 \
    -f contrib/evals/benchmarks/Dockerfile.tb2-overlay \
    -t opspai/tb2-${task}:v1 contrib/evals/benchmarks/
done
docker push opspai/tb2-*:v1  # Volcengine mirrors automatically
```

## Results: Doubao (doubao-seed-2-0-pro) on Terminal Bench 1.0

**Dataset**: 21 tasks from longcli-bench (xv6 OS, CS61A, CMU 15-445 DB, AP1400)

### pass@1

| Metric | Value |
|---|---|
| Overall Pass Rate | 5.0% (1/20) |
| F2P Step Score | 32.5% |
| P2P Step Score | 98.2% |

Only `cs61_fa24_hw08` passed.

### pass@8

| Metric | Value |
|---|---|
| **pass@8** | **19.0% (4/21)** |
| Avg F2P Step | 30.0% |
| Avg P2P Step | 98.5% |

Passed tasks: `cs61_fa24_hw08`, `cs61_fa24_hog`, `ap1400_2_hw26`, `61810_syscall`.

#### By category

| Category | pass@8 | Avg F2P Step |
|---|---|---|
| CS61A (intro) | 2/5 | 46.9% |
| AP1400 (C++) | 1/2 | 54.8% |
| xv6 OS | 1/10 | 25.6% |
| CMU 15-445 DB | 0/3 | 8.6% |

#### Key observations

1. **High ceiling, low consistency.** 19/21 tasks scored >0% at least once;
   10/21 hit best F2P >= 80%. The bottleneck is variance, not capability.
   - `61810_syscall`: 7x 0%, 1x 100% — pure luck
   - `cs61_fa24_scheme`: swings between 7% and 98% across attempts
   - `cs61_fa24_hw08`: 7/8 attempts at 100% — rare stability

2. **Near-misses** (best >= 80% but not passed): `61810_net` (8x 99%,
   systematically missing 1 test), `cs61_fa24_scheme` (97.6%), `ap1400_2_hw35`
   (90.7%), `61810_thread` (90%), `cmu15_445_p0` (83.3%), `61810_fs` (80%).

3. **P2P near-perfect (98.5%)** — the model rarely breaks pre-passing tests.
   The challenge is fixing failing tests (F2P), not regressions.

4. **Improvement vectors**: larger k (brute force), harness-guided retries
   (rescue window / fork replay), or stronger base model.
