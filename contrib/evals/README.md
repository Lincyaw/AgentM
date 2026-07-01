# Benchmark Evaluation

Multi-format benchmark runner for evaluating AgentM on terminal agent tasks.

## Supported formats

| Format | Flag | Source | Scoring |
|---|---|---|---|
| Terminal Bench 1.0 | `--bench tb1` | Local repo (Dockerfile + task.yaml) | F2P / P2P step scores |
| Harbor (TB 2.0) | `--bench harbor` | Local repo (task.toml + instruction.md) | reward (0-1 float) |
| SWE-bench Pro (Harbor) | `--bench harbor` | Harbor export of `scale-ai/swe-bench-pro` | reward (0-1 float) |
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

## Private evaluator containers

`bench.py` supports `--private-eval` as an adapter-level capability: the runner
creates the ARL private container and calls the adapter's private scoring hook,
but it does not know the benchmark's test layout. Adapters that want this path
implement `build_eval_image()`, optional `private_eval_container()`, and
`evaluate_private_container()`.

The Terminal Bench adapter implements these hooks for `tests/` +
`run-tests.sh` tasks:

```bash
uv run python contrib/evals/bench.py build --bench tb1 \
    --repo ~/longcli-bench/tasks_long_cli --eval-only --push
uv run python contrib/evals/bench.py batch --bench tb1 \
    --repo ~/longcli-bench/tasks_long_cli --private-eval
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

## Running TB2 end-to-end

```bash
# 1. Download dataset from Harbor registry
harbor dataset download "terminal-bench@2.0" -o ../harbor-datasets/

# 2. Mirror upstream images to Docker Hub (crane does registry-to-registry, no local pull)
#    Install crane: go install github.com/google/go-containerregistry/cmd/crane@latest
uv run python contrib/evals/bench.py mirror --bench harbor \
    --repo ../harbor-datasets/terminal-bench \
    --registry opspai --prefix tb2 --tag v0 -j 8

# 3. Build overlay images (pre-install uv + python for offline sandbox)
for task in $(ls ../harbor-datasets/terminal-bench/); do
  docker build --quiet --build-arg BASE_IMAGE=opspai/tb2-${task}:v0 \
    -f contrib/evals/benchmarks/Dockerfile.tb2-overlay \
    -t opspai/tb2-${task}:v1 contrib/evals/benchmarks/
done

# 4. Push overlay images (Volcengine mirrors docker.io/opspai/ automatically)
for task in $(ls ../harbor-datasets/terminal-bench/); do
  docker push opspai/tb2-${task}:v1
done

# 5. Run evaluation (use Volcengine registry prefix for ARL cluster)
uv run python contrib/evals/bench.py batch --bench harbor \
    --repo ../harbor-datasets/terminal-bench \
    --registry pair-cn-shanghai.cr.volces.com/opspai --prefix tb2 --tag v1 \
    --model litellm --gateway http://<arl-gateway> --api-key <key> \
    -j 5 --eval-timeout 900 --results /tmp/tb2-results
```

**Notes:**
- Keep concurrency <= 5 for first run; Volcengine registry has pull QPS limits
  on cold images. After images are cached on nodes, concurrency can go higher.
- Re-running with existing logs skips completed tasks automatically.
- The `--registry` must be the full Volcengine prefix
  (`pair-cn-shanghai.cr.volces.com/opspai`), not bare `opspai`, because the
  K8s cluster cannot reach docker.io directly.

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

## Results: Doubao on Terminal Bench 2.0

**Dataset**: 89 tasks from Harbor terminal-bench@2.0

### pass@1

| Metric | Value |
|---|---|
| **pass@1** | **16.9% (15/89)** |
| reward=0 | 74 |

Passed tasks: `cobol-modernization`, `code-from-image`, `constraints-scheduling`,
`crack-7z-hash`, `git-leak-recovery`, `git-multibranch`, `kv-store-grpc`,
`large-scale-text-editing`, `mailman`, `merge-diff-arc-agi-task`, `polyglot-rust-c`,
`prove-plus-comm`, `regex-log`, `sqlite-with-gcov`, `vulnerable-secret`.

#### Key observations

1. **16.9% pass@1** — comparable to TB1's 19% pass@8, suggesting TB2's difficulty
   is roughly similar when measured against the same model.

2. **Diverse capabilities**: passes span code generation (cobol, polyglot-rust-c),
   git operations (leak-recovery, multibranch), system admin (mailman, kv-store-grpc),
   security (crack-7z-hash, vulnerable-secret), math proofs (prove-plus-comm),
   and data processing (regex-log, merge-diff-arc-agi-task).

3. **Infra note**: ~15 tasks consistently hit gRPC executor timeout during eval
   (ephemeral-storage eviction on nodes → pod killed → connection refused).
   These get reward=0.0 as fallback. True pass rate may be slightly higher
   with a healthier cluster.

## Running SWE-bench Pro from Harbor

Harbor publishes Scale AI's SWE-bench Pro as `scale-ai/swe-bench-pro`.
Export the dataset or a selected task with the Harbor CLI, then use the generic
Harbor adapter:

```bash
# Full dataset export (731 tasks; usually do this once)
uv run harbor download scale-ai/swe-bench-pro@latest \
    -o runs/harbor-datasets --export

# Or export one known task for smoke tests
uv run harbor task download scale-ai/<task-name>@<sha256-ref> \
    -o runs/swebenchpro-harbor-sample --export

# Mirror task images into the Guangzhou registry used by ARL.
# The adapter reads source images from either task.toml environment.docker_image
# or environment/Dockerfile's FROM line.
uv run python contrib/evals/bench.py mirror --bench harbor \
    --repo runs/harbor-datasets/swe-bench-pro \
    --registry pair-diag-cn-guangzhou.cr.volces.com/pair \
    --prefix swebenchpro --tag v0 -j 8

# Baseline
uv run python contrib/evals/bench.py batch --bench harbor \
    --repo runs/harbor-datasets/swe-bench-pro \
    --registry pair-diag-cn-guangzhou.cr.volces.com/pair \
    --prefix swebenchpro --tag v0 \
    --model azure-gpt --gateway http://<arl-gateway> --api-key <key> \
    --scenario terminal_bench:arl \
    --agent-timeout 7200 --eval-timeout 3000 \
    --results runs/swebenchpro-baseline

# Adapt agent
uv run python contrib/evals/bench.py batch --bench harbor \
    --repo runs/harbor-datasets/swe-bench-pro \
    --registry pair-diag-cn-guangzhou.cr.volces.com/pair \
    --prefix swebenchpro --tag v0 \
    --model azure-gpt --gateway http://<arl-gateway> --api-key <key> \
    --scenario terminal_bench:arl_adapt \
    --agent-timeout 7200 --eval-timeout 3000 \
    --results runs/swebenchpro-adapt
```
