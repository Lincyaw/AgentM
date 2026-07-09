# LongCLI-Bench Evaluation

Evaluate AgentM on [LongCLI-Bench](https://github.com/finyorko/longcli-bench)
(20 long-horizon programming tasks) via ARL sandbox.

## Architecture

```
AgentM (host) → ARL Gateway → k8s sandbox pod (task image)
    ↓                              ↓
ClickHouse (trajectory)      bash/file tools execute here
```

Agent runs on host, tools execute in an isolated sandbox. Trajectory
stored in host ClickHouse for rescue-window fork/replay.

## Image setup (one-time)

Task images are built from the longcli-bench repo and pushed to a
container registry. ARL pulls them on demand.

```bash
# Prerequisites
git clone https://github.com/finyorko/longcli-bench.git ../longcli-bench

# List all discovered tasks
agentm eval sandbox list --repo ../longcli-bench/tasks_long_cli

# Build + push all 20 task images
agentm eval sandbox build --repo ../longcli-bench/tasks_long_cli \
    --base-dir ../longcli-bench/longcli_dockerImage --push

# Build paired private evaluator images
agentm eval sandbox build --repo ../longcli-bench/tasks_long_cli --eval-only --push

# Only build specific tasks
agentm eval sandbox build --repo ../longcli-bench/tasks_long_cli \
    -t cs61_fa24_hog -t 61810_cow --push
```

Pre-built images are available at `docker.io/opspai/longcli-*:v0`.

## Run

### Batch evaluation (all tasks)

```bash
agentm eval sandbox batch --bench tb1 \
    --repo ../longcli-bench/tasks_long_cli \
    --model litellm -j 20

# pass@k (multiple independent attempts)
agentm eval sandbox batch --bench tb1 \
    --repo ../longcli-bench/tasks_long_cli \
    --model litellm -j 20 --attempts 8

# Only specific tasks
agentm eval sandbox batch --bench tb1 \
    --repo ../longcli-bench/tasks_long_cli \
    -t cs61_fa24_hog -t 61810_cow --model glm47
```

Results go to `~/.agentm/eval_runs/{exp_id}/`. Re-running with existing
results skips completed tasks automatically (resumable).

### Evaluate results

After the agent finishes, evaluation works by reusing the agent's sandbox:

1. The eval phase runs test scripts in the still-live sandbox
2. F2P/P2P scores are computed from test output
3. Results are recorded to `results.jsonl` in the experiment directory

With private evaluator containers, the eval sandbox also starts a private
container with the test suite mounted.

## Task list

| Task | Category | Difficulty | Domain |
|---|---|---|---|
| 61810_cow | feature_add | easy | xv6 kernel (COW fork) |
| 61810_fs | feature_add | medium | xv6 file system |
| 61810_lock | feature_add | medium | xv6 kernel locks |
| 61810_mmap | feature_add | hard | xv6 mmap |
| 61810_net | feature_add | medium | xv6 networking |
| 61810_pgtbl | feature_add | medium | xv6 page tables |
| 61810_syscall | feature_add | easy | xv6 system calls |
| 61810_thread | feature_add | medium | xv6 threads |
| 61810_traps | feature_add | medium | xv6 traps |
| 61810_util | from_scratch | easy | xv6 utilities |
| ap1400_2_hw26 | feature_add | medium | C++ (AP1400) |
| ap1400_2_hw35 | feature_add | medium | C++ (AP1400) |
| cmu15_445_p0 | feature_add | medium | C++ (database) |
| cmu15_445_p1 | feature_add | hard | C++ (buffer pool) |
| cmu15_445_p2 | feature_add | hard | C++ (B+ tree) |
| cs61_fa24_ants | feature_add | easy | Python (CS61A) |
| cs61_fa24_cats | feature_add | easy | Python (CS61A) |
| cs61_fa24_hog | feature_add | easy | Python (CS61A) |
| cs61_fa24_hw08 | feature_add | easy | Python (CS61A) |
| cs61_fa24_scheme | feature_add | medium | Python (CS61A) |
