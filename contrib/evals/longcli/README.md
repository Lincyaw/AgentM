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

## Prerequisites

```bash
# 1. kind cluster with ARL deployed
kubectl --context kind-arl-agentm get pods -n arl

# 2. Build task Docker images
cd ~/AoyangSpace/longcli-bench
for task in tasks_long_cli/*/; do
  name=$(basename "$task")
  docker build -t "tb/$name:v0" "$task" -q
done

# 3. Load into kind
kind load docker-image tb/*:v0 --name arl-agentm

# 4. Ensure NodePort 28080 maps to gateway
curl -s http://localhost:28080/healthz
```

## Run

### Single task

```bash
AGENTM_AGENT_ENV_IMAGE="tb/cs61_fa24_hog:v0" \
AGENTM_AGENT_ENV_GATEWAY_URL="http://localhost:28080" \
AGENTM_AGENT_ENV_EXPERIMENT_ID="longcli-hog" \
uv run agentm --scenario terminal_bench_arl --model glm47 \
  -p "Open and follow the detailed project specification at INSTRUCTION.md. \
Implement the CS61A Hog project tasks accordingly in folder cs61-hog."
```

### All 20 tasks (batch)

```bash
bash contrib/evals/longcli/run_batch.sh
```

Results go to `/tmp/longcli-results/<task>.log`. The script auto-skips
completed tasks (resumable).

### Evaluate results

After agent finishes, evaluate by replaying the trajectory in a fresh
sandbox and running the task's test suite. See `rescue_window/harness/replay.py`
for the replay module.

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
