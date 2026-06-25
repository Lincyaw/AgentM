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
container registry. ARL pulls them on demand — no manual loading needed.

```bash
# Build + push all 20 task images to docker.io
bash contrib/evals/longcli/setup_images.sh --push

# Custom registry
bash contrib/evals/longcli/setup_images.sh --registry ghcr.io/myorg --push

# Local only (no push, for kind dev clusters)
bash contrib/evals/longcli/setup_images.sh --load-kind arl-agentm
```

Prerequisites for building:
```bash
git clone https://github.com/finyorko/longcli-bench.git ~/AoyangSpace/longcli-bench
```

The script builds base images (`tb/make-pytest:v0`, `tb/c-env:v0`) from
`longcli_dockerImage/`, then builds each task image and tags as
`<registry>/longcli-<task>:<tag>`.

## Run

### Single task

```bash
AGENTM_AGENT_ENV_IMAGE="opspai/longcli-cs61_fa24_hog:v0" \
AGENTM_AGENT_ENV_GATEWAY_URL="http://<arl-gateway>:8080" \
AGENTM_AGENT_ENV_EXPERIMENT_ID="longcli-hog" \
uv run agentm --scenario terminal_bench_arl --model glm47 \
  -p "Open and follow the detailed project specification at INSTRUCTION.md. \
Implement the CS61A Hog project tasks accordingly in folder cs61-hog."
```

### All 20 tasks (batch)

```bash
bash contrib/evals/longcli/run_batch.sh \
  --gateway http://<arl-gateway>:8080 \
  --model glm47

# Only specific tasks
bash contrib/evals/longcli/run_batch.sh --task cs61_fa24_hog --task 61810_cow
```

Results go to `/tmp/longcli-results/<task>.log`. The script auto-skips
completed tasks (resumable).

### Evaluate results

Task images contain only the project skeleton and INSTRUCTION.md — no
tests. After the agent finishes, evaluate by:

1. Creating a fresh sandbox with the same task image
2. Replaying the agent's tool calls (from ClickHouse trajectory)
3. Uploading the task's test suite via `session.upload_file()` / `download_file()`
4. Running `run-tests.sh` → parsing F2P/P2P scores

See `rescue_window/harness/replay.py` for the replay module and
`longcli-bench/tasks_long_cli/<task>/tests/` for test suites.

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
