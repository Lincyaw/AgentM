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
| senior-swe-bench (Snorkel v2026.06) | `sandbox --bench senior-swe` | reward (binary 0/1), LLM-judge verifier |
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
`--agent-timeout 5400`. Task resource sizing (`[environment]
cpus/memory_mb`) is forwarded to the sandbox (note: ARL managed pools pin
resources at first creation per image, so a pool created before this
takes effect keeps its old sizing until GC'd).

Compose sidecar tasks (chess-mate) run their non-`main` services as ARL
private containers in the agent's pod. Three in-pod deltas the adapter
handles automatically: service hostnames collapse to localhost (agent
Dockerfile ENV URLs are rewritten), ports that collide with the ARL
executor's reserved 8080 are remapped (+10000, with the sidecar's own
`*_PORT` env updated to match), and the sidecar's ENTRYPOINT/CMD is
passed explicitly because the gateway otherwise replaces an unset
command with `sleep infinity`. The verifier gets `[verifier.env]` from
task.toml injected (e.g. chess-mate's `RESULT_TOKEN`), with
`${VAR:-default}` references expanded against the host environment.

Caveats:

- `sudoku-recovery` relies on Harbor's healthcheck to pre-start its game
  daemon as root and declares an unprivileged `[agent] user`; the ARL
  runner honors neither, so the agent starts the daemon itself with full
  privileges and that task's anti-cheat guarantees are void here.
- Prebuilt images can lag the git tree. A byte-level audit (2026-07-13,
  every git `environment/` file COPYed by each Dockerfile diffed against
  the pinned image's content) found 44/46 tasks consistent and 2 stale:
  - `unknown-config-semantics` — git revised the whole `cfg` harness
    (HMAC-chained daemon log) after the `:20260615` image; the uploaded
    verifier fails with a `missing HMAC key` infrastructure error, so two
    different models both scored 0.0. Rebuilt image:
    `opspai/lhtb-unknown-config-semantics:20260713` (run it via
    `--registry pair-cn-guangzhou.cr.volces.com/opspai --prefix lhtb
    --tag 20260713`, without `--source-images`).
  - `chess-mate` — git reworked the task into a docker-compose sidecar
    architecture (agent container + isolated Stockfish `game` referee); no
    published image matches. Supported via ARL private containers: build
    and push both images from the git checkout
    (`docker build -t opspai/lhtb-chess-mate:20260713 environment/` and
    `docker build -f environment/Dockerfile.game -t
    opspai/lhtb-chess-mate-game:20260713 environment/`), then run like
    `unknown-config-semantics` above — the adapter derives the sidecar
    from `docker-compose.yaml` automatically.
  Apparent mismatches on the four `apex-*-matter` tasks and
  `tabular-data-feature-covshift` are deliberate anti-cheat build steps
  (world-builder inputs and dataset generators are consumed and deleted
  during `docker build`), not staleness. If any other task scores 0.0
  with a verifier-infrastructure error rather than a grading failure,
  rebuild its image from `environment/Dockerfile` and re-run.

### Running senior-swe-bench (Snorkel v2026.06)

[senior-swe-bench](https://github.com/snorkel-ai/senior-swe-bench-v2026.06)
is a 50-task Harbor dataset of senior-level fixes/features across real repos
(better-auth, electric, gitea, immich, posthog, prefect, turborepo, …). Two
things make it different from LHTB:

1. **No prebuilt images.** Every `task.toml` uses a local `base_image` tag;
   the `environment/Dockerfile` clones the target repo at build time, rewinds
   to the pre-fix commit, and builds the workspace. Both `harbor run -d` and
   `harbor run --repo` build locally — there is nothing to pull. We build once
   and push under our own registry:

   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone \
       https://github.com/snorkel-ai/senior-swe-bench-v2026.06.git
   agentm eval sandbox build --bench senior-swe \
       --repo senior-swe-bench-v2026.06/tasks \
       --registry docker.io/opspai --prefix ssb --tag v1 --push
   ```

2. **LLM-judge verifier.** `test.sh` runs deterministic tests plus rubric,
   taste, and (on 25/50 tasks) a validation-agent judge. The judge model and
   credentials come from `[verifier.env]` in each `task.toml`; the adapter
   expands those `${VAR:-default}` placeholders against the **host** env at
   eval time, so export judge creds before running. The adapter additionally
   forwards `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` /
   `OPENAI_API_BASE` from the host when set, treating a task.toml `${VAR:-}`
   empty default as absent so the host value wins (the allowlist predates
   non-big-three providers). Route the judges to DeepSeek via litellm's native
   provider — set `DEEPSEEK_API_KEY` and the `SSB_OVERRIDE_*` slugs; the
   adapter mirrors `DEEPSEEK_API_KEY` into `OPENAI_API_KEY` so the verifier's
   `have_credentials()` gate (which only recognizes Portkey/Anthropic/OpenAI
   keys) passes without a manual export:

   ```bash
   export DEEPSEEK_API_KEY=sk-...                     # judge + gate (auto-bridged)
   export SSB_OVERRIDE_ALL_JUDGE_MODEL=deepseek/deepseek-chat
   export SSB_OVERRIDE_CLASSIFIER_MODEL=deepseek/deepseek-chat
   export SSB_OVERRIDE_VA_HARNESS=mini-swe-agent      # validation-stage tasks
   export SSB_OVERRIDE_VA_MODEL=deepseek/deepseek-chat

   agentm eval sandbox batch --bench senior-swe \
       --repo senior-swe-bench-v2026.06/tasks \
       --registry pair-cn-guangzhou.cr.volces.com/opspai --prefix ssb --tag v1 \
       --model litellm-dsv4flash --scenario terminal_bench:arl \
       --agent-timeout 7200 --eval-timeout 900 -j 5
   ```

`reward.txt` is binary 0/1. An EMPTY reward.txt is not a 0 — it marks an
invalid trial (verifier-infra failure: judge unreachable, validation-agent
crash, etc.); the adapter records `invalid_trial` on those. The judge
pipeline (build + rubric + taste + optional validation agent, each an LLM
round-trip) runs many minutes, so the adapter floors the senior-swe eval
timeout at 2400s — a small `--eval-timeout` no longer truncates test.sh
mid-run (which was surfacing as a spurious `invalid_trial`). The per-task
`[verifier] timeout_sec` still extends `--eval-timeout` when larger, same as
LHTB.

#### Running single tasks via `harbor run` (direct mode)

For single-task runs, debugging, or experiments with supervisor interrupts,
use `harbor run` directly instead of the `agentm eval sandbox batch`
wrapper. This gives full control over env injection and timeout multipliers.

**Prerequisites:**

- ARL gateway running on a Kubernetes cluster with an internal registry
- Model profile configured in `~/.agentm/config.toml` (see below)
- `ARL_GATEWAY_URL` and `ARL_API_KEY` set in environment

**Model profile** (`~/.agentm/config.toml`):

```toml
[models.<profile-name>]
provider = "openai"
name = "<profile-name>"
model = "<model-id-on-proxy>"           # e.g. "azure-chat", "DeepSeek-V4-pro"
base_url = "${LITELLM_BASE_URL}/v1"     # litellm proxy endpoint
api_key = "${LITELLM_API_KEY}"
context_window = 262144
```

**Run command:**

```bash
ARL_GATEWAY_URL="$ARL_GATEWAY_URL" \
ARL_API_KEY="$ARL_API_KEY" \
uv run harbor run \
  -p <path-to-task-dir> \
  -a agentm_harbor:ExternalAgentMAgent \
  --environment-import-path arl.harbor:ArlEnvironment \
  -m <model-profile> \
  --ek gateway_url="$ARL_GATEWAY_URL" \
  --agent-timeout-multiplier 5 \
  --verifier-timeout-multiplier 5 \
  --no-delete -k 1 -y \
  --ae SSB_OVERRIDE_ALL_JUDGE_MODEL=openai/<judge-model> \
  --ae OPENAI_API_KEY="$LITELLM_API_KEY" \
  --ae OPENAI_BASE_URL="$LITELLM_BASE_URL" \
  --ve SSB_OVERRIDE_ALL_JUDGE_MODEL=openai/<judge-model> \
  --ve OPENAI_API_KEY="$LITELLM_API_KEY" \
  --ve OPENAI_BASE_URL="$LITELLM_BASE_URL"
```

**`--ae` vs `--ve` (critical):** `--ae` injects env vars into the **agent**
container; `--ve` injects into the **verifier** container. The rubric and
taste judges run inside the verifier, not the agent. If you only pass
credentials via `--ae`, judges skip with `skipped:no_api_key`. Always pass
judge credentials via **both** `--ae` and `--ve`.

**Verifier LLM model routing:** The verifier uses three independent model
slots, each with its own override env var and default:

| Slot | Override env var | Default | Purpose |
|---|---|---|---|
| Judge | `SSB_OVERRIDE_ALL_JUDGE_MODEL` | `anthropic/claude-sonnet-4-6` | Rubric scoring, taste evaluation, validation review |
| Classifier | `SSB_OVERRIDE_CLASSIFIER_MODEL` | `anthropic/claude-haiku-4-5` | Patch file classification (behavioral vs cosmetic) |
| Validation agent | `SSB_OVERRIDE_VA_MODEL` | (mini-swe-agent default) | Runs user stories against the agent's patch (25/50 tasks) |

The credential gate (`have_credentials()`) checks for any of:
`PORTKEY_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. If none is set in
the verifier env, all LLM judges skip.

Routing: with no `PORTKEY_API_KEY`, litellm dispatches directly by the
provider prefix in the model slug (`openai/...` → OpenAI endpoint,
`anthropic/...` → Anthropic endpoint). So `OPENAI_API_KEY` +
`OPENAI_BASE_URL` + `SSB_OVERRIDE_ALL_JUDGE_MODEL=openai/<model>` routes all
judge calls through your litellm proxy. But `SSB_OVERRIDE_CLASSIFIER_MODEL`
must be set **separately** — `ALL_JUDGE_MODEL` does not cover it.

Minimal `--ve` set for full scoring via a litellm/OpenAI-compatible proxy:

```bash
--ve OPENAI_API_KEY="$LITELLM_API_KEY" \
--ve OPENAI_BASE_URL="$LITELLM_BASE_URL" \
--ve SSB_OVERRIDE_ALL_JUDGE_MODEL=openai/<judge-model> \
--ve SSB_OVERRIDE_CLASSIFIER_MODEL=openai/<classifier-model>
```

Known issue: DeepSeek models do not reliably follow structured output
schemas. The taste judge expects top-level `{practice_alignment, relative_taste}`
but DeepSeek wraps them in `{scores: {...}}`, causing a Pydantic validation
error and `taste_judge_ok=0`. Use a model with strict structured output
support (GPT-4o, Claude) for the taste judge if full scoring is needed.

**Key flags:**

| Flag | Purpose |
|---|---|
| `-p <path>` | Local path to a single task directory |
| `-a agentm_harbor:ExternalAgentMAgent` | AgentM as the agent harness |
| `--environment-import-path arl.harbor:ArlEnvironment` | ARL sandbox backend |
| `-m <profile>` | Model profile name from `config.toml` |
| `--ek key=value` | Extra kwargs passed to `ArlEnvironment.__init__` |
| `--ae KEY=VALUE` | Env var injected into the agent container |
| `--ve KEY=VALUE` | Env var injected into the verifier container |
| `--no-delete` | Keep sandbox session alive after the run (for debugging) |
| `-k N` | Number of trials per task |

**Inspecting results:**

```bash
# Overall result
cat jobs/<job-dir>/result.json | python3 -m json.tool

# Detailed reward breakdown (correctness, rubric, taste, patch_bloat)
cat jobs/<job-dir>/<trial>/verifier/reward_details.json | python3 -m json.tool

# Agent patch diff
cat jobs/<job-dir>/<trial>/verifier/agent.patch

# Judge logs (diagnose skipped judges)
cat jobs/<job-dir>/<trial>/verifier/run_judge_rubric.stderr
cat jobs/<job-dir>/<trial>/verifier/run_judge_taste.stderr

# Functional test stdout
cat jobs/<job-dir>/<trial>/verifier/test-stdout.txt
```

**Verifier scoring:**

| Metric | Description |
|---|---|
| `correctness` | 1.0 if all functional tests pass |
| `rubric_judge_ok` | LLM judge score against rubric criteria |
| `taste_judge_ok` | LLM judge code style/taste score |
| `taste_patch_bloat` | `agent_sloc / oracle_sloc` ratio |
| `verifier_score` | Aggregate verifier score |

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
