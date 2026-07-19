# agentm-harbor

AgentM as a Harbor external agent. Agent runs locally, tool calls (bash, file
I/O) route through Harbor's `BaseEnvironment` into the sandbox. The sandbox
backend is pluggable (Docker, Modal, E2B, GKE, Daytona, etc.) -- this adapter
is backend-agnostic, it only talks to the `BaseEnvironment` interface.
Trajectory is managed by AgentM's selected `TrajectoryStore`; OTLP is an
independent diagnostics channel.

## Architecture

```
Host                              Harbor sandbox (any backend)
┌──────────────────────┐          ┌──────────────────┐
│  AgentM session      │          │  task environment │
│  (model inference)   │          │                   │
│         │            │          │                   │
│  host-injected ports │─exec()──>│  bash commands    │
│         │            │─upload()>│  file writes      │
│         │            │<download─│  file reads       │
│         │            │          │                   │
│ TrajectoryStore      │          │  verifier (test.sh)
│ OTLP (optional) ──> collector   │
└──────────────────────┘          └──────────────────┘
```

## Install

Tested with Harbor **0.19.0**.

```bash
uv tool install "harbor==0.19.0" \
    --with "agentm-harbor @ file://path/to/contrib/scenarios/harbor" \
    --with "agentm @ file://path/to/AgentM"
```

## Usage

Single-task runs with full control over env injection and timeout multipliers.

**Prerequisites:**

- ARL gateway running on a Kubernetes cluster with an internal registry
- Model profile configured in `~/.agentm/config.toml`
- `ARL_GATEWAY_URL` and `ARL_API_KEY` set in environment

**Model profile** (`~/.agentm/config.toml`):

```toml
[models.<profile-name>]
provider = "openai"
model = "<model-id-on-proxy>"           # e.g. "azure-chat", "DeepSeek-V4-pro"
base_url = "https://<litellm-host>/v1"
api_key_env = "OPENAI_API_KEY"
context_window = 262144
```

Harbor's `-m <profile-name>` selects this AgentM profile. Per-trial `--ae`
values are passed to an explicit resolver snapshot; the adapter never mutates
process-wide environment variables, so concurrent trials cannot exchange
provider or trajectory configuration.

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
  --ek image_registry=<internal-registry-host:port> \
  --ek image_tag=<content-hash> \
  --agent-timeout-multiplier 5 \
  --verifier-timeout-multiplier 5 \
  --no-delete -k 1 -y \
  --ae SSB_OVERRIDE_ALL_JUDGE_MODEL=openai/<judge-model> \
  --ae SSB_OVERRIDE_CLASSIFIER_MODEL=openai/<classifier-model> \
  --ae OPENAI_API_KEY="$LITELLM_API_KEY" \
  --ae OPENAI_BASE_URL="$LITELLM_BASE_URL" \
  --ve SSB_OVERRIDE_ALL_JUDGE_MODEL=openai/<judge-model> \
  --ve SSB_OVERRIDE_CLASSIFIER_MODEL=openai/<classifier-model> \
  --ve OPENAI_API_KEY="$LITELLM_API_KEY" \
  --ve OPENAI_BASE_URL="$LITELLM_BASE_URL"
```

### `--ae` vs `--ve`

`--ae` injects env vars into the **agent** container; `--ve` injects into the
**verifier** container. The rubric and taste judges run inside the verifier,
not the agent. If you only pass credentials via `--ae`, judges skip with
`skipped:no_api_key`. Always pass judge credentials via **both** flags.

### Verifier LLM model routing

The SSB verifier uses three independent model slots:

| Slot | Override env var | Default | Purpose |
|---|---|---|---|
| Judge | `SSB_OVERRIDE_ALL_JUDGE_MODEL` | `anthropic/claude-sonnet-4-6` | Rubric scoring, taste evaluation, validation review |
| Classifier | `SSB_OVERRIDE_CLASSIFIER_MODEL` | `anthropic/claude-haiku-4-5` | Patch file classification (behavioral vs cosmetic) |
| Validation agent | `SSB_OVERRIDE_VA_MODEL` | (mini-swe-agent default) | Runs user stories against the agent's patch (25/50 tasks) |

`SSB_OVERRIDE_ALL_JUDGE_MODEL` covers rubric + taste + validation review.
`SSB_OVERRIDE_CLASSIFIER_MODEL` is **independent** — `ALL_JUDGE_MODEL` does
not cover it. If you forget to set it, the classifier falls back to
`anthropic/claude-haiku-4-5` and fails with `classifier API error` when no
Anthropic key is available.

The credential gate (`have_credentials()` in `ssb_lib/llm_utils.py`) checks
for any of: `PORTKEY_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. At
least one must be set in the verifier env or all LLM judges skip.

Routing: with no `PORTKEY_API_KEY`, litellm dispatches by the provider prefix
in the model slug (`openai/...` uses `OPENAI_API_KEY` + `OPENAI_BASE_URL`).

### Verifier scoring

| Metric | Description |
|---|---|
| `correctness` | 1.0 if all functional tests pass |
| `rubric_judge_ok` | LLM judge score against rubric criteria (skipped if task has no rubric) |
| `taste_judge_ok` | LLM judge code style/taste score |
| `taste_patch_bloat` | `agent_sloc / oracle_sloc` ratio |
| `verifier_score` | Aggregate verifier score |

### Inspecting results

```bash
# Overall result
cat jobs/<job-dir>/result.json | python3 -m json.tool

# Detailed reward breakdown
cat jobs/<job-dir>/<trial>/verifier/reward_details.json | python3 -m json.tool

# Agent patch diff
cat jobs/<job-dir>/<trial>/verifier/agent.patch

# Judge logs (diagnose skipped judges)
cat jobs/<job-dir>/<trial>/verifier/run_judge_rubric.stderr
cat jobs/<job-dir>/<trial>/verifier/run_judge_taste.stderr

# Functional test stdout
cat jobs/<job-dir>/<trial>/verifier/test-stdout.txt
```

### Key flags

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
| `--ek image_registry=<host:port>` | Internal registry with pre-built task images |
| `--ek image_tag=<tag>` | Image tag (content hash from first build) |

## Trajectory

Set `AGENTM_TRAJECTORY_DSN` for PostgreSQL or
`AGENTM_TRAJECTORY_DIR` for JSONL. Without either setting, the normal AgentM
configuration resolver selects the project/user trajectory backend. OTLP and
ClickHouse contain diagnostic events and spans, not a second trajectory copy.
Inspect the selected trajectory store with:

```bash
agentm trace messages --session <session-id> --format text
agentm trace tools    --session <session-id> --format text
agentm trace usage    --session <session-id>
```

Session ID is logged at agent start:
`agentm-external: session <id> started`.

## Scenario manifest

The adapter owns and packages
`contrib/scenarios/harbor/scenario.yaml`. Harbor injects its
`BaseEnvironment` as explicit `EnvironmentOperations` and `ResourceWriter`
ports before the SDK session is created. The manifest selects only reusable
AgentM policy/tool atoms. It is intentionally separate from source-tree CLI
scenarios: installing the Harbor extra must not change or shadow the default
SDK composition.

## Human interrupt

`agentm-interrupt <session-id> <message>` queues the operator message with
immediate priority and cancels the active model/tool request with the
`submit_interrupt` reason. The command succeeds only after the running session
acknowledges the message.

## Components

| File | Role |
|---|---|
| `external_agent.py` | `BaseAgent` subclass; sets up env vars, creates AgentM session, reports token counts |
| `harbor_ops.py` | Host bindings; `HarborEnvironmentOperations` wraps sandbox identity and execution, while `HarborResourceWriter` wraps transfer APIs |
| `human_interrupt.py` | Presenter-owned Unix socket that uses the public immediate-prompt API |
| `interrupt_cli.py` | Sends one interrupt and verifies the session acknowledgement |
