# agentm-harbor

AgentM as a Harbor external agent. Agent runs locally, tool calls (bash, file
I/O) route through Harbor's `BaseEnvironment` into the sandbox. The sandbox
backend is pluggable (Docker, Modal, E2B, GKE, Daytona, etc.) -- this adapter
is backend-agnostic, it only talks to the `BaseEnvironment` interface.
Trajectory is managed by AgentM's own observability layer (ClickHouse / OTLP).

## Architecture

```
Host                              Harbor sandbox (any backend)
┌──────────────────────┐          ┌──────────────────┐
│  AgentM session      │          │  task environment │
│  (model inference)   │          │                   │
│         │            │          │                   │
│    harbor_ops atom   │─exec()──>│  bash commands    │
│         │            │─upload()>│  file writes      │
│         │            │<download─│  file reads       │
│         │            │          │                   │
│  observability ──> ClickHouse   │  verifier (test.sh)
└──────────────────────┘          └──────────────────┘
```

## Install

```bash
uv tool install harbor \
    --with "agentm-harbor @ file://path/to/contrib/scenarios/harbor" \
    --with "agentm @ file://path/to/AgentM"
```

## Usage

```bash
harbor trial start \
    -p path/to/task \
    -a "agentm_harbor:ExternalAgentMAgent" \
    -m <model-profile>            # from ~/.agentm/config.toml
    --agent-timeout 300
```

Model profile is resolved from `~/.agentm/config.toml` (same as `agentm
--model <name>`). API keys, base URLs, and provider settings are read from
the profile. You can also pass them explicitly via `--ae`:

```bash
harbor trial start \
    -p path/to/task \
    -a "agentm_harbor:ExternalAgentMAgent" \
    -m doubao \
    --ae AGENTM_API_KEY=... \
    --ae AGENTM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

## Trajectory

Trajectories go to ClickHouse via OTLP (AgentM's default observability path).
Inspect with:

```bash
agentm trace messages --session <session-id> --format text
agentm trace tools    --session <session-id> --format text
agentm trace usage    --session <session-id>
```

Session ID is logged at agent start:
`agentm-external: session <id> started`.

## Scenario manifest

Harbor uses the `arl:harbor` variant in `contrib/scenarios/arl/`. The
tool overlay is shared with the default ARL scenario:

```
contrib/scenarios/arl/
  overlays/base.yaml       # ARL operations atom (agent_env backend)
  overlays/harbor.yaml     # Harbor operations atom (harbor_ops)
  overlays/tools.yaml      # shared tool atoms (file_tools, bash, etc.)
  manifest.yaml            # arl (default) = base + tools
  manifest.harbor.yaml     # arl:harbor = harbor + tools
  manifest.harness.yaml    # arl:harness = base + tools + llmharness
```

## Components

| File | Role |
|---|---|
| `external_agent.py` | `BaseAgent` subclass; sets up env vars, creates AgentM session with `scenario="harbor_external"` |
| `harbor_ops.py` | Operations atom; `HarborBashOperations` wraps `environment.exec()`, `HarborResourceWriter` wraps `upload_file()`/`download_file()` |
| `scenarios/harbor_external/manifest.yaml` | Atom list and config for the session |
