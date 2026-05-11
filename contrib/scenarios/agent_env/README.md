# `agent_env` scenario

Same tool surface as `general_purpose`, but **every bash / read / write / edit
call runs inside an ARL agent-env sandbox** (Kubernetes pod), not on the host.

Implemented by replacing the `operations_local` atom with
[`operations_agent_env`](../../extensions/operations_agent_env.py) — both atoms
satisfy the same `BashOperations` + `FileOperations` protocols, so tool atoms
downstream are unchanged.

## When to use this scenario

- Sandboxed code execution (untrusted code, exploit research, evolution
  trials that shouldn't touch the host FS).
- RL trajectory collection — ARL auto-snapshots after every step, so a
  rollout can roll back to any intermediate state.
- Pinning the agent to a reproducible toolchain image (pool's executor
  container) rather than whatever happens to be on the operator's host.

If you don't need isolation, prefer `general_purpose` — it has no external
dependencies.

## Prerequisites

1. **ARL Gateway reachable from the host.** The Python SDK calls a plain HTTP
   endpoint; that's it. NodePort, port-forward, ingress — any of them work.
   Default `http://localhost:8080`.

2. **A ready WarmPool** whose name matches `pool_ref` in the manifest (or env
   var `AGENTM_AGENT_ENV_POOL_REF`). The pool's pods must include a `sidecar`
   container; if you're starting from scratch the agent-env repo ships a
   sample at `examples/local-test/manifests/warmpool.yaml`.

3. **Python deps.** From the AgentM repo root:

   ```bash
   uv sync --extra agent-env
   ```

   This pulls the `arl-env` SDK alongside the rest of AgentM. Without it the
   atom raises a clear error at install time.

## Configuration

All optional except `pool_ref` (which must come from either the manifest or
the env var):

| Field         | Env var                          | Default                  |
| ------------- | -------------------------------- | ------------------------ |
| `pool_ref`    | `AGENTM_AGENT_ENV_POOL_REF`      | — (required)             |
| `gateway_url` | `AGENTM_AGENT_ENV_GATEWAY_URL`   | `http://localhost:8080`  |
| `namespace`   | `AGENTM_AGENT_ENV_NAMESPACE`     | `default`                |
| `work_dir`    | —                                | `/workspace`             |
| `timeout`     | —                                | none (no per-step limit) |

Atom config wins when both the manifest and the env var are set, so a
scenario can pin one knob while leaving the rest to the operator.

## End-to-end bring-up on minikube

This is the verified recipe used during the integration smoke test (~15 min
from cold). Adapt it for kind, k3d, or a real cluster as needed — only steps
1 and 2 are minikube-specific.

### 0. Host prerequisites

```bash
docker info        # daemon must be running
go version         # Go 1.25+ (only needed to build the ARL images)
which minikube kubectl helm   # all three required
```

### 1. Start minikube

```bash
minikube start \
  --cpus=4 --memory=8192 \
  --driver=docker \
  --kubernetes-version=v1.32.0
```

If you already have a minikube profile, either reuse it or pass
`-p arl-env` to keep this cluster isolated.

### 2. Build the four ARL images inside minikube's docker daemon

```bash
eval $(minikube docker-env)            # point docker CLI at minikube
cd ../agent-env                        # the agent-env repo (sibling to AgentM)

docker build -t arl-operator:latest       -f Dockerfile.operator       .
docker build -t arl-sidecar:latest        -f Dockerfile.sidecar        .
docker build -t arl-gateway:latest        -f Dockerfile.gateway        .
docker build -t arl-executor-agent:latest -f Dockerfile.executor-agent .

docker images | grep ^arl-               # sanity check: four images present
```

> Building inside minikube's daemon avoids a registry push. After this step
> the cluster can pull `arl-*:latest` via `imagePullPolicy: IfNotPresent`.

### 3. Helm install the operator + gateway

The default Helm chart enables ClickHouse / Redis / Prometheus / Grafana /
Alertmanager / VMServiceScrape — most of those images live in a private
registry (`pair-diag-cn-guangzhou.cr.volces.com/...`) that won't be
reachable from a fresh minikube. Override them off:

```yaml
# /tmp/arl-values-minikube.yaml
gateway:
  enabled: true
  service:
    type: NodePort
    nodePort: 30080

redis:        { enabled: false }
clickhouse:   { enabled: false }
alerting:     { enabled: false }
prometheus:   { enabled: false }
grafana:      { enabled: false }
apiPriority:  { enabled: false }

metrics:
  service:        { enabled: false }
  serviceMonitor: { enabled: false }
```

```bash
helm upgrade --install arl-operator ../agent-env/charts/arl-operator \
  --namespace arl --create-namespace \
  --set crds.install=true \
  -f /tmp/arl-values-minikube.yaml \
  --wait --timeout=3m

kubectl get pods -n arl
# arl-operator-...          1/1 Running
# arl-operator-gateway-...  1/1 Running
```

### 4. Create a WarmPool

```bash
kubectl apply -f ../agent-env/examples/local-test/manifests/warmpool.yaml
kubectl wait --for=condition=Ready pod \
  -l arl.infra.io/pool=python-39-std --timeout=180s

kubectl get pods -l arl.infra.io/pool=python-39-std
# python-39-std-xxxxx   2/2 Running   (one or more idle warm pods)
```

If pod creation fails with
`spec.containers[0].volumeMounts[3].mountPath: Invalid value: "/arl-bin": must be unique`
on an older `agent-env` checkout, you need the deep-copy fix in
`pkg/controller/warmpool_controller.go` (already on `agent-env` `main` —
`git pull` and rebuild the operator image).

### 5. Expose the gateway to the host

Pick **one** of the two options.

| Option              | Pros                          | Cons                                                       |
| ------------------- | ----------------------------- | ---------------------------------------------------------- |
| `minikube service`  | one command, no extra process | the minikube node IP is only routable from the host        |
| `kubectl port-forward` | works from anywhere on host | needs to stay running for the lifetime of the agentm run   |

```bash
# Option A — NodePort URL (most convenient on a single-host workflow)
export AGENTM_AGENT_ENV_GATEWAY_URL=$(
  minikube service arl-operator-gateway -n arl --url | head -1
)

# Option B — long-running port-forward
kubectl port-forward -n arl svc/arl-operator-gateway 18080:8080 &
export AGENTM_AGENT_ENV_GATEWAY_URL=http://localhost:18080

curl -fsS "$AGENTM_AGENT_ENV_GATEWAY_URL/healthz"   # → ok
```

### 6. Wire up AgentM's environment

```bash
# From the AgentM repo root.
uv sync --extra agent-env                     # installs arl-env SDK

export AGENTM_AGENT_ENV_POOL_REF=python-39-std

# Optional overrides — only needed if you don't want to bake them into the
# scenario manifest:
# export AGENTM_AGENT_ENV_NAMESPACE=default
# export AGENTM_AGENT_ENV_GATEWAY_URL=...   # already set in step 5
```

The full env-var contract:

| Variable                          | Used by                              | Default                  |
| --------------------------------- | ------------------------------------ | ------------------------ |
| `AGENTM_AGENT_ENV_POOL_REF`       | `operations_agent_env` install       | — (required if no manifest config) |
| `AGENTM_AGENT_ENV_GATEWAY_URL`    | `operations_agent_env` install       | `http://localhost:8080`  |
| `AGENTM_AGENT_ENV_NAMESPACE`      | `operations_agent_env` install       | `default`                |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | AgentM provider atoms           | — (one is required)      |

Manifest-config and env-vars are merged at install time, with **atom config
winning** — useful when the scenario pins a pool but you want to retarget the
gateway from the shell. (`AGENTM_AGENT_ENV_POOL_REF` only takes effect when
`pool_ref` is unset in the manifest.)

### 7. Run a one-shot prompt

```bash
uv run agentm --scenario agent_env \
  "Print uname -a and pwd, then tell me the hostname the sandbox reports."
```

What to look for:

- The agent calls `bash` two or more times via the same sandbox session.
- The reported hostname matches one of the warmpool pods:
  `kubectl get pods -l arl.infra.io/pool=python-39-std`.
- After the command exits, that pod is gone (`delete_sandbox()` fires on
  `SessionShutdownEvent`) and the WarmPool reconciler spawns a fresh idle
  one.

For trace-level evidence:

```bash
ls ~/.agentm/sessions/ | tail -1   # most-recent session log path
grep '"name": "bash"' ~/.agentm/sessions/<that-path>/*.jsonl | head
```

### 8. Teardown

```bash
kubectl delete warmpool python-39-std
helm uninstall arl-operator -n arl
minikube delete                  # or: minikube stop, to keep state for next run
```

## Session lifecycle

The atom creates **one** sandbox at install time and deletes it on the
`SessionShutdownEvent`, so an `agentm` invocation maps 1:1 to an ARL session.
Every `tool_bash` call inside that session reuses the same pod — file edits,
git state, environment changes all persist across tool calls (and survive
`tool_propose_change` reloads). Set `keep_alive` / idle-timeout knobs on the
WarmPool side if you need cross-invocation persistence.

## Known limitations

- The `arl` Python SDK is synchronous; `BashOperations.exec` wraps the call
  in `asyncio.to_thread`. The `on_data` streaming callback fires *after* the
  step completes (one blob), not chunk-by-chunk. For true streaming hook up
  ARL's `InteractiveShellClient` (WebSocket) — left as a future enhancement.
- `signal: asyncio.Event` (caller-side cancellation) isn't pushed into the
  sandbox: ARL `execute` runs the step to completion. The flag is only used
  to mark `timed_out=True` on the returned `ExecResult`.
- File ops route through `cat` / `test` / `ls` shell steps, so each read is
  one HTTP round-trip. Fine for typical agent loops; not appropriate for
  bulk file enumeration.
