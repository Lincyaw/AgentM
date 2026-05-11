# agentm-worker

Agent-side worker process for the AgentM channels gateway. Phase 5a of
the gateway/client process-split work.

## What it does

`agentm-worker` connects to a running `agentm-gateway --bind unix://…`
as a wire peer of kind `agent_worker`. The gateway forwards user
inbound messages to the worker; the worker drives an in-process
`AgentSession` per chat and ships the assistant's replies, approval
requests, and turn-complete signals back over the wire.

This is the same `AgentSession`-holding code that lived inside the
gateway in Phases 1-4 — just extracted into its own process so the
gateway can be restarted without dropping live sessions (worker stays
up), and so multiple gateways or scenarios can share a worker pool in
future phases.

## Usage

```bash
# 1) Start the gateway (no in-process worker).
agentm-gateway \
    --bind unix:///tmp/gw.sock \
    --no-inproc-worker \
    --scenario general_purpose \
    --cwd /path/to/workspace

# 2) Start a worker that advertises general_purpose.
agentm-worker \
    --connect unix:///tmp/gw.sock \
    --scenario general_purpose \
    --cwd /path/to/workspace

# 3) Start a chat client.
agentm-terminal --connect unix:///tmp/gw.sock
```

All sessions on a single worker share its `--cwd`. To serve multiple
project roots, start more workers.

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Clean shutdown (SIGTERM, server BYE, EOF on --no-input). |
| 1    | Generic runtime error. |
| 2    | Bad argument or configuration. |
| 4    | Auth rejected by the gateway. |
| 6    | SIGINT (Ctrl-C). |
| 7    | Cannot connect to the gateway socket. |

## Multi-agent (A2A) example

Phase 6 ships an opt-in `tool_peer_send` atom that lets one agent
delegate to another worker via the gateway. Layout for a
`researcher → coder → reviewer` chain:

```
                      +--------------------+
   user (terminal) -->|     gateway        |
                      +--------------------+
                       |        |        |
                       v        v        v
                  worker-     worker-   worker-
                  researcher  coder     reviewer
```

```bash
# 1) Gateway: allow `tool_peer_send` to be /atom:install'd from chat.
cat > /tmp/gw.yaml <<'YAML'
commands:
  atoms:
    enabled: true
    allow: ["tool_peer_send"]
YAML

agentm-gateway \
    --bind unix:///tmp/gw.sock \
    --no-inproc-worker \
    --scenario general_purpose \
    --config /tmp/gw.yaml \
    --max-a2a-hops 10 \
    --cwd /path/to/workspace

# 2) Three workers. Each gets an auto-generated peer_id of the form
#    "worker-<8-hex>"; record those from the worker startup log lines
#    (`worker ready peer_id=worker-…`) and pass them as the `to`
#    argument when calling peer_send from the LLM. (A future revision
#    will add an explicit --peer-id flag for stable identifiers.)
agentm-worker --connect unix:///tmp/gw.sock \
    --scenario general_purpose --cwd /path/to/workspace &
agentm-worker --connect unix:///tmp/gw.sock \
    --scenario general_purpose --cwd /path/to/workspace &
agentm-worker --connect unix:///tmp/gw.sock \
    --scenario general_purpose --cwd /path/to/workspace &

# 3) Terminal client. From the chat, the researcher agent calls
#    /atom:install tool_peer_send, then uses peer_send to delegate:
#       peer_send(to="worker-<id>", content="implement X")
#       peer_send(to="worker-<id>", content="review the diff")
agentm-terminal --connect unix:///tmp/gw.sock
```

Approval requests emitted by `worker-coder` are routed back to the
terminal chat (the *root* of the dispatch chain) via the
`root_session_key` propagated on every forwarded envelope; the
user's click flows back to `worker-coder` through the existing
synthetic-channel path — see `.claude/designs/client-server-architecture.md`
§7.7 + §8 Phase 6.

The gateway-level hop limit (`--max-a2a-hops`, default 10) breaks
runaway delegation loops with a `hop_limit_exceeded` error sent
back to the originating worker.

## Out of scope (Phase 5b candidates)

* Load-balanced worker pools.
* Restart resumption (session_key survival across worker crash).
* Capability matching beyond plain scenario string equality.
* Per-worker resource quotas / health reporting back to the gateway.
