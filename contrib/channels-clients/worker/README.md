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

## Out of scope (Phase 5b candidates)

* Load-balanced worker pools.
* Restart resumption (session_key survival across worker crash).
* Capability matching beyond plain scenario string equality.
* Per-worker resource quotas / health reporting back to the gateway.
