---
name: gateway-probe
description: Debug the AgentM single-process gateway by connecting a synthetic wire client and dumping the outbound frames it emits. Use when a chat client (terminal/feishu) "doesn't see" something — a slash command, a tool, status, a reply — or when diagnosing wire/session_ready/command-catalog issues. Shows exactly what the gateway puts on the wire, bypassing the TUI.
---

# gateway-probe

When a client symptom is "X doesn't show up" (a command like `/compact` missing
from autocomplete, no status, a missing reply), **do not theorize layer by
layer** — connect a synthetic peer to the *actual running gateway* and look at
the bytes on the wire. That single observation collapses most of the search
space: if the frame isn't on the wire it's a gateway/session bug; if it is, it's
the client.

## How to run

Always through the repo venv (editable gateway code):

```bash
uv run python .agents/skills/gateway-probe/probe.py [options]
```

Find the running gateway's address first:

```bash
pgrep -af "agentm gateway"   # look for the --bind ws://… / unix://… value
```

## Recipes

```bash
# Full picture: what does the gateway emit for a fresh session + a real prompt?
uv run python .agents/skills/gateway-probe/probe.py \
    --connect ws://127.0.0.1:8770 --chat-id probe-1 --send "hi" --summary

# Is the command catalog on the wire? (session_ready carries command_names)
uv run python .agents/skills/gateway-probe/probe.py \
    --connect ws://127.0.0.1:8770 --chat-id probe-2 --send "hi" --only session_ready

# Does anything arrive on connect, before the first message?
uv run python .agents/skills/gateway-probe/probe.py --listen-on-connect 3 --listen 3
```

Key flags: `--connect` (ws:// or unix:///), `--chat-id` (FRESH id → new session;
reuse → hit an existing/persisted one), `--send` (repeatable; a non-command
triggers a real LLM turn — costs tokens), `--only <kind>` filter, `--summary`,
`--token` (omit for `--bind-allow-anonymous`/`--bind-allow-uid`).

## What to look at

- `session_ready.command_names` — the slash-command catalog the client builds
  autocomplete from. Missing/empty here ⇒ client autocomplete is empty by design.
- `KIND_COUNTS` (`--summary`) — which event kinds reached the wire at all.
- A frame present here but absent in the client ⇒ client bug; absent here ⇒
  gateway/session bug (e.g. emitted before the outbound sink was wired).

## Notes

- Throwaway sessions are `/end`-ed automatically (disable with `--no-end`).
- This is debug tooling under `.agents/`, not shipped product code — keep
  gateway/wire probing here rather than re-deriving the `WireClient` +
  `resolve_connect` scaffolding ad hoc each time.
- The probe reuses the real `agentm.gateway.client.WireClient` and
  `agentm.gateway.client_cli.resolve_connect`, so it exercises the genuine wire
  path, not a reimplementation.
