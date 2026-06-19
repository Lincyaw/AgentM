---
name: self-debug
description: Inspect your own running session to debug yourself — see the terminal UI the user sees (tui_snapshot), read your own live trace, and mine historical traces. Use when the user reports a bug in how you behaved, a rendering/interaction glitch in the terminal client, or asks you to figure out what went wrong in this or a past session.
---

# self-debug

You are running inside the gateway. Your behaviour, your tool calls, and
the UI the user sees are all observable from where you sit. When something
looks wrong, look at the evidence before guessing.

## Principle

Three independent surfaces, each for a different question:

| Question | Surface |
|----------|---------|
| "What does the user actually see on screen?" | `tui_snapshot` tool |
| "What did *I* just do this session?" | `agentm trace … --latest` over `bash` |
| "What happened in some past session?" | `agentm trace index` → drill in |

Prefer evidence over speculation. The session trace is the source of
truth for your own actions; the conversation is a lossy view of it.

## See the UI (tui_snapshot)

> Scope: `tui_snapshot` exists **only for the terminal client**. A
> chat-channel deployment (e.g. Feishu/Lark over the gateway) has no TUI
> frame and no such tool — skip this section there and rely on the trace
> surfaces below. See the `deployment-awareness` skill for what your
> deployment actually exposes.

The terminal client renders a frame the agent never sees directly. When
the user runs **`/dump`** in the client, that frame is written to a file
(`$AGENTM_TUI_DUMP`, default `/tmp/agentm-tui-dump.txt`). Read it with the
`tui_snapshot` tool.

- The dump is **user-triggered**. If `tui_snapshot` reports no file or a
  stale frame (it prints the frame's age), ask the user to press `/dump`
  again, then retry.
- Pass `raw: true` to keep ANSI colour codes when debugging a *styling*
  bug; the default strips them for clean layout text.
- Pass `tail: N` for just the last N lines of a long transcript.

Use this for rendering glitches, layout overflow, wrong colours, a button
that isn't showing, an approval card that looks off — anything you cannot
infer from the conversation text alone.

## Read your own trace

Every turn lands in `.agentm/observability/<session_id>.jsonl`. Query it
with the `agentm trace` CLI via `bash` — never parse the JSONL directly.
`--latest` selects the most recent session, which is **you** (the current
turn is not flushed yet, but all prior turns are).

```bash
agentm trace messages --latest            # your conversation trajectory
agentm trace tools --latest               # every tool call + args + result
agentm trace tools --latest --tool bash   # filter to one tool
agentm trace turns --latest               # per-turn token/stop-reason
agentm trace usage --latest               # token economics
```

Caveat: if you have spawned subagents, `--latest` may pick a child
session. Disambiguate with `index` (below) and select by `session_id`.

## Mine historical traces

`index` is the only directory-granular verb — it lists every session file
once. Select with `jq`, then drill into each `--file`.

```bash
# list all sessions (path, trace_id, session_id, parent, purpose, scenario)
agentm trace index --format ndjson

# all files in one logical trace, then extract tool calls from each
TID=<trace_id>
agentm trace index --format ndjson \
  | jq -r --arg t "$TID" 'select(.trace_id==$t)|.path' \
  | while read -r f; do agentm trace tools --file "$f" --format ndjson; done
```

For full atomic-command reference and composition idioms, see the
`trace-analysis` skill.

## Workflow

1. Reproduce-from-evidence first: `agentm trace tools --latest` to see
   exactly what you did, `tui_snapshot` to see what the user saw.
2. Form a hypothesis grounded in that evidence.
3. If the bug is historical, `index` → select the session → drill in.
4. Only then propose or apply a fix.
