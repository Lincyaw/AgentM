# llmharness

Claude Code plugin that runs an out-of-band "harness agent" against the main
agent's transcript: PostToolUse / Stop hooks fold turns into a stable event
stream; a background worker calls AgentM to detect drift; UserPromptSubmit
injects a one-line reminder when confidence is high.

Three repos work together:

- **AgentM** — pluggable agent SDK. Hosts the `harness_monitor` scenario we
  drive via its CLI.
- **llmharness** *(this repo)* — Claude Code plugin + `llmharness` Python package.
  Hooks, file protocol (`.harness/`), worker loop, AgentM subprocess bridge.
- **rca-autorl** — RL training. Imports `llmharness` to reuse the event-stream
  schema and trace store as PRM training data.

## Layout

```
.claude-plugin/plugin.json     plugin manifest
hooks/hooks.json               PostToolUse + Stop + UserPromptSubmit
scripts/                       hook entrypoints + worker daemon installer
src/llmharness/                Python package (schema, store, worker, bridge)
```

The `harness_monitor` scenario lives as a sibling of this directory at
`<AgentM-root>/scenarios/harness_monitor/manifest.yaml` so `agentm --scenario
harness_monitor` resolves it directly.

## Single-call architecture

```
PostToolUse / Stop hook  →  inbox/<sid>.jsonl   (delta, non-blocking)
                                   ↓
                            background worker
                                   ↓
              harness_monitor (turns + history → events + verdict)
                                   ↓
                            events/<sid>.jsonl
                            pending_reminders/<sid>.json
                                   ↓
UserPromptSubmit hook   ←  pending_reminders/<sid>.json  (inject + clear)
```

One LLM call per tick. Throttling (skip the call entirely when the delta is
empty or below threshold) lives in the worker, not in the scenario.

## AgentM bridge

`src/llmharness/agentm_bridge.py` shells out to `agentm <prompt> --scenario
harness_monitor` with a single JSON payload as the user prompt. Configure via env:

| var                       | meaning                                            |
|---------------------------|----------------------------------------------------|
| `LLMHARNESS_PROVIDER`     | `rule` (default) or `agentm`                       |
| `LLMHARNESS_AGENTM_BIN`   | path to the `agentm` CLI                           |
| `LLMHARNESS_AGENTM_CWD`   | AgentM checkout root (so `scenarios/<name>/` resolves) |
| `LLMHARNESS_AGENTM_MODEL` | provider model id, e.g. `claude-sonnet-4-6`        |
| `LLMHARNESS_AGENTM_TIMEOUT` | seconds (default 120)                            |
| `LLMHARNESS_DISTILL_DIR`  | if set, dump `(input, output)` pairs per call      |

Switching from a commercial model to a distilled 8B is a one-flag change:
update `LLMHARNESS_AGENTM_MODEL` and `LLMHARNESS_AGENTM_BIN`'s provider
config; the plugin doesn't move.

## Install

1. Install the plugin in Claude Code (drop this repo into a plugins dir or
   reference it via plugin settings — Claude Code auto-loads `hooks/hooks.json`).
2. The `harness_monitor` scenario already lives at `<AgentM-root>/scenarios/`
   alongside this package; nothing to copy.
3. Start the worker:

   ```bash
   LLMHARNESS_PROVIDER=agentm \
   LLMHARNESS_AGENTM_CWD=/path/to/AgentM \
   LLMHARNESS_AGENTM_MODEL=claude-sonnet-4-6 \
   LLMHARNESS_DISTILL_DIR=$PWD/.harness/distill \
     scripts/install.sh
   ```

   The default provider is `rule` (no LLM call) — useful for smoke-testing
   the file protocol before wiring in AgentM.

## Distillation

When `LLMHARNESS_PROVIDER=agentm` and `LLMHARNESS_DISTILL_DIR` is set, every
tick appends one JSONL line:

```
$LLMHARNESS_DISTILL_DIR/<sid>.monitor.jsonl   {input, output}
```

`output` carries `{events, verdict}` — the same shape the scenario emits.
This is the SFT corpus for the eventual 8B replacement.
