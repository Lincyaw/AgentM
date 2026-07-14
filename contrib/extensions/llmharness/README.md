# llmharness

`llmharness` is an AgentM extension package for online cognitive audit.
The supported online entrypoint is:

```text
llmharness.atom
```

It supervises a main AgentM session by spawning extractor and auditor child
sessions at configured turn intervals. The main agent does not receive any new
tools.

## Mounting

From the CLI:

```bash
uv run agentm --scenario <scenario> \
  -e llmharness.atom:'{"mode":"sync"}' \
  -p "..."
```

From a scenario manifest:

```yaml
extensions:
  - module: llmharness.atom
    config:
      mode: sync
      extractor_interval_turns: 10
      audit_interval_turns: 10
      enable_auditor: true
      enable_reminders: true
```

The current RCA harness manifest is
`contrib/scenarios/rca/manifest.harness.sync.yaml`.

## Runtime Shape

`llmharness.atom` registers these AgentM events:

| Event | Role |
|---|---|
| `BeforeAgentStartEvent` | Adds the system-reminder contract to the main agent system prompt. |
| `TurnEndEvent` | Runs the extractor and auditor cadence. |
| `DecideTurnActionEvent` | Injects queued auditor reminders into the next loop action. |
| `SessionShutdownEvent` | Drains pending audit work and forces a final audit pass. |

The pipeline is:

1. Extractor child session indexes the visible trajectory prefix.
2. Extractor tools write validated index ops to `.agentm/audit_ops/`.
3. Parent atom folds those ops into `CumulativeAuditState` and persists
   `llmharness.audit_index_op` session entries.
4. Auditor child session reads the folded index plus derived `context_index`.
5. Auditor calls `submit_verdict`.
6. Parent atom persists `llmharness.verdict` and queues a reminder when the
   verdict asks to surface one.

## Important Paths

| Path | Purpose |
|---|---|
| `src/llmharness/atom.py` | Online AgentM extension entrypoint. |
| `src/llmharness/schema.py` | Public event, edge, verdict, and entry-type dataclasses. |
| `src/llmharness/state.py` | Event-sourced cumulative audit state. |
| `src/llmharness/trajectory_utils.py` | Shared trajectory serialization helpers. |
| `src/llmharness/agents/auditor/` | Auditor child scenario, context atom, and `submit_verdict` tool. |
| `src/llmharness/agents/analyst/` | Analyst child scenario and trace tools. |
| `src/llmharness/agents/tel/` | TEL child scenario, tools, and workflow. |

Eval-specific code (TELBench runner, aggregate CLI, offline audit pipeline)
has been moved to ``agentm_eval`` — this package contains only runtime code.
