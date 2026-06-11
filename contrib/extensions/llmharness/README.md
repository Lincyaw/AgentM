# llmharness

LLM-as-harness for AgentM: a two-phase cognitive audit pipeline
(extractor + auditor) that supervises a main agent, plus offline
tooling for replay, distill (SFT data generation), evaluation, and
case aggregation.

Version 0.6.0 &middot; Python &ge;3.12 &middot; `uv` only

## What it does

An agent running an investigation (or any multi-turn task) can drift —
follow dead leads, forget earlier findings, converge prematurely.
llmharness watches the agent's turns and periodically:

1. **Extracts** a logic-flow graph from the conversation (nodes =
   semantic events like hypotheses / actions / decisions; edges =
   causal or referential links with witness citations).
2. **Audits** the graph for reasoning faults (drift, blind spots,
   premature conclusions) and optionally injects a one-line reminder
   into the main agent's next turn.

Both the extractor and auditor run as AgentM child sessions with their
own prompts, tools, and provider configs — the main agent's tool
surface is unchanged.

---

## Quick start

Mount llmharness onto any AgentM session:

```bash
agentm --extension llmharness.atom --scenario <your-scenario> -p "..."
```

Or add it to a scenario manifest:

```yaml
# contrib/scenarios/<your-scenario>/manifest.yaml
extensions:
  # ... your scenario's atoms ...
  - module: llmharness.atom
    config:
      mode: async                      # or "sync"
      extractor_interval_turns: 1      # extract every turn
      audit_interval_turns: 3          # audit every 3 turns
      enable_auditor: true
      enable_reminders: true           # false = opinions-only, no injection
```

For distill data collection (reminders off, with sample-id binding):

```bash
LLMHARNESS_DISTILL_SAMPLE_ID=<id> \
LLMHARNESS_DISTILL_DATASET=<dataset.jsonl> \
  agentm --extension llmharness.atom \
         --extension llmharness.distill.binding \
         --scenario rca ...

llmharness-distill label  --replay-dir .agentm/audit_replay \
                          --dataset <dataset.jsonl> --out ./labels
llmharness-distill export --labels ./labels \
                          --replay-dir .agentm/audit_replay --out ./sft
```

---

## Repository layout

```
contrib/extensions/llmharness/
├── src/llmharness/              # Core library (shipped in wheel)
│   ├── __init__.py              #   Public API (re-exports from schema)
│   ├── schema.py                #   Data types + entry-type constants
│   ├── atom.py                  #   Main extension atom (the orchestrator)
│   ├── state.py                 #   CumulativeAuditState (event-sourced graph state)
│   ├── agents/                  #   Child session scenarios
│   │   ├── __init__.py          #     Path resolvers (extractor_scenario, auditor_scenario)
│   │   ├── extractor/           #     Extractor child: graph builder
│   │   │   ├── manifest.yaml    #       Scenario manifest for extractor child
│   │   │   ├── graph.py         #       Graph ops model, fold, phase merge
│   │   │   ├── context.py       #       Context injection atom
│   │   │   ├── prompt.py        #       Prompt templates
│   │   │   └── tools.py         #       Witness validation, ExtractionState, tool builders
│   │   └── auditor/             #     Auditor child: verdict emitter
│   │       ├── manifest.yaml    #       Scenario manifest for auditor child
│   │       ├── context.py       #       Context injection atom
│   │       ├── prompt.py        #       10 prompt variants (minimal, bench, telbench, trajectory_*)
│   │       └── tools.py         #       submit_verdict tool
│   ├── replay/                  #   Replay record I/O
│   │   └── record.py            #     ReplayRecord dataclass, read/write helpers
│   └── eval/                    #   Offline evaluation
│       └── telbench/            #     TELBench trajectory-error evaluation
│           ├── adapter.py       #       Dataset loader + span→message converter
│           ├── runner.py        #       Per-instance eval driver
│           ├── scoring.py       #       P/R/F1/FEA scoring
│           └── cli.py           #       `llmharness-eval telbench` entry point
│
├── tools/                       # Offline tooling (NOT in wheel, dev-checkout only)
│   ├── replay/                  #   Replay CLI + offline replay engine
│   │   ├── cli.py               #     `llmharness-replay` entry point
│   │   ├── runner.py            #     Replay orchestrator
│   │   ├── engine.py            #     Standalone session runner
│   │   ├── chain.py             #     Bulk-replay with cumulative graph state
│   │   ├── prefix_replay.py     #     Branch + resume from a specific turn
│   │   ├── fork_tree.py         #     Fork-tree experiment helpers
│   │   ├── reminder_seed.py     #     Reminder-seeding atom for prefix-replay
│   │   └── offline.py / offline_driver.py
│   ├── distill/                 #   Distill pipeline (SFT data generation)
│   │   ├── cli.py               #     `llmharness-distill` entry point
│   │   ├── oracle.py            #     GT-aware labeling
│   │   ├── causal.py            #     Causal masking (graph → turn t)
│   │   ├── gt.py                #     Ground-truth loader
│   │   ├── export.py            #     SFT JSONL exporter
│   │   ├── binding.py           #     Distill-binding atom (meta sidecar writer)
│   │   ├── dpo_pairs.py         #     DPO pair construction
│   │   ├── rl_prompts.py        #     RL prompt templates
│   │   ├── signals.py           #     Signal extraction helpers
│   │   ├── _submit_oracle.py    #     Oracle tool atom
│   │   └── _submit_rewriter.py  #     Rewriter tool atom
│   ├── aggregate/               #   Case aggregation for review + export
│   │   ├── cli.py               #     `llmharness-aggregate` entry point
│   │   ├── collector.py         #     Per-case data collection
│   │   ├── case.py              #     Case data model
│   │   └── writer.py            #     Directory layout writer
│   ├── extensions/              #   Reference audit checks (§11 atoms)
│   │   ├── check_premature_conclusion.py
│   │   ├── check_repeated_actions.py
│   │   └── check_open_branches.py
│   └── eval/telbench/           #   (Legacy eval location)
│
├── docs/                        # Design and reference docs
├── references/papers/           # Related academic papers
├── pyproject.toml               # Package config (hatchling build)
└── eval.db                      # Local eval database
```

Key boundary: `src/llmharness/` is the pip-installable library;
`tools/` is host-side offline tooling that may import
`agentm.core.runtime.*` and is only available from a dev checkout
(not shipped in the wheel).

---

## How it works

### The main atom (`atom.py`)

The `llmharness` atom (MANIFEST name: `"llmharness"`) installs onto the
main agent's `EventBus` and drives the two-phase pipeline:

| Event | What happens |
|---|---|
| `TurnEndEvent` | Computes whether extractor / auditor are due based on interval configs. Spawns child sessions as needed. Persists results to the session entry tree. |
| `DecideTurnActionEvent` | If a verdict with `surface_reminder=true` is pending, returns `Inject([reminder_msg])` so the kernel re-opens the loop with the reminder. |
| `SessionShutdownEvent` | Drains the async worker queue (async mode) with a configurable timeout. |

No tools are exposed to the main agent — the audit runs silently
alongside the conversation.

### Config knobs

| Knob | Default | Effect |
|---|---|---|
| `mode` | `"async"` | `"sync"` fires children in the event handler; `"async"` uses a background worker queue |
| `extractor_interval_turns` | 1 | Run extractor every N turns |
| `audit_interval_turns` | 3 | Run auditor every N turns (only after successful extraction) |
| `enable_auditor` | `true` | Set `false` to run extractor-only |
| `enable_reminders` | `true` | Set `false` for opinions-only (verdicts recorded but not injected) |
| `extractor_prompt` / `auditor_prompt` | `"default"` / `"minimal"` | Named prompt variant or absolute file path |
| `extractor_provider` / `auditor_provider` | `null` | Override the LLM provider for child sessions |
| `audit_summary_threshold` | 30 | Degrade witness fields when event count exceeds this |

### Extractor child

Runs as an AgentM child session with scenario
`agents/extractor/manifest.yaml`. The extractor maintains an
**event-sourced logic-flow graph** via an append-only op log:

- **Graph ops**: `NodeUpsert`, `NodeDelete`, `EdgeUpsert`, `EdgeDelete`
- **Event kinds**: `task`, `hyp` (hypothesis), `act` (action), `dec`
  (decision), `concl` (conclusion)
- **Edge kinds**: `data` (causal/data dependency), `ref` (referential)
- **Witness validation**: every edge must cite entities or a verbatim
  quote that appears in the source turn text. Invalid edges are dropped.
- **Tools**: `upsert_node`, `upsert_edge`, `delete_node`, `delete_edge`,
  `reset_extraction`, `finalize_extraction` (terminal)

The graph is rebuilt deterministically via `fold_graph(ops)` — pure
fold over the op sequence.

### Auditor child

Runs as an AgentM child session with scenario
`agents/auditor/manifest.yaml`. Reads the current graph snapshot
(events + edges + phases + findings + continuation notes) and emits a
verdict via the `submit_verdict` tool:

```python
Verdict(
    surface_reminder=True,       # whether to inject
    reminder_text="...",         # the one-liner for the main agent
    continuation_notes=["..."],  # passed to the NEXT auditor firing
    matched_event_ids=[2, 7],    # events that justify the verdict
)
```

10 prompt variants available (selected via `auditor_prompt` config):
`minimal`, `bench`, `telbench`, `trajectory`, `trajectory_cascade`,
`trajectory_coverage`, `trajectory_dual`, `trajectory_receipt`,
`trajectory_reflect`, `trajectory_sniper`.

### Cumulative state

`CumulativeAuditState` (in `atom.py`) is the adapter's in-memory graph
state. It is **event-sourced** from the session entry tree — on startup,
`hydrate_from_session_log(branch)` replays all persisted
`audit_graph_op` / `verdict` / `extractor_cursor` entries. This means
the graph survives session restarts and can be rebuilt from the log alone.

---

## Public API

Importable from `llmharness` top-level only — everything else is
internal:

| Symbol | Source |
|---|---|
| `Event`, `EventKind`, `Edge`, `EdgeKind` | `schema.py` |
| `Finding`, `Phase`, `Verdict`, `Reminder` | `schema.py` |

Replay I/O (from `replay.record`):

| Symbol | Purpose |
|---|---|
| `ReplayRecord` | One line per phase firing in the sidecar |
| `iter_records(path)` | Lazy iterator over sidecar JSONL |
| `read_records(path, phase, turn_index)` | Eager filtered read |
| `write_record(path, record)` | Append one record |

---

## CLI entry points

| Script | Module | Purpose |
|---|---|---|
| `llmharness-eval` | `llmharness.eval.telbench.cli` | TELBench offline evaluation |
| `llmharness-replay` | `tools/replay/cli.py` | Replay recorded firings with different provider/prompt |
| `llmharness-distill` | `tools/distill/cli.py` | SFT data generation pipeline |
| `llmharness-aggregate` | `tools/aggregate/cli.py` | Case aggregation for review |

### Replay subcommands

| Subcommand | What it does |
|---|---|
| `llmharness-replay extractor` / `auditor` | Replay one recorded phase with overrides (A/B bisection) |
| `llmharness-replay chain` | Bulk-replay every record; threads cumulative graph state |
| `llmharness-replay list` | Index records by phase / turn / status / latency |
| `llmharness-replay agent-from-reminder` | Branch a main-agent session at turn t, seed with recorded reminder |

---

## Sequence diagram: live supervision

```
TurnEndEvent
  │
  ├─ extractor_due? ──▶ spawn extractor child
  │                        │
  │                        ▼
  │                     tools: upsert_node/edge, finalize_extraction
  │                        │
  │                        ▼
  │                     persist: graph ops → session entries
  │                              replay record → sidecar JSONL
  │
  ├─ auditor_due? ──▶ spawn auditor child (graph + findings + notes)
  │                        │
  │                        ▼
  │                     tool: submit_verdict
  │                        │
  │                        ▼
  │                     persist: verdict → session entries + sidecar
  │                     if surface_reminder → queue Reminder
  │
  ▼
DecideTurnActionEvent
  │
  └─ pending reminder? ──▶ Inject([reminder_msg])
                              kernel re-opens the loop
```

---

## Schema stability

`src/llmharness/schema.py` is the public contract for downstream
consumers (e.g. rca-autorl). Breaking changes bump the package version
in `pyproject.toml`. Current version: v4 wire shape. Pre-v4 records are
not supported.

See [docs/02-schemas.md](docs/02-schemas.md) for the full schema
reference (in-memory types, session entries, replay sidecar, distill
labels, SFT JSONL).

---

## Docs index

| File | When to read it |
|---|---|
| [docs/01-architecture.md](docs/01-architecture.md) | Components, dependency graph, runtime data flow |
| [docs/02-schemas.md](docs/02-schemas.md) | All wire types, entry types, sidecar + SFT JSONL shapes |
| [docs/03-distill-recipe.md](docs/03-distill-recipe.md) | End-to-end SFT data generation recipe |
| [docs/04-extending.md](docs/04-extending.md) | Adding audit checks; adapting to non-rca datasets |
| [docs/05-profiles-and-prompts.md](docs/05-profiles-and-prompts.md) | Prompt variants + provider overrides for A/B |
| [docs/06-case-aggregation.md](docs/06-case-aggregation.md) | Per-case directory layout for review |
| [docs/07-prefix-replay.md](docs/07-prefix-replay.md) | Iterate on auditor/reminder without full re-run |
| [docs/08-running-modes.md](docs/08-running-modes.md) | Decoupling extractor / auditor / reminder injection |
| [docs/09-extractor-strategy-iteration.md](docs/09-extractor-strategy-iteration.md) | Extractor prompt/model strategy iteration workflow |
