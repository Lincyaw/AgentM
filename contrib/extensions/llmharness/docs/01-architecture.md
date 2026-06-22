# Architecture

llmharness has two layers stacked on top of AgentM, plus offline
tooling that consumes the replay sidecar:

```
┌──────────────────────────────────────────────────────────────────┐
│  tools/  (offline, dev-checkout only)                            │
│    replay/   distill/   aggregate/   extensions/   eval/         │
├──────────────────────────────────────────────────────────────────┤
│  src/llmharness/  (core library, shipped in wheel)               │
│    atom.py  (main adapter)                                       │
│    agents/extractor/   agents/auditor/                           │
│    replay/  (record I/O)   eval/telbench/                        │
│    schema.py                                                     │
├──────────────────────────────────────────────────────────────────┤
│  AgentM substrate (core.abi, core.runtime — provided by parent)  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 1. Components

### Core library (`src/llmharness/`)

| Component | File | Job |
|---|---|---|
| **Main atom** | `atom.py` | The orchestrator. Subscribes to `TurnEndEvent`, `DecideTurnActionEvent`, `SessionShutdownEvent` on the main agent's bus. Manages `CumulativeAuditState` (event-sourced from session entries). Spawns extractor/auditor children at configured intervals. Persists results and queues reminders. MANIFEST name: `"llmharness"`. |
| **Schema** | `schema.py` | All shared data types (`Event`, `Edge`, `Finding`, `Verdict`, `Phase`, `Reminder`, `ExternalRef`) plus session-entry-type constants. The public contract for downstream consumers. |
| **Context index** | `context_index.py` | Derived LSP-style view over the visible trajectory: turns, entities, observations, claims, candidates, obligations, contract events, and weak links. This is the auditor's default context surface. |
| **Extractor child** | `agents/extractor/` | Phase 1. Builds an incremental context index using record/link index ops. Uses 6 tools (`upsert_record`, `upsert_link`, `delete_record`, `delete_link`, `reset_extraction`, `finalize_extraction`). Witness validation ensures link citations appear in source text. |
| **Auditor child** | `agents/auditor/` | Phase 2. Reads `CONTEXT_INDEX` by default and emits a `Verdict` via `submit_verdict`. Prompt variants can still request legacy graph or combined context for A/B. |
| **Replay record** | `replay/record.py` | `ReplayRecord` dataclass + read/write helpers. One JSONL line per phase invocation at `<cwd>/.agentm/audit_replay/<session_id>.jsonl`. |
| **TELBench eval** | `eval/telbench/` | Offline evaluation harness for trajectory-error localization: loads TELBench dataset, runs extractor+auditor per instance, scores span-level P/R/F1/FEA. |
| **Agent path resolvers** | `agents/__init__.py` | `extractor_scenario()` / `auditor_scenario()` return absolute paths to the child scenario directories. |

### Offline tooling (`tools/`)

| Component | File | Job |
|---|---|---|
| **Replay CLI** | `tools/replay/cli.py` | `llmharness-replay {extractor,auditor,chain,list,agent-from-reminder}`. Rebuilds extension list + payload from a sidecar record and re-runs with different provider/prompt for A/B. |
| **Replay engine** | `tools/replay/engine.py` | `run_phase_standalone` — constructs a standalone AgentM session for offline child execution. May import `agentm.core.runtime.*`. |
| **Chain replay** | `tools/replay/chain.py` | Bulk-replay every record in order, threading cumulative index state across firings. |
| **Prefix replay** | `tools/replay/prefix_replay.py` | Branch a session at turn t, resume with reminder seeded. |
| **Reminder seed** | `tools/replay/reminder_seed.py` | §11 atom mounted on resumed sessions to inject a recorded reminder. |
| **Distill CLI** | `tools/distill/cli.py` | `llmharness-distill {label,export}`. Drives the oracle + rewriter pipeline. |
| **Distill oracle** | `tools/distill/oracle.py` | For each auditor firing: causal-mask the graph to turn t, run GT-aware oracle + GT-blind rewriter, drop unjustifiable samples. |
| **Causal mask** | `tools/distill/causal.py` | Pure function. Filters events/edges/findings to those causally available at turn t. |
| **GT loader** | `tools/distill/gt.py` | Loads ground-truth labels from the dataset. |
| **SFT exporter** | `tools/distill/export.py` | Labeled rows → `sft/{extractor,auditor,dropped}.jsonl`. |
| **Distill binding** | `tools/distill/binding.py` | §11 atom mounted on the main agent during distill runs. Writes `<session_id>.meta.json` next to the replay sidecar. |
| **Aggregate CLI** | `src/llmharness/aggregate/cli.py` | `llmharness-aggregate {replay,one,sessions}`. Folds sidecar(s) or ClickHouse sessions into per-case directories. |
| **Reference checks** | `tools/extensions/check_*.py` | Three §11 atoms: `check_premature_conclusion`, `check_repeated_actions`, `check_open_branches`. Mounted with `--extension`. |

---

## 2. Live runtime flow

```
                       ┌───────────────────────────┐
   TurnEndEvent ──────▶│ atom.py (main adapter)    │
                       │  hydrate CumulativeState   │
                       │  compute due intervals     │
                       │  spawn extractor child     │
                       └──────────┬────────────────┘
                                  │ child.prompt(payload)
                                  ▼
                       ┌───────────────────────────┐
                       │ Extractor child            │
                       │  index via node/edge tools │
                       │  finalize_extraction       │
                       └──────────┬────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │ witness layer (cited entities +    │
                │  quote must appear in src_turns)   │
                └─────────────────┬──────────────────┘
                                  ▼
            entries: audit_index_op, extractor_cursor
            sidecar: ReplayRecord(phase="extractor", ...)

   (every k turns, after a successful extractor firing)
                                  │
                                  ▼
                       ┌───────────────────────────┐
                       │ atom.py                    │
                       │  derive context index      │
                       │  spawn auditor child       │
                       └──────────┬────────────────┘
                                  ▼
                       ┌───────────────────────────┐
                       │ Auditor child              │
                       │  submit_verdict({surface,  │
                       │     reminder_text, ...})   │
                       └──────────┬────────────────┘
                                  ▼
            entries: verdict
            sidecar: ReplayRecord(phase="auditor", ...)
                                  │
                                  ▼  (when surface_reminder=true
                                  │   AND enable_reminders=true)
   DecideTurnActionEvent ─────▶ Inject([reminder_msg])
```

Failure paths (typed session entries, never silent):
`extractor_no_call`, `extractor_empty`, `extractor_error`,
`extractor_partial`, `audit_no_call`, `audit_error`. Every failure
also emits a `DiagnosticEvent` on the observability JSONL.

### Async vs sync mode

- **sync** (`mode: "sync"`): extractor/auditor children run directly in
  the `TurnEndEvent` handler. Simpler but blocks the main agent loop.
- **async** (`mode: "async"`, default): uses `asyncio.Queue` + a
  background worker task. Queue guarantees extractor-before-auditor
  ordering. On shutdown, drains with a configurable timeout.

### CumulativeAuditState

Event-sourced in-memory state in `atom.py`. Rebuilt on startup from
session entries via `hydrate_from_session_log()`. Maintains:

- `ops: list[IndexOp]` — the append-only index op log
- `cursor_last_turn_index` — last turn consumed by extractor
- `recent_verdicts: deque[dict]` (maxlen 5) — recent auditor verdicts
- `last_continuation_notes` — carried into the next auditor firing
- Lazy `index_view()` via `fold_index(ops)` → `(events, edges, phases)`

---

## 3. Distill flow (offline)

```
  Stage 0  live run with distill knobs
    enable_reminders=false               (auditor still fires;
                                          no main-agent side effect)
    + tools/distill/binding.py atom      (writes <sid>.meta.json)
                  │
                  ▼  produces
    .agentm/audit_replay/<sid>.jsonl       ← ReplayRecords
    .agentm/audit_replay/<sid>.meta.json   ← sample_id + dataset

  Stage 1  llmharness-distill label
    for each auditor replay record at turn t:
      causal_mask(graph, edges, findings, trajectory, t)
                  │
                  ▼
      oracle child   (sees GT)    submit_oracle_label(...)
                  │
                  ▼  selected_finding_indices, matched_event_ids
      rewriter child (no GT)      submit_rewrite(
                                    justifiable_from_graph,
                                    reminder_text,
                                    drop_reason)
                  │
                  ▼
      <sid>.labels.jsonl  (one row per auditor firing)

  Stage 2  llmharness-distill export
    extractor records  ────▶ sft/extractor.jsonl
    labeled rows       ────▶ sft/auditor.jsonl
    dropped rows       ────▶ sft/dropped.jsonl  (audit trail)
```

---

## 4. Static dependency graph

Arrows mean "imports from / uses". Service lookups are at install time
only (no module-level coupling between atoms).

```
                    ┌──────────────────────┐
                    │ schema.py            │  Event / Edge / Finding /
                    │                      │  Verdict / Phase / Reminder
                    └──────────┬───────────┘
                               ▲
       ┌───────────────────────┼─────────────────────────┐
       │                       │                         │
┌──────┴────────┐    ┌─────────┴───────┐         ┌───────┴────────────┐
│ agents/       │    │ atom.py         │         │ replay/record.py   │
│  extractor/   │    │  CumulativeState│         │                    │
│  auditor/     │    │  install()      │         └────────────────────┘
└──────┬────────┘    └─────────┬───────┘                  ▲
       │                       │                          │
       │        (spawns child sessions)                   │
       ▼                       │                          │
  manifest.yaml ◄──────────────┘                          │
  (scenario for child)                                    │
                                                          │
  ┌───────────────────────────────────────────────────────┘
  │ (tools/ layer — offline, may use core.runtime)
  │
  ├── tools/replay/     cli.py ──▶ runner.py ──▶ engine.py
  │                                                ▲
  ├── tools/distill/    cli.py ──▶ oracle.py ──▶ engine
  │                      │
  │                      ├──▶ causal.py
  │                      ├──▶ gt.py
  │                      └──▶ export.py ──▶ schema.py
  │
  ├── tools/aggregate/  cli.py ──▶ collector.py ──▶ writer.py
  │
  └── tools/extensions/ check_*.py  (§11 atoms, no cross-imports)
```

Key invariants:

* **atom.py never imports from `tools/`**. The tools layer is a pure
  offline consumer of the replay sidecar.
* **Extractor and auditor children share no module-level state**; they
  communicate through the session entry tree and the replay sidecar.
* **`schema.py` is the only module both layers share**. Anything that
  needs to round-trip between live and offline goes through it.
* **No atom-to-atom imports** (§11 contract). Atoms communicate through
  the service registry (`api.set_service` / `api.get_service`).

---

## 5. `src/` vs `tools/` boundary

| Layer | Role | May import `agentm.core.runtime.*`? |
|---|---|---|
| `src/llmharness/` (atoms, schema, replay record) | §11 atoms or modules consumed by atoms — single-file contract, no `core.runtime.*` import, no atom-to-atom imports | No |
| `tools/` (replay engine, distill driver, aggregate) | Host-side drivers that construct standalone AgentM sessions for offline operations | Yes — that is what `tools/` exists for |

Structural rule: nothing under `src/llmharness/agents/` may import from
`tools/`. The boundary test enforces this — `tools/` is the explicit
"host driver, not atom surface" marker.

---

## 6. TELBench evaluation (`eval/telbench/`)

Standalone evaluation harness for trajectory-error localization on the
TELBench dataset. Lives in `src/llmharness/eval/telbench/` (shipped in
the wheel).

- **adapter.py**: Loads TELBench JSONL, converts spans to synthetic
  `AgentMessage` objects.
- **runner.py**: Per-instance eval driver. Creates
  `CumulativeAuditState.fresh()`, walks turns, fires extractor/auditor
  at configured cadence. Maps `matched_event_ids` → `source_turns` →
  predicted span indices.
- **scoring.py**: Span-level precision / recall / F1 + FEA (First Error
  Accuracy). Macro-aggregation across instances.
- **cli.py**: `llmharness-eval telbench` with filters for difficulty,
  answer-status, concurrency, and provider/model overrides.
