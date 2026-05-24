# Architecture

llmharness has two layers stacked on top of AgentM:

```
┌──────────────────────────────────────────────────────────────────┐
│  distill/   (offline)                                            │
│    cli.py · oracle.py · causal.py · gt.py · binding.py · ...     │
├──────────────────────────────────────────────────────────────────┤
│  adapter + audit children   (live)                               │
│    adapters/agentm.py                                            │
│    audit/extractor/   audit/auditor/   audit/registry.py         │
│    replay/  (sidecar JSONL writer)                               │
├──────────────────────────────────────────────────────────────────┤
│  AgentM substrate (core.abi, core.runtime — provided by parent)  │
└──────────────────────────────────────────────────────────────────┘
```

This doc covers both layers and how they connect through the replay
sidecar.

---

## 1. Components

| Component | File | Job |
|---|---|---|
| **Adapter** | `adapters/agentm.py` | Subscribes to `TurnEndEvent` and `DecideTurnActionEvent` on the main agent's bus. Drives the two phases below. Persists results onto the session entry tree and (optionally) the replay sidecar. |
| **Extractor child** | `audit/extractor/` | Phase 1. Per-turn LLM-driven graph builder. Single terminal tool `submit_events` emits typed events with embedded `refs[]`; the witness layer validates entities + verbatim quotes against the source turns and unrolls refs into `Edge` records. Framing is a swappable prompt file under `audit/extractor/prompts/`. |
| **Auditor child** | `audit/auditor/` | Phase 2. Every k turns. Reads the live graph (events + edges + scenario findings + recent verdicts + continuation_notes from the prior firing). Tool surface is controlled by the **auditor profile** (default `minimal` = just `submit_verdict`; `with_drill_down` adds `get_event_detail` + `get_turn`). Framing is a swappable prompt file under `audit/auditor/prompts/`. |
| **Check registry** | `audit/registry.py` | Service published by the adapter under key `llmharness.audit_registry`. Scenario atoms register pure graph checks; the adapter snapshots a frozen `CheckContext` at every auditor firing and folds emitted `Finding`s into the auditor prompt as advisory signals. |
| **Reference checks** | `extensions/check_*.py` | Three reference §11 atoms: `check_repeated_actions`, `check_open_branches`, `check_premature_conclusion`. Each is one file (MANIFEST + install), mounted with `--extension`. |
| **Replay writer** | `replay/record.py` | Appends one JSONL record per phase invocation to `<cwd>/.agentm/audit_replay/<root_session_id>.jsonl`. Best-effort; failures never break the live agent. |
| **Replay CLI** | `replay/cli.py` | `llmharness-replay {extractor,auditor} --record <line>` rebuilds the exact extension list + payload from a sidecar record and re-runs the child with a different provider/prompt for A/B. |
| **Distill binding** | `distill/binding.py` | §11 atom mounted on the **main agent**. At install time writes a `<root_session_id>.meta.json` sidecar next to the replay log, carrying `sample_id` + dataset coordinates. The labeler joins replay records to ground truth through this file. |
| **Distill labeler** | `distill/oracle.py`, `distill/cli.py` | Offline. For each auditor replay record, causally masks the graph to turn t, then runs two children: a GT-aware **oracle** (which findings matter given GT) and a GT-blind **rewriter** (can this selection be justified from the graph alone). Drops samples that fail the justifiability gate. |
| **Causal mask** | `distill/causal.py` | Pure function. `events: max(source_turns)≤t`; `edges: both endpoints kept AND max(src_turns|dst_turns)≤t`; `findings: all related_event_ids kept`. Load-bearing — without it the oracle uses post-hoc evidence and the student can't reproduce. |
| **Distill exporter** | `distill/export.py` | Labeled rows + extractor replay records → `sft/{extractor,auditor,dropped}.jsonl`. Extractor records pass through (already produced by the live teacher); auditor records carry only the rewriter-approved fields. |

---

## 2. Live runtime flow (Phase 1 + Phase 2)

```
                       ┌───────────────────────────┐
   TurnEndEvent ──────▶│ Adapter                   │
                       │  scan branch → window     │
                       │  spawn extractor child    │
                       └──────────┬────────────────┘
                                  │ child.prompt(payload)
                                  ▼
                       ┌───────────────────────────┐
                       │ Extractor child           │
                       │  submit_events([Event{    │
                       │     refs:[{dst, kind, ... │
                       │            cited_quote}]  │
                       │   }, ...])                │
                       └──────────┬────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │ witness layer (cited entities +    │
                │  quote must appear in src_turns)   │
                └─────────────────┬──────────────────┘
                                  ▼
            entries: audit_event, audit_edge, extractor_cursor
                                  +
                replay sidecar: ReplayRecord(phase="extractor", ...)

   (every k turns, after a successful extractor firing)
                                  │
                                  ▼
                       ┌───────────────────────────┐
                       │ Adapter                   │
                       │  rebuild graph from       │
                       │  entries; run checks;     │
                       │  spawn auditor child      │
                       └──────────┬────────────────┘
                                  ▼
                       ┌───────────────────────────┐
                       │ Auditor child             │
                       │  get_event_detail(ids)    │
                       │  get_turn(idx)            │
                       │  submit_verdict({surface, │
                       │     reminder_text, ...})  │
                       └──────────┬────────────────┘
                                  ▼
            entries: verdict
                                  +
                replay sidecar: ReplayRecord(phase="auditor", ...)
                                  │
                                  ▼  (only when surface_reminder=true
                                  │   AND adapter config enable_reminders=true)
   DecideTurnActionEvent ─────▶ Inject([reminder_msg])
```

Failure paths (typed entries, never silent):
`extractor_no_call`, `extractor_empty`, `extractor_error`,
`extractor_partial`, `audit_no_call`, `audit_error`. Every failure
also emits a `DiagnosticEvent` so the observability JSONL has
evidence. The cursor advances only on success or partial.

---

## 3. Distill flow (offline)

```
  Stage 0  live run with knobs flipped for clean collection
    enable_reminders=false               (auditor still fires;
                                          no main-agent side effect)
    + distill_binding extension          (writes <sid>.meta.json)
                  │
                  ▼  produces
    <cwd>/.agentm/audit_replay/<sid>.jsonl       ← ReplayRecords
    <cwd>/.agentm/audit_replay/<sid>.meta.json   ← sample_id + dataset

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
      <sid>.labels.jsonl  (one row per auditor firing,
                           with drop flag + rewriter-approved target)

  Stage 2  llmharness-distill export
    extractor records  ────▶ sft/extractor.jsonl
                              (replay output re-packaged as one
                               submit_events tool call)
    labeled rows       ────▶ sft/auditor.jsonl
                              (rewriter-approved fields only)
    dropped rows       ────▶ sft/dropped.jsonl  (audit trail)
```

---

## 4. Static dependency graph

Arrows mean "imports from / uses". Dashed arrows are runtime
service lookups (not static imports).

```
                    ┌──────────────────────┐
                    │ schema.py            │  Event/Edge/Finding/
                    │                      │  Verdict/Phase/Reminder
                    └──────────┬───────────┘
                               ▲
       ┌───────────────────────┼─────────────────────────┐
       │                       │                         │
       │                       │                         │
┌──────┴────────┐    ┌─────────┴───────┐         ┌───────┴────────────┐
│ audit/        │    │ audit/registry. │         │ replay/record.py   │
│  extractor/   │    │  py             │         │                    │
│  auditor/     │    │   AuditCheck    │         └────────────────────┘
│  _compose.py  │    │   Registry      │                  ▲
└──────┬────────┘    └─────────┬───────┘                  │
       │                       ▲                          │
       │                       │ (service lookup)         │
       │                       │                          │
       ▼                       │                          │
┌──────────────────────────────┴──────────────────────────┴─────────┐
│ adapters/agentm.py                                                │
│   ExtensionManifest "agentm"                                      │
│   - publishes the audit_registry service                          │
│   - subscribes to TurnEndEvent / DecideTurnActionEvent            │
│   - persists entries + writes replay sidecar                      │
└────────────────────────┬──────────────────────────────────────────┘
                         ▲ (mount via --extension)
                         │
              ┌──────────┴──────────┐
              │ extensions/check_*  │   reference checks
              │ (one file each)     │   register against registry
              └─────────────────────┘

  distill/  (offline, no live-adapter import)
  ────────
  cli.py ──▶ oracle.py ──▶ tools/engine.run_phase_standalone
              │   ├──▶ causal.py
              │   ├──▶ gt.py            (loads rca-shaped dataset)
              │   ├──▶ _submit_oracle.py   (§11 atom)
              │   ├──▶ _submit_rewriter.py (§11 atom)
              │   └──▶ prompts/{oracle,rewriter}.md
              └──▶ export.py ──▶ schema.py (re-uses Event/Edge shapes)

  binding.py  (§11 atom, mounted on main agent — NOT imported by
               the adapter; reads from env at install time only)
```

Key invariants:

* Adapter never imports anything under `distill/`. Distill is a
  pure offline consumer of the replay sidecar.
* Extractor and auditor children share no module-level state; they
  communicate through the session entry tree and the replay sidecar.
* `schema.py` is the only module both layers share. Anything that
  needs to round-trip between live and distill goes through it.
* The check registry is resolved through `api.get_service` at
  install time, not imported. That keeps reference checks single-
  file §11-compliant (no atom-to-atom imports).

---

## 5. `tools/` vs the rest — host-side drivers are not atoms

| Subpackage | Role | May import `agentm.core.runtime.*`? |
|---|---|---|
| `audit/`, `adapters/`, `replay/` (atoms + sidecar I/O), `extensions/check_*`, `distill/_submit_*` | §11 atoms or modules consumed by atoms — single-file contract, no `core.runtime.*` import, no atom-to-atom imports | No |
| `tools/` (`tools/engine.py`, `tools/prefix_replay.py`) | **Host-side drivers** that construct standalone AgentM sessions for offline replay / fork-and-replay | Yes — that is what `tools/` exists for |
| `replay/chained_fork.py` | Orchestration helper that combines `tools/engine` runs with `replay/` sidecar I/O; called by host code (e.g. rca eval), not by atoms | No direct `core.runtime.*` import — it composes through the `tools/` helpers |
| `distill/` (the labeler / exporter) | Offline CLIs; not loaded into a live session | Yes (drives standalone children via `tools/engine`) |

Structural rule: nothing under `llmharness/audit/`, `llmharness/adapters/`,
`llmharness/atoms/`, or `llmharness/extensions/` may import from
`llmharness.tools.*`. The boundary test
`tests/test_replay_engine_boundary.py` enforces this — `tools/` is the
explicit "host driver, not atom surface" marker. When a future workflow
needs to spawn standalone sessions offline, add the module under
`tools/`, not under `audit/` or `replay/`.
