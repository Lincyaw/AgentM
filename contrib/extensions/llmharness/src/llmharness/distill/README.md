# Distill — data collection for training a small (~4B) harness model

This subpackage produces the SFT training data needed to fine-tune a small
model into the role llmharness currently fills (extractor + auditor). It
runs **entirely offline** against the replay sidecar (`<cwd>/.agentm/
audit_replay/<root_session_id>.jsonl`) that the live adapter already
writes by default.

The pipeline is deliberately small — minimal-extension audit children,
plain prompts, two stages.

---

## 1. Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 0  Clean trajectory collection (live)                            │
│   agentm --extension llmharness.adapters.agentm \                      │
│          --extension llmharness.distill.binding \                      │
│          --scenario rca ...                                            │
│   Adapter config: enable_reminders=false                               │
│     → extractor + auditor still fire (data still collected),           │
│       but verdicts do NOT influence the main agent                     │
│   Binding atom writes <root_session_id>.meta.json with sample_id       │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1  Offline oracle labeling                                       │
│   llmharness-distill label \                                           │
│        --replay-dir <cwd>/.agentm/audit_replay \                       │
│        --dataset /path/to/rca/data.jsonl \                             │
│        --out  ./distill_labels \                                       │
│        --oracle-provider <module> [--rewriter-provider <module>]       │
│                                                                        │
│   For each auditor replay record (turn t):                             │
│     a. causal-mask graph/edges/findings to events with                 │
│        max(source_turns) ≤ t                                           │
│     b. Stage A — oracle child (sees GT):                               │
│          input  = {snapshot, gt}                                       │
│          output = submit_oracle_label(                                 │
│            selected_finding_indices, matched_event_ids,                │
│            rationale_with_gt, continuation_notes)                      │
│     c. Stage B — rewriter child (NO GT, no rationale_with_gt):         │
│          input  = {snapshot, selected_finding_indices,                 │
│                    matched_event_ids}                                  │
│          output = submit_rewrite(reminder_text,                        │
│                                  justifiable_from_graph: bool,         │
│                                  drop_reason: str)                     │
│     d. If justifiable_from_graph=False → record DROP                   │
│        else → write distill label JSONL line                           │
│                                                                        │
│   Extractor records pass through unchanged — no oracle needed.         │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2  SFT export                                                    │
│   llmharness-distill export \                                          │
│        --labels ./distill_labels \                                     │
│        --out    ./sft \                                                │
│        --phase  both        # or extractor / auditor                   │
│                                                                        │
│   Produces:                                                            │
│     ./sft/extractor.jsonl                                              │
│     ./sft/auditor.jsonl                                                │
│   Each line: {sample_id, turn_index, input:{system,user},              │
│               target:{tool_calls:[...]}}.                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Design decisions (load-bearing)

### D1 — Two-stage oracle, not one

Stage A's prompt explicitly carries GT and tells the oracle to **decide
which findings matter** given GT. Stage B's prompt sees zero GT and is
asked **"can this selection be justified from graph alone?"**. If no,
the sample is dropped. This protects the student from learning a
GT-conditioned selection function.

### D2 — Causal masking is non-negotiable

Stage A is given only the slice of the graph that existed at turn t
(`max(source_turns) ≤ t` for events, both endpoints kept for edges,
all `related_event_ids` kept for findings). Without this, the oracle
uses post-hoc evidence to label the verdict, and the student — which
only sees past turns at inference — learns a function it cannot
reproduce.

### D3 — Minimal extensions on the audit children

Per the project rule "smallest correct surface", oracle and rewriter
children run with only:

- `observability` — events to JSONL
- `otel_tracing` — spans
- `operations_local` — required by substrate freeze
- `system_prompt` — the short, focused prompt for that stage
- one submit tool — `submit_oracle_label` or `submit_rewrite`

No cards, no skills, no FS / bash. The student model will see the same
surface at inference time, so training matches deployment.

### D4 — Negative samples are intra-sample, not synthetic

A run that produces (say) 200 findings across 50 firings will naturally
contain firings where the oracle says "flag none". Those firings ARE the
negative samples — they teach the model "I see findings but choose not
to fire". No synthetic negatives.

### D5 — Selection is the learnable signal; phrasing is templated

The auditor SFT target is mostly `(surface_reminder, matched_event_ids)`.
`reminder_text` is rewriter-produced and follows the methodological
vocabulary (verify hypothesis, close branch, avoid repeated action) —
which is graph-derivable. We do not ask the student to invent free-text
critique.

### D6 — Sample-id binding via a sidecar meta file

The §11 atom `llmharness.distill.binding` reads
`LLMHARNESS_DISTILL_SAMPLE_ID` (and `LLMHARNESS_DISTILL_DATASET`) at
install time and writes `<cwd>/.agentm/audit_replay/<root_session_id>.meta.json`.
The labeler joins replay records → GT through this meta file. We do
NOT add a field to `ReplayRecord` (no schema churn).

---

## 3. File layout

```
distill/
├── README.md          this file
├── __init__.py
├── binding.py         §11 atom + read_sample_meta helper
├── gt.py              GroundTruth dataclass + JSONL loader (rca shape)
├── causal.py          CausalSnapshot + causal_mask()
├── prompts/
│   ├── oracle.md
│   └── rewriter.md
├── _submit_oracle.py    §11 atom: submit_oracle_label tool
├── _submit_rewriter.py  §11 atom: submit_rewrite tool
├── oracle.py          two-stage orchestrator (uses run_phase_standalone)
├── export.py          replay + labels → SFT JSONL
└── cli.py             llmharness-distill {label, export}
```

---

## 4. SFT JSONL record shape

Each line in `./sft/{extractor,auditor}.jsonl`:

```json
{
  "phase": "auditor",
  "sample_id": "ts0-mysql-corrupt-kwx8n5",
  "root_session_id": "abc123",
  "turn_index": 12,
  "input": {
    "system": "<system prompt verbatim>",
    "user":   "<json payload string, verbatim>"
  },
  "target": {
    "tool_calls": [
      {"name": "submit_verdict",
       "arguments": {"surface_reminder": true, "reminder_text": "...",
                     "matched_event_ids": [3, 7], "continuation_notes": [],
                     "cited_cards": []}}
    ]
  },
  "meta": {
    "fault_type": "NetworkCorrupt",
    "fault_category": "NetworkChaos",
    "drop": false
  }
}
```

Dropped samples (rewriter said `justifiable_from_graph=false`) are
emitted to a sibling `dropped.jsonl` so we can audit why they were
rejected — not silently lost.
