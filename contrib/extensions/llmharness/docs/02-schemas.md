# Schemas

Authoritative shapes for every artifact that crosses a process or
file boundary. Source of truth for the in-memory dataclasses is
`src/llmharness/schema.py`; this doc adds the on-disk schemas
(replay sidecar, meta sidecar, SFT JSONL) and explains how the
pieces relate.

---

## 1. In-memory contracts (`schema.py`)

All dataclasses are `frozen=True` and ship `to_dict` / `from_dict`.

### `Event`

```python
Event(
  id: int,                        # per-firing fresh-numbered (1, 2, ...)
  kind: EventKind,                # task | hyp | evid | act | dec | concl
  summary: str,
  source_turns: list[int],        # turn indices this event was extracted from
)
```

V3 short-form kinds: `task`, `hyp`, `evid`, `act`, `dec`, `concl`.
The v2 long forms (`hypothesis`, `evidence`, …) and `REFLECTION`
are gone.

### `Edge`

```python
Edge(
  src: int,                       # event id
  dst: int,                       # event id
  kind: EdgeKind,                 # data | ref
  reason: str,
  src_turns: tuple[int, ...],     # the source-side turns the edge was witnessed in
  dst_turns: tuple[int, ...],     # likewise destination-side
  cited_entities: tuple[str, ...],
  cited_quote: str,               # verbatim quote from one of the source turns
)
```

Witness fields (`cited_entities`, `cited_quote`) are validated by
the extractor's witness layer against the actual turn text before
the edge is persisted. Invalid edges are dropped and accounted
for as `extractor_partial`.

The extractor LLM emits these as embedded `refs[]` on events via
`submit_events`; the witness layer unrolls them into `Edge`
records. The on-the-wire shape (after unrolling) is what the
auditor and distill pipelines see.

### `Finding`

```python
Finding(
  category: str,                  # free-text (no preset enum)
  description: str,
  related_event_ids: tuple[int, ...],
)
```

Output of scenario-registered audit checks. Advisory only — the
auditor LLM may ignore, contradict, or extend them.

### `Verdict`

```python
Verdict(
  surface_reminder: bool,         # false = stay silent
  reminder_text: str,             # required non-empty when surface_reminder
  continuation_notes: list[str],  # passed into the NEXT auditor firing
  matched_event_ids: list[int],   # event ids that justify the verdict
  cited_cards: list[str],         # AFC card ids cited (optional)
)
```

### `Phase`

A merged "basic block" over consecutive raw events. `task` /
`hyp` / `dec` / `concl` stay singleton; consecutive `act` / `evid`
events coalesce into a single block tagged `act_evid_run`. The
auditor reads the phase view for high-level reasoning and drills
back to raw events via `get_event_detail` when needed.

```python
Phase(
  id: int,
  kind: str,                      # EventKind value OR "act_evid_run"
  member_event_ids: tuple[int, ...],
  source_turns: tuple[int, ...],
  summary: str,
)
```

### `Reminder`

Simple wrapper for an injection payload. `Reminder(text=str)`.

---

## 2. Session-entry types (live persistence)

Persisted on the AgentM session entry tree by the adapter. Names
live in `audit/entry_types.py`; payloads are `<X>.to_dict()`.

| Entry type | Payload | Written when |
|---|---|---|
| `llmharness.audit_event` | `Event.to_dict()` | extractor accepts an event |
| `llmharness.audit_edge` | `Edge.to_dict()` | extractor accepts an edge (witness passed) |
| `llmharness.audit_phase` | `Phase.to_dict()` | adapter post-extraction phase-merge |
| `llmharness.verdict` | `Verdict.to_dict()` | auditor terminates successfully |
| `llmharness.extractor_cursor` | `{last_turn_index: int}` | extractor success/partial — marks window consumed |
| `llmharness.extractor_no_call` | `{reason, ...}` | extractor never called terminator |
| `llmharness.extractor_empty` | `{reason, ...}` | terminator called but non-trivial window had no events |
| `llmharness.extractor_error` | `{reason, ...}` | spawn / prompt / coercion crash |
| `llmharness.extractor_partial` | `{dropped_edges, ...}` | some edges failed witness, others kept |
| `llmharness.audit_no_call` | `{reason, ...}` | auditor never called terminator |
| `llmharness.audit_error` | `{reason, ...}` | auditor spawn / prompt / coercion crash |
| `llmharness.reminder_delivered` | `{reminder_text, turn_index}` | reminder actually injected |

Every typed failure entry is paired with a `DiagnosticEvent` on
the observability JSONL — no silent failures.

---

## 3. Replay sidecar JSONL

Path: `<cwd>/.agentm/audit_replay/<root_session_id>.jsonl`.
One line per phase invocation. Source: `replay/record.py`.

```json
{
  "phase": "extractor" | "auditor",
  "turn_index": 6,
  "root_session_id": "abc123",
  "ts_ns": 1700000000000000000,

  "compose_kwargs": {
    "base_prompt": "<resolved framing text>",
    "cards_tools_config": {},
    "observability_config": {},
    "trajectory_snapshot": [...],   // auditor only
    "events": [...],                // auditor only
    "edges": [...],                 // auditor only
    "phases": [...],                // auditor only
    "findings": [...],              // auditor only
    "check_errors": {},             // auditor only
    "continuation_notes": [],       // auditor only
    "summary_threshold": 30,        // auditor only
    "tools": ["submit_verdict"]     // auditor only — resolved tool list
  },

  "payload": {
    "new_turns": [...],             // extractor only
    "recent_graph": [...],          // extractor only

    "graph": [...],                 // auditor only
    "recent_verdicts": [...],       // auditor only
    "continuation_notes_from_prior_firing": [...]
  },

  "provider": ["agentm.ai.providers.openai_provider", {"model": "..."}]
             | null,                // null = use parent session default

  "output": {
    "events": [...], "edges": [...], "dropped_edges": [...]   // extractor
    |
    "surface_reminder": false, "reminder_text": "",           // auditor
    "continuation_notes": [], "matched_event_ids": [],
    "cited_cards": []
  } | null,

  "status": "ok" | "no_call" | "spawn_error" | "prompt_error",
  "error": null | "<stringified exception>",
  "latency_ms": 1234,
  "extras": {}
}
```

Two practical points:

* The record is **self-contained**: `compose_kwargs` + `payload`
  is everything you need to rebuild the exact extension list +
  child input. That is what `llmharness-replay` does for A/B.
* The schema is intentionally string-keyed JSON, not pickled
  dataclasses, so consumers can be written in any language.

---

## 4. Distill meta sidecar

Path: `<cwd>/.agentm/audit_replay/<root_session_id>.meta.json`.
Written by the `distill_binding` §11 atom at install time on the
main agent.

```json
{
  "sample_id": "ts0-mysql-corrupt-kwx8n5",
  "dataset_name": "rca-openrca2-lite",
  "dataset_path": "/path/to/data.jsonl",
  "root_session_id": "abc123"
}
```

Read by the labeler via
`llmharness.distill.binding.read_sample_meta(path)`. We do NOT
add a sample_id field to `ReplayRecord` — keeping the replay
schema agnostic to downstream use cases.

---

## 5. Distill labels JSONL (intermediate, Stage 1 output)

Path: `<labels-dir>/<root_session_id>.labels.jsonl`. One row per
auditor firing.

```json
{
  "sample_id": "ts0-mysql-corrupt-kwx8n5",
  "root_session_id": "abc123",
  "turn_index": 12,

  "input_payload": {                          // student-visible — NO GT
    "turn_index": 12,
    "events": [...],                          // causally masked
    "edges": [...],                           // causally masked
    "findings": [...],                        // causally masked
    "trajectory": [...]                       // truncated to t
  },

  "oracle": {                                 // audit-only — may cite GT
    "selected_finding_indices": [1],
    "matched_event_ids": [2, 7],
    "rationale_with_gt": "...",
    "continuation_notes": [...]
  },

  "rewriter": {                               // audit-only
    "justifiable_from_graph": true,
    "reminder_text": "...",
    "drop_reason": "",
    "matched_event_ids": [2, 7]
  },

  "target_verdict": {                         // present iff not dropped
    "surface_reminder": true,
    "reminder_text": "...",
    "matched_event_ids": [2, 7],
    "continuation_notes": [...],
    "cited_cards": []
  } | null,

  "drop": false,
  "drop_reason": "",

  "gt_meta": {                                // audit-only
    "fault_type": "NetworkCorrupt",
    "fault_category": "NetworkChaos"
  }
}
```

Audit-only fields (`oracle.rationale_with_gt`, `gt_meta`) MUST NOT
appear in the final SFT files — the exporter filters them out.

---

## 6. SFT JSONL (Stage 2 output)

Two student-visible files, one audit-only file:

### `sft/extractor.jsonl`

```json
{
  "phase": "extractor",
  "sample_id": "...",
  "root_session_id": "...",
  "turn_index": 6,
  "input": {
    "system": "<EXTRACTOR_SYSTEM_PROMPT verbatim>",
    "user":   "<json.dumps(payload) verbatim>"
  },
  "target": {
    "tool_calls": [
      {"name": "submit_events",
       "arguments": {
         "events": [
           {"id": 1, "kind": "task", "summary": "...",
            "source_turns": [0],
            "refs": [{"dst": 2, "kind": "ref", "reason": "...",
                      "src_turns": [0], "dst_turns": [1],
                      "cited_entities": ["..."], "cited_quote": "..."},
                     ...]},
           ...
         ]
       }}
    ]
  },
  "meta": {"replay_ts_ns": 1700000000000000000}
}
```

Events carry edges as embedded `refs[]` (the v3.1 `submit_events`
shape). The exporter re-attaches them from the recorded `edges`
list.

### `sft/auditor.jsonl`

```json
{
  "phase": "auditor",
  "sample_id": "...",
  "root_session_id": "...",
  "turn_index": 12,
  "input": {
    "system": "<AUDITOR_SYSTEM_PROMPT verbatim>",
    "user":   "<json.dumps(input_payload) verbatim>"   // causal snapshot, no GT
  },
  "target": {
    "tool_calls": [
      {"name": "submit_verdict",
       "arguments": {"verdict": {
         "surface_reminder": true,
         "reminder_text": "...",
         "matched_event_ids": [2, 7],
         "continuation_notes": [],
         "cited_cards": []
       }}}
    ]
  },
  "meta": {"fault_type": "NetworkCorrupt", "fault_category": "NetworkChaos"}
}
```

The `user` field is a JSON string identical to the rewriter's
input — the student sees exactly the surface it must learn to
reproduce at inference. The `meta` block is for batching /
analysis and should not be fed to the model.

### `sft/dropped.jsonl`

Same row as Stage-1 labels but only for `drop=true` rows.
Audit-trail file, not for training.
