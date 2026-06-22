# Case aggregation

Offline pipeline that takes one run's replay sidecar(s) and groups
the data per case into a directory layout suited for human review
and downstream training-data export.

A **case** = one main-agent session run on one input. Identified by
`sample_id` from the distill binding sidecar; falls back to
`session_id` if no binding is present.

---

## 1. CLI

Three input layouts via subcommands; every command writes the same
canonical case-directory shape.

### `replay` — live-run sidecars

```bash
llmharness-aggregate replay \
  --cwd /path/to/run-dir \
  --out ./cases
```

Walks every `.agentm/audit_replay/*.jsonl` under `--cwd`, joins each
session to its `.meta.json` sidecar, and writes the layout below
under `--out/<case_id>/`. Existing case dirs are overwritten file by
file (no destructive cleanup of stray files).

Aggregate a single session:

```bash
llmharness-aggregate replay \
  --cwd /path/to/run-dir \
  --session-id abc123def \
  --out ./cases
```

When the run did NOT mount `llmharness.distill.binding` (e.g. `rca
llm-eval` runs through rcabench-platform), the meta sidecar is
missing and `case_id` falls back to the session id. Inject the
sample-id manually:

```bash
llmharness-aggregate replay \
  --cwd . \
  --session-id eddfe314... \
  --sample-id ts0-mysql-corrupt-kwx8n5 \
  --dataset-name rca-openrca2-lite \
  --dataset-path /abs/path/to/data.jsonl \
  --out ./cases
```

The overrides win over any meta sidecar — useful when re-tagging a
session whose binding was forgotten.

### `one` — a single replay-format JSONL file

```bash
llmharness-aggregate one \
  --replay-path /tmp/abc.jsonl \
  --out ./cases \
  --sample-id rca-mysql-001 \
  --dataset-name rca-openrca2-lite
```

### `sessions` — AgentM sessions from ClickHouse

```bash
llmharness-aggregate sessions \
  --session-file /tmp/session_ids.txt \
  --out ./cases
```

Fetches ordinary AgentM session trajectories from the ClickHouse trace
backend and writes the same case-directory layout. This is for runs
that did not mount `llmharness.atom`, so no `.agentm/audit_replay/`
sidecar exists. Each line in `--session-file` starts with a session id;
blank lines and `#` comments are ignored. Repeat `--session-id` to pass
ids inline.

Because these sessions have no extractor/auditor firings,
`main_agent.jsonl` is the primary artifact and `trajectory.jsonl`
contains a flat main-agent message timeline pointing at
`main_agent.jsonl` lines.

---

## 2. Per-case layout

```
cases/<case_id>/
├── meta.json                          summary fields (machine-readable)
├── main_agent.jsonl                   full main-agent message trajectory
│                                      (one per line, AgentM-native shape)
├── extractor/
│   ├── 001_turn_002.json              one extractor firing — input + output
│   └── 002_turn_005.json
├── auditor/
│   └── 001_turn_006.json              one auditor firing — input + verdict
├── event_graph/
│   ├── after_extractor_001.json       accumulated events + edges after that firing
│   └── after_extractor_002.json
├── verdicts.jsonl                     verdict timeline (one verdict per line)
├── trajectory.jsonl                   flat review timeline pointing into the artefacts
└── README.md                          auto-generated overview for human review
```

Filenames are zero-padded so directory listings sort correctly.
`<NNN>` is the firing's 1-based order within its phase; `<T>` is the
main-agent turn index it fired on.

### `meta.json`

```json
{
  "case_id": "rca-mysql-001",
  "session_id": "abc123def",
  "trace_id": "def456abc",
  "sample_id": "rca-mysql-001",
  "dataset_name": "rca-openrca2-lite",
  "dataset_path": "/data/rca.jsonl",
  "started_at_ns": 1700000000000000000,
  "ended_at_ns":   1700000001000000000,
  "extractor_firings": 2,
  "auditor_firings": 1,
  "surfaced_reminders": 0,
  "silent_verdicts": 1
}
```

### `main_agent.jsonl`

One AgentM-native message per line. Lossless — every block of every
message (text, tool_use, tool_result, thinking, etc.) is preserved.
This is the canonical source for converting to any chat-completions
or trainer-specific format downstream.

Reconstruction strategy (load-bearing — without this the case dir
silently truncates runs that end mid-auditor-interval):

1. **Base**: the latest `compose_kwargs.trajectory_snapshot` from a
   successful auditor firing. Authoritative up to that turn.
2. **Tail**: every extractor firing whose `turn_index` is greater
   than the base contributes its `payload.new_turns` window. Walks
   in chronological order; deduplicates by message `index`.

This stitches in the trailing main-agent turns that occurred after
the last auditor fired — the common case when runs hit a timeout or
`submit_final_report` mid-interval.

### `extractor/NNN_turn_T.json` and `auditor/NNN_turn_T.json`

```json
{
  "phase": "extractor",
  "sequence": 1,
  "turn_index": 2,
  "ts_ns": 1700000000000000000,
  "status": "ok",
  "error": null,
  "latency_ms": 1234,
  "input": { "payload": { ... }, "summary_threshold": 30 },
  "output": { "events": [...], "edges": [...], "dropped_edges": [] }
}
```

Auditor records additionally surface `findings`, `check_errors`,
`continuation_notes`, and the resolved `tools` tuple from the
adapter's profile resolution.

### `event_graph/after_extractor_NNN.json`

Cumulative index state immediately after that extractor firing
succeeded. Non-ok firings (`spawn_error`, `no_call`, etc.) do NOT
advance the snapshot — mirrors the live adapter's cursor semantics.

### `verdicts.jsonl`

```jsonl
{"sequence": 1, "turn_index": 6, "ts_ns": ..., "surface_reminder": true, "reminder_text": "verify hypothesis 2", "matched_event_ids": [2, 7], ...}
```

### `trajectory.jsonl`

Flat per-source timeline for human review. Each line points back at
the per-firing JSON so a reviewer can grep + open:

```jsonl
{"ts_ns": ..., "source": "extractor", "sequence": 1, "turn_index": 2, "summary": "extractor#1 turn=2 status=ok events=7 edges=4", "ref": "extractor/001_turn_002.json"}
{"ts_ns": ..., "source": "auditor",   "sequence": 1, "turn_index": 6, "summary": "auditor#1 turn=6 status=ok verdict=silent",       "ref": "auditor/001_turn_006.json"}
```

For plain ClickHouse sessions aggregated through `sessions`, there are no
per-firing files, so rows point into `main_agent.jsonl`:

```jsonl
{"ts_ns": ..., "source": "main_agent", "sequence": 1, "turn_index": null, "summary": "message#1 role=user text=1", "ref": "main_agent.jsonl", "line": 1}
```

### `README.md`

One-page case summary. Lists meta + firing summaries with status and
verdict outcomes. Read this first when scanning a case.

---

## 3. Relationship to distill

| Pipeline | Input | Output | When to use |
|---|---|---|---|
| `llmharness-aggregate` | replay sidecar | per-case directories | **always** — for human review of any run |
| `llmharness-distill label` | replay sidecar + dataset GT | per-session label JSONL | when you need SFT data for the harness model |
| `llmharness-distill export` | label JSONL + replay sidecar | `sft/{extractor,auditor}.jsonl` | the actual training files |

The two pipelines share the same replay sidecar input; they do
**different** things and can be run independently or together.
Aggregation does not consume or produce SFT JSONL.

### Working with `rca llm-eval` flows

The rca scenario is usually exercised via
`rca llm-eval run … --ak scenario=rca:harness.sync` (the
rcabench-platform driver). That path runs the harness scenario but
does NOT mount `llmharness.distill.binding`, so no meta sidecar is
written. Aggregation still works — pass `--sample-id` /
`--dataset-name` / `--dataset-path` to tag the case correctly:

```bash
rca llm-eval run config.yaml -a agentm --ak scenario=rca:harness.sync -l 1
# replay sidecar lands under $AGENTM_CWD/.agentm/audit_replay/<sid>.jsonl

llmharness-aggregate replay \
  --cwd "$AGENTM_CWD" \
  --session-id <sid> \
  --sample-id <sample_id_from_dataset> \
  --dataset-name rca-openrca2-lite \
  --dataset-path /abs/path/to/data.jsonl \
  --out ./cases
```

If you need the labeler downstream, also pre-write the meta sidecar
file yourself (`<sid>.meta.json`) — its shape is in
[02-schemas.md §4](02-schemas.md#4-distill-meta-sidecar).

---

## 4. Exporting trajectories for training

`main_agent.jsonl` is the AgentM-native message format. Common
downstream conversions:

* **OpenAI chat completions** — map each `role: user|assistant|tool` +
  flatten content blocks; tool calls become `tool_calls`.
* **Anthropic messages** — preserve content blocks largely as-is.
* **Plain text trajectory** — concatenate text from all content blocks.

The conversion is one-way and trainer-specific, so it lives outside
this package. Open `main_agent.jsonl` directly in your trainer's
preprocessing step.

The auditor / extractor SFT data is shipped separately by
`llmharness-distill export` — see [03-distill-recipe.md](03-distill-recipe.md).

---

## 5. Programmatic use

```python
from pathlib import Path
from llmharness.aggregate import collect_case, write_case

case = collect_case(
    replay_path=Path("/run/.agentm/audit_replay/abc.jsonl"),
    meta_path=Path("/run/.agentm/audit_replay/abc.meta.json"),
)
case_dir = write_case(case, Path("./cases"))
```

`CaseData`, `FiringRecord`, `GraphSnapshot`, and `CaseMeta` are all
frozen dataclasses — safe to pass around and inspect. The data model
is the source of truth; the on-disk layout is just one
materialisation. Add new exports (e.g. an HTML viewer) on top of
`CaseData` without touching the collector.

---

## 6. Serving cases over HTTP

For remote review (multiple humans pointing a browser at the same
cases set), upload the `cases/` tree to a blob bucket and point the
viewer at the bucket prefix:

```bash
aegisctl blob mirror ./cases/ shared:cases/<iter-name>/
```

The companion frontend lives in `aegis-ui`'s `Case Review` sub-app
(`/cases` → Connection settings → paste the bucket / prefix). Each
browser instance configures its own backend URL in the sub-app's
settings.
