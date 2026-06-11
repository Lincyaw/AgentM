# Extractor strategy iteration workflow

This is the review loop for changing the extractor prompt / strategy and
checking the resulting semantic graph in `aegis-ui`.

Goal: replay the same recorded main-agent trajectory through a candidate
extractor model or prompt, convert the replayed extractor outputs into the
canonical aggregate case-directory layout, upload that layout to blob
storage, and review it in the `llmharness-review` app.

Use this workflow when evaluating whether the extractor maintains the
graph well: adding new semantic events, repairing bad historical cuts,
merging setup / retry noise, splitting overloaded nodes, and repointing
edges when later evidence changes the graph.

## Inputs

- A replay sidecar:
  `<cwd>/.agentm/audit_replay/<session_id>.jsonl`
- Optional prompt edits:
  `src/llmharness/agents/extractor/prompt.py (PROMPT_DEFAULT)`
- A target model profile from `~/.agentm/config.toml`, for example
  `kimi` (`K2.6`) or another provider override.
- A blob prefix for the review set, for example:
  `shared:llmharness-review/kimi-k26-YYYYMMDD/`

The `aegis-ui` reader expects a case root shaped like:

```text
<prefix>/
  <case_id>/
    meta.json
    main_agent.jsonl
    trajectory.jsonl
    verdicts.jsonl
    extractor/NNN_turn_TTT.json
    auditor/NNN_turn_TTT.json
    event_graph/after_extractor_NNN.json
```

The aggregate layout is the contract. Do not upload only the raw replay
report if the goal is UI review.

## 1. Replay the extractor chain

For quick single-firing checks:

```bash
PYTHONPATH=src:contrib/extensions/llmharness/src \
  uv run --no-sync llmharness-replay extractor \
  --record .agentm/audit_replay/<session_id>.jsonl \
  --index 0 \
  --cwd .
```

For strategy review, replay the extractor as a threaded chain so each
later extractor firing sees the graph produced by earlier replayed
firings. The built-in chain CLI prints progress only; use a small dump
driver around `llmharness.replay.chain` when you need the full replayed
outputs for aggregation.

The dump must record, for each extractor firing:

- original record index
- turn index
- status / error / latency
- replayed extractor output (`events`, `edges`, `dropped_edges`, `ops`)
- final cumulative graph

The model override should be explicit in the report. For Kimi, resolve
the `kimi` profile from `~/.agentm/config.toml` and pass the resulting
provider tuple to `replay_extractor_record`.

Expected high-level checks:

- Every firing should reach `finalize_extraction`; `no_call` is a tool-use
  failure.
- Node count should not grow as a pure transcript chain.
- Later firings should add edges back to prior semantic nodes when new
  evidence reinforces, refutes, or revises an earlier hypothesis.
- If a prompt change is supposed to encourage repair, inspect the op log
  for old-id `upsert_node`, `delete_node`, `delete_edge`, or edge
  repointing. A run with only new `node_upsert` + `edge_upsert` is still
  append-only.

## 2. Build a replay sidecar from the candidate outputs

The aggregator consumes replay-format JSONL. To view a candidate chain in
the UI, write a derived sidecar that keeps each original extractor
record's input (`payload`, `compose_kwargs`, ids, timestamps) but replaces
its output/status/latency with the candidate replay result.

For extractor-only review, omit auditor records. This produces a case with
`auditor_firings = 0`, which is fine when reviewing extractor graph
quality.

The derived record should preserve:

- `phase = "extractor"`
- `turn_index`
- `session_id`
- `trace_id`
- `ts_ns`
- `compose_kwargs`
- `payload`

And replace:

- `provider` with the tested model/provider tuple
- `output` with the replayed extractor output
- `status`, `error`, `latency_ms` with the replayed result
- `extras.replay_model_profile` or similar metadata for traceability

## 3. Aggregate into case-directory layout

Run `llmharness-aggregate one` on the derived sidecar:

```bash
PYTHONPATH=src:contrib/extensions/llmharness/src \
  uv run --no-sync llmharness-aggregate one \
  --replay-path /tmp/<candidate-extractor>.jsonl \
  --out /tmp/<candidate-cases> \
  --sample-id <case-id>-<model-or-strategy-tag> \
  --dataset-name <scenario-or-dataset-name> \
  --dataset-path <dataset-path-if-known>
```

Verify the final snapshot:

```bash
jq '{after_extractor_firing, turn_index, events:(.events|length), edges:(.edges|length)}' \
  /tmp/<candidate-cases>/<case-id>/event_graph/after_extractor_*.json
```

For `aegis-ui`, the important files are `meta.json`,
`main_agent.jsonl`, `extractor/*.json`, and
`event_graph/after_extractor_*.json`.

## 4. Upload to blob for UI review

Upload the whole cases root, not the case directory contents alone:

```bash
aegisctl blob cp -r /tmp/<candidate-cases>/ \
  shared:llmharness-review/<iteration-name>/
```

Then confirm the prefix has case child directories:

```bash
aegisctl blob ls shared:llmharness-review/<iteration-name>/ -o json
```

The UI configuration is:

```text
bucket = shared
prefix = llmharness-review/<iteration-name>/
```

The prefix must end at the cases root. The UI enumerates immediate child
directories under that prefix and expects each child to contain
`meta.json`.

## 5. Review in `../aegis-ui`

Open the `llmharness-review` app and configure the blob source:

```text
Bucket: shared
Prefix: llmharness-review/<iteration-name>/
```

Review at least:

- final `event_graph/after_extractor_NNN.json`
- per-firing extractor outputs
- whether edges encode semantic dependency rather than transcript
  adjacency
- whether late evidence updates or reconnects prior hypotheses
- whether setup/retry/thin-decision nodes were merged away
- whether important independent evidence branches remain separate

Record feedback against the exact blob prefix and case id. The prefix is
the experiment identifier for future comparison.

## 6. Example from the Kimi K2.6 check

Source replay:

```text
.agentm/audit_replay/2fe74933da3b42d4b1174810fe8921a7.jsonl
```

Candidate model:

```text
~/.agentm/config.toml profile: kimi
provider override module: agentm.extensions.builtin.llm_anthropic
model: K2.6
```

Uploaded review set:

```text
shared:llmharness-review/kimi-k26-20260526/
```

UI settings:

```text
Bucket: shared
Prefix: llmharness-review/kimi-k26-20260526/
Case: batch-01KQHDBBFCJTTJNJV624R7GPT3-kimi-k26
```

Observed final graph:

```text
after_extractor_004.json: 12 events / 14 edges
```

The Kimi run was better than the Doubao baseline on this case because all
four extractor firings called `finalize_extraction`, the graph was more
compact, and the last firing linked new evidence back to the earlier
recommendation/profile hypothesis. It still did not emit delete/repair ops,
so future strategy changes should be judged by whether they actually
rewrite old graph state rather than only appending more nodes.
