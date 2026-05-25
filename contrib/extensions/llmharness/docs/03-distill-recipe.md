# Distill recipe

End-to-end: from a fault-injection dataset with ground-truth labels
to `sft/{extractor,auditor}.jsonl` ready for SFT.

Read [01-architecture.md](01-architecture.md) §3 for the flow
diagram; this doc gives the concrete commands plus the design
decisions that make the labels safe to learn from.

---

## 1. Prerequisites

* A dataset JSONL where each row has at minimum:
  ```json
  {"source": "<sample_id>",
   "ground_truth": ["<root_cause_1>", "..."],
   "fault_type": "<short tag>",
   "fault_category": "<short tag>"}
  ```
  The rca dataset shape is the default; see
  [04-extending.md](04-extending.md) §2 to adapt to another
  shape.

* AgentM workspace synced (`uv sync`). The `agentm` CLI must be
  on PATH.

* Provider credentials for the oracle/rewriter children, e.g.
  `OPENAI_API_KEY` in `.env`. The labeler reuses AgentM's provider
  registry, so any short id (`openai`, `anthropic`, ...) works and
  env-derived knobs (`WARPGATE_TICKET`, `OPENAI_VERIFY_SSL`, ...)
  are folded in automatically.

---

## 2. Stage 0 — collect clean trajectories (live)

For each sample, run the main agent on the fault-injection scenario
with reminders OFF (so the auditor still records data but doesn't
perturb the trajectory) and the distill binding ON (so we get a
meta sidecar joining the run to a sample id).

```bash
# per-sample loop, executed by your driver script
LLMHARNESS_DISTILL_SAMPLE_ID="$SAMPLE_ID" \
LLMHARNESS_DISTILL_DATASET="$DATASET_PATH" \
LLMHARNESS_DISTILL_DATASET_NAME="$DATASET_NAME" \
  agentm \
    --cwd "$RUN_DIR" \
    --scenario rca \
    --extension llmharness.adapters.agentm \
    --extension llmharness.distill.binding \
    --extension-config llmharness.adapters.agentm='{
        "enable_reminders": false,
        "enable_auditor": true,
        "enable_replay_log": true,
        "audit_interval_turns": 3
    }' \
    "$RCA_PROMPT_FOR_THIS_SAMPLE"
```

What each knob does:

| Knob | Effect |
|---|---|
| `enable_reminders: false` | Auditor still fires and a `Verdict` is recorded, but it is never injected as an advisory before the next agent turn. The main-agent trajectory stays clean. |
| `enable_auditor: true` | Phase 2 runs. We want auditor data even in Stage 0 — only the **delivery** is suppressed. |
| `enable_replay_log: true` | Default; states it explicitly. Produces `<sid>.jsonl`. |

What you get under `$RUN_DIR/.agentm/audit_replay/`:

```
<session_id>.jsonl       # replay records (one per phase invocation)
<session_id>.meta.json   # sample_id binding (written by distill_binding)
```

Run as many samples as you want into the same `$RUN_DIR` — each
session has its own `<session_id>`, so files don't collide.

### Alternative: use the host scenario's eval runner

For scenarios that ship their own eval driver (e.g. rca runs samples
via `rca llm-eval run …` from rcabench-platform), mounting
`llmharness.distill.binding` is not always possible — the driver
controls the extension list. In that case:

1. Run the eval as usual; the replay sidecar still gets written.
2. **Skip the meta sidecar** and inject the sample-id at aggregation
   time via the `llmharness-aggregate --sample-id …` override (see
   [06-case-aggregation.md §1](06-case-aggregation.md)).
3. Then feed the tagged case dir into the labeler.

The labeler reads its GT join from the meta sidecar by default; for
the runner-driven flow, pre-write a sidecar alongside the replay
file (`<sid>.meta.json` with `{"sample_id": "...", ...}`) or extend
the driver script to call `llmharness.distill.binding.SampleMeta`
directly. The format is documented in
[02-schemas.md §4](02-schemas.md#4-distill-meta-sidecar).

---

## 3. Stage 1 — offline labeling

```bash
llmharness-distill label \
  --replay-dir       "$RUN_DIR/.agentm/audit_replay" \
  --dataset          "$DATASET_PATH" \
  --out              ./distill_labels \
  --cwd              /tmp/distill-children \
  --oracle-provider  'openai#{"model":"gpt-4o"}' \
  --rewriter-provider 'openai#{"model":"gpt-4o"}'
```

For each session sidecar, the driver:

1. Reads `<sid>.meta.json` to get `sample_id`; looks up GT.
2. For each `auditor` replay record at turn t:
   * Reconstructs the graph from `compose_kwargs`.
   * **Causally masks** events / edges / findings / trajectory to
     `max(*_turns) ≤ t`.
   * **Stage A — oracle child** (sees GT). Reads the masked
     snapshot + GT. Calls `submit_oracle_label(
     selected_finding_indices, matched_event_ids,
     rationale_with_gt, continuation_notes)`. May cite GT in
     `rationale_with_gt`.
   * **Stage B — rewriter child** (NO GT). Reads only the masked
     snapshot + the oracle's selection. Calls `submit_rewrite(
     justifiable_from_graph: bool, reminder_text: str,
     drop_reason: str, matched_event_ids)`.
   * If `justifiable_from_graph=false` ⇒ DROP. Otherwise emit a
     label row with the rewriter-approved fields.
3. Writes `<sid>.labels.jsonl` to `--out`.

Extractor replay records pass through unchanged — the live
teacher already met the extractor quality bar.

Per-session log lines show counts:

```
abc123.jsonl: kept=8 dropped=3
def456.jsonl: kept=5 dropped=0
total: kept=13 dropped=3 out=distill_labels/
```

The two child sessions are spawned with a **minimal extension
list** (see §5 D3 below): observability + operations_local +
system_prompt + one submit tool. No FS, no bash, no skills, no
cards. The student will see the same surface at inference, so
training matches deployment.

---

## 4. Stage 2 — SFT export

```bash
llmharness-distill export \
  --labels     ./distill_labels \
  --replay-dir "$RUN_DIR/.agentm/audit_replay" \
  --out        ./sft \
  --phase      both     # or: extractor | auditor
```

Outputs:

```
./sft/extractor.jsonl   # one row per ok extractor replay record
./sft/auditor.jsonl     # one row per non-dropped label
./sft/dropped.jsonl     # one row per dropped label (audit only)
```

Shapes are defined in [02-schemas.md §6](02-schemas.md#6-sft-jsonl-stage-2-output).

---

## 5. Design decisions (load-bearing)

These are the choices that make the SFT labels safe for the
student to learn from. Changing any one of them changes the
distillation's correctness story — flag in a PR if you do.

### D1 — Two-stage oracle, not one

Stage A's prompt explicitly carries GT and asks "which findings
matter given GT". Stage B's prompt sees zero GT and asks "is this
selection justifiable from the graph alone". If no, drop. This
protects the student from learning a GT-conditioned selection
function.

### D2 — Causal masking is non-negotiable

Stage A is given only the slice of the graph that existed at
turn t (events with `max(source_turns) ≤ t`, both endpoints kept
for edges, all `related_event_ids` kept for findings). Without
this, the oracle uses post-hoc evidence to label the verdict,
and the student — which only sees past turns at inference — learns
a function it cannot reproduce. Six fail-stop tests in
`tests/test_distill_causal.py` lock this down.

### D3 — Minimal extensions on the audit children

Oracle and rewriter children run with four atoms only:

* `observability` — events + spans to JSONL via OTLP
* `operations_local` — required by substrate freeze
* `system_prompt` — the short, focused prompt
* one submit tool — `submit_oracle_label` or `submit_rewrite`

No FS, no bash, no cards, no skills. This is the surface the
student will reproduce at inference time.

### D4 — Negative samples are intra-sample, not synthetic

A run that produces (say) 200 findings across 50 firings will
naturally contain firings where the oracle says "flag none".
Those firings ARE the negative samples — they teach the model
"I see findings but choose not to fire". No synthetic negatives.

### D5 — Selection is the learnable signal; phrasing is templated

The auditor SFT target is mostly `(surface_reminder,
matched_event_ids)`. `reminder_text` is rewriter-produced and
follows a methodological vocabulary (verify hypothesis, close
branch, avoid repeated action) — all graph-derivable. We do not
ask the student to invent free-text critique.

### D6 — Sample-id binding via a sidecar, not a schema change

The §11 atom `distill_binding` reads `LLMHARNESS_DISTILL_SAMPLE_ID`
at install time and writes `<sid>.meta.json` next to the replay
log. The labeler joins replay records → GT through this file. We
do NOT add a `sample_id` field to `ReplayRecord` — keeping the
replay schema agnostic to use-case.

---

## 6. Verifying the run

After Stage 2, sanity-check that the audit-only fields didn't leak
into the student-visible files:

```bash
# These should print zero matches (or only matches inside `meta` blocks).
# target.messages[0].tool_calls[0].function.arguments is a JSON string
# (OpenAI-compatible tool-call shape) — no extra parse needed for a
# substring grep.
jq -r '.input.user' sft/auditor.jsonl | grep -ci "$ROOT_CAUSE_KEYWORD" || echo OK
jq -r '.target.messages[0].tool_calls[0].function.arguments' sft/auditor.jsonl \
  | grep -ci "$FAULT_TYPE_KEYWORD" || echo OK
```

For the smoke run we shipped, this returned zero matches while the
oracle's `rationale_with_gt` in `distill_labels/*.labels.jsonl`
explicitly cited GT — exactly the isolation we want.

---

## 7. Scaling

The current CLI is `asyncio.run` per session — i.e. sessions
process serially. For larger collections, parallelize at the
session level:

```bash
ls "$RUN_DIR/.agentm/audit_replay/"*.jsonl \
  | xargs -P 8 -I{} llmharness-distill label \
      --replay-dir "$(dirname {})" \
      --dataset    "$DATASET_PATH" \
      --out        ./distill_labels \
      --cwd        "/tmp/distill-children-$$" \
      ...
```

Each worker handles a disjoint set of sessions and writes its own
`<sid>.labels.jsonl`, so there is no contention. Export still
runs once at the end against the merged label directory.
