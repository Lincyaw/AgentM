# Extractor prompt-iteration loop

One round of the loop:

```bash
scripts/extractor-iter.sh <iter-name> [--prompt PATH]
```

1. Re-run the extractor on the pinned 10 rows with the chosen prompt
   (`rerun_extractor.py` — auditor records pass through).
2. Aggregate to `runs/iters/<iter-name>/cases/` (canonical layout).
3. Upload to `shared/cases/iter-<iter-name>/` on aegis-blob
   (`upload_cases_to_blob.py`).
4. Review at https://118.196.98.178:8082/cases — point Settings at
   `bucket=shared, prefix=cases/iter-<iter-name>/`.

The case ids are stable across iterations
(`openrca-2-lite-n500-t20-row<N>`), so a deep-link like
`/cases/openrca-2-lite-n500-t20-row1#mode=extractor&E=1` survives a
prefix swap — handy for diffing the same case across two iterations
in two browser tabs.

## Inputs you'll edit between iterations

- `src/llmharness/audit/extractor/prompts/extractor_default.md` —
  the prompt itself (or supply a different file via `--prompt`).
- `runs/iters/pinned-10.txt` — the row id set. Keep this stable
  unless you're deliberately switching corpora.

## Knobs

| Flag             | Default                                                       | Meaning                                                   |
| ---------------- | ------------------------------------------------------------- | --------------------------------------------------------- |
| `--prompt`       | `src/llmharness/audit/extractor/prompts/extractor_default.md` | Extractor system-prompt file to use this iteration.       |
| `--rows`         | `runs/iters/pinned-10.txt`                                    | Pinned row id set; one int per line, `#`-comments OK.     |
| `--source`       | `runs/eval_db/openrca-2-lite-n500-t20`                        | Existing eval-db extract root with `<row>/records.jsonl`. |
| `--concurrency`  | `10`                                                          | Parallel rows. Each row is internally sequential.         |

Override the Python interpreter with `LLMH_PY=/path/to/python`. By
default the orchestrator uses `/home/ddq/AoyangSpace/AgentM/.venv/bin/python`,
which already has `agentm` + `llmharness` importable.

## Why not just call `llmharness adapter eval-db extract` again?

The original `eval.db` for `openrca-2-lite-n500-t20` isn't on this
machine — only the post-extract `runs/eval_db/.../records.jsonl`
sidecars are. `rerun_extractor.py` re-uses those captured trajectories
and only swaps the extractor LLM call, which is exactly what prompt
A/B-ing needs. Auditor records flow through untouched (the loop
optimises extractor first; auditor judgement quality is a downstream
question).

## Notes on infra you might want later

- `scripts/upload_cases_to_blob.py` is a stop-gap. The right home is
  a native `aegisctl blob put|sync|ls|rm` subcommand under
  `/home/ddq/AoyangSpace/aegis/aegislab/src/cli/cmd/`. Bring that in
  when more scripts need the same surface.
- The auditor side of the loop is wired but unused: pass `--phase both`
  in `chain_replay` if/when you want a parallel auditor prompt loop.
- The blob prefix `cases/iter-<name>/` overlaps the existing
  `cases/openrca-2-lite-n500-t20/` root the UI already serves; the
  `iter-` infix keeps them visually separate in object-listings.

## Extractor prompt — open issues observed during review

These are the candidates worth trying in successive iterations. They
were surfaced by auditing 6 cases of the current baseline data and
finding edge/node ratio ≈ 0.6–0.7 (graph is dominantly a chain) and
5–10 isolated events per case:

1. **`external_refs` is documented but rarely emitted.** The prompt
   already calls it "the single most common quality bug" — but the
   rendered cumulative graphs still come out as N disconnected
   per-firing islands. Worth tightening: require ≥1 `external_refs`
   on every non-genesis evid/act event that has a non-empty
   `recent_graph`, with a worked counter-example for refusal cases.
2. **Sparse intra-firing `refs`.** Each firing produces ~5 events with
   ~3 edges; most events have in-degree 1 / out-degree 1. The
   "summary ≤ 30 words" cap is plausibly forcing the LLM to drop
   shared dependencies (e.g. two evid events that both rest on the
   same upstream hyp). Try lifting to ≤ 60 words *and* adding a rule:
   "if two events share a witness, both must cite the common
   ancestor".
3. **`concl`/`dec` events end up isolated.** They're terminal claims
   but the prompt doesn't require them to cite the evidence that
   justified them. Add: "every `concl` / `dec` must carry ≥1 `refs`
   or `external_refs` pointing at the evidence it summarises".
4. **Information loss in `summary`.** Auditor downstream is asked to
   trace causal chains but ≤30 words rarely captures the parameter
   that mattered (e.g. "queried abnormal_traces" vs. "queried
   abnormal_traces for service=cart between 14:02–14:05"). Consider
   a `details` field separate from `summary`, or relax the cap for
   `act`/`evid`.
5. **Witness-retry budget vs. drop.** The current adapter drops refs
   that fail witness; the prompt tells the LLM "an empty `refs` list
   is fine". Combined, this nudges the LLM toward not bothering to
   ground refs. Maybe surface the drop count back to the LLM so it
   retries with literal witnesses on the second pass.

Iteration order I'd suggest: 1 → 3 → 4 → 2 → 5. (1 and 3 fix
disconnection; 4 fixes information loss for the auditor; 2 is the
biggest structural change so save it for after the wins are pinned;
5 is a harness change, not a prompt-only knob.)
