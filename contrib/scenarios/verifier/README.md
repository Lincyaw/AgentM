# verifier — ground-truth label verification

Verifier is a **GT label auditor**: it checks whether existing
ground-truth propagation labels are correct and discovers propagation
that GT missed.

## What it does

Given a fault-injection case with observability data (traces, metrics,
logs), the verifier runs BFS-style propagation from the injection
seed(s).  At each hop a lightweight agent queries the data to decide
whether the neighbour service is genuinely degraded.  The output is:

- **propagation_nodes** — services confirmed as degraded, with SQL
  evidence.
- **propagation_edges** — directed edges between confirmed services.

## Why it exists

The platform's GT labels are produced by a rule-based causal-graph
engine operating at span granularity.  This leaves two classes of
error:

1. **Missing labels** — services that are genuinely degraded but not
   reachable by the engine's rule set (e.g. co-deployed services
   affected by node-level faults, backing components without spans).
2. **Wrong labels** — services marked as propagated that show no
   measurable degradation in the data (composition artifacts, load
   drift, sub-noise-floor changes).

The verifier acts as an independent, data-driven second opinion at
service granularity.  Comparing its findings against GT highlights
both gaps.

## How to use

```bash
# Single case
uv run python contrib/scenarios/verifier/eval/audit_case_propagate.py \
    run <case_dir> --model litellm

# Batch (ablation runs)
uv run python contrib/scenarios/verifier/eval/audit_case_propagate.py \
    batch <dataset_dir> --run-dir /tmp/verifier-run-1 \
    --model litellm --parallel 4 --limit 10

# Resume (cached results are skipped automatically)
uv run python contrib/scenarios/verifier/eval/audit_case_propagate.py \
    batch <dataset_dir> --run-dir /tmp/verifier-run-1 --model litellm
```

## Output

Each case produces under `<run-dir>/<case>/`:

| File | Content |
|------|---------|
| `report.json` | injections, propagation_nodes (with evidence), propagation_edges |
| `propagation_trace.json` | full BFS trace: rounds, hop_log, node_evidence |
| `relationships.json` | the relationship graph used for BFS |
| `hops/<from>__<to>/` | per-hop agent session (stdout, stderr, observability) |

Batch runs additionally write `run_summary.jsonl` — one JSON line per
case with `{case, fault_kind, seeds, confirmed, propagated, edges,
rounds}`, suitable for cross-run diff and ablation analysis.

## Interpreting results

The verifier's job is to surface discrepancies against GT, not to
produce a final label.  Each finding needs human review:

- **NEW (verifier found, GT missing)**: check the SQL evidence — is
  the degradation real and causally related, or a composition artifact
  / load drift?
- **MISSED (GT has, verifier missed)**: usually a graph reachability
  gap (no edge to the service) or the degradation is below the
  agent's noise threshold.
- **MATCH**: GT and verifier agree.

Known limitations:

- Graph reachability: services without a direct or transitive edge
  from the seed can't be checked (e.g. frontend can't reach rate
  unless search is confirmed first).
- Co-deployed node effects: NetworkDelay on `frontend→profile`
  affects all pods on profile's k8s node via tc netem, not just
  profile — verifier models this via `co_deployed` edges but the
  mechanism is subtle.
- Composition artifacts: when the abnormal window has drastically
  fewer spans with different type mix, percentile changes reflect
  composition shift, not real degradation.
