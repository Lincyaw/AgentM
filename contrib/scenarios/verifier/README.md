# verifier — fault propagation graph construction

## Task setting

Given a microservice system where one or more faults have been injected
(pod failure, network delay, CPU stress, etc.), determine which services
are **genuinely degraded as a result of the fault propagation**, and
produce the service-level propagation graph (nodes + edges + evidence).

Key clarifications:

- **Service-level, not span-level.** The unit of analysis is a service.
  A service is "propagated to" if it exhibits genuine degradation that
  is causally linked to the upstream fault — not merely because one of
  its span paths has fewer invocations.
- **Independent discovery, not GT matching.** The verifier constructs
  its own propagation graph from observability data. It does NOT try to
  reproduce GT labels. GT is a separate artifact used for comparison
  after the fact; the verifier's job is to find the truth in the data.
- **Edge-level evaluation.** Each directed edge (source → target) with
  a specific relationship type is an independent propagation hypothesis.
  The same target service may be reached from multiple sources via
  different relationships; each is evaluated separately. A rejection
  from one path does not preclude confirmation from another.
- **What counts as "genuinely degraded":**
  - Latency increase, error rate increase, service unavailability
    (zero spans + zero logs = service down, not just "nobody called it")
  - Throughput drop ALONE is not degradation of the service itself —
    unless the system is in cascading failure (>80% load generator
    throughput drop), or the drop is concentrated on a specific fault-
    related call path rather than uniform.
- **What does NOT count:**
  - Fewer incoming requests because callers stopped calling (that's the
    CALLER's problem, not this service's degradation)
  - Sub-noise-floor changes (sub-millisecond absolute differences)
  - Changes matching system-wide load drift

## Architecture

```
injection seed(s)
    │
    ▼  BFS expansion
┌─────────────────────────────────────────────┐
│ For each edge (confirmed → neighbor):       │
│   • already confirmed? → SQL verify edge    │
│   • not confirmed? → spawn hop agent        │
│     hop agent gets:                         │
│       - fault type + injection config       │
│       - upstream's observed symptoms (SQL)  │
│       - relationship type                   │
│       - instruction to find DIFFERENT       │
│         signals on the downstream           │
│   • confirmed → add to queue, fan out       │
│   • rejected → this edge is done            │
│     (target may still be reached via        │
│      another edge in a later round)         │
└─────────────────────────────────────────────┘
    │
    ▼  Post-processing
┌─────────────────────────────────────────────┐
│ Judge agent (promotion-only): per-edge hop  │
│ verdicts are authoritative — judge does NOT │
│ prune them. Its sole job is the global view │
│ one edge can't see: under a system-wide     │
│ cascade (>80% load-gen throughput drop) it  │
│ PROMOTES services hop agents rejected for   │
│ "fewer calls" that are in fact unavailable. │
└─────────────────────────────────────────────┘
    │
    ▼
final_propagation.json (nodes + edges + evidence)
```

Why promotion-only: the hop agents already reason per-edge about error
rate, latency magnitude, fault mechanism, and direction. An LLM judge
re-deciding from rationale text alone reliably *removes genuinely-
degraded* services (it can't tell a real tail-latency spike from
throughput noise without re-querying everything). On the validation set,
judge pruning was strongly net-negative, so only cascade promotion remains.

## How to use

```bash
# Single case
uv run python contrib/scenarios/verifier/eval/audit_case_propagate.py \
    run <case_dir> --model litellm

# Batch
uv run python contrib/scenarios/verifier/eval/audit_case_propagate.py \
    batch <dataset_dir> --run-dir /tmp/verifier-run \
    --model litellm --parallel 8 --case-parallel 30

# Judge (post-processing, after hop agents complete)
uv run python contrib/scenarios/verifier/eval/audit_case_propagate.py \
    judge-batch <dataset_dir> --run-dir /tmp/verifier-run \
    --model litellm --parallel 30

# Resume (cached results are skipped automatically)
uv run python contrib/scenarios/verifier/eval/audit_case_propagate.py \
    batch <dataset_dir> --run-dir /tmp/verifier-run --model litellm
```

## Output

Each case produces under `<run-dir>/<case>/`:

| File | Content |
|------|---------|
| `report.json` | injections, propagated nodes, edges, rounds |
| `all_verdicts.json` | all hop verdicts (confirmed + rejected) with SQL evidence |
| `final_propagation.json` | final graph after judge: nodes, edges, evidence, `added` (cascade promotions), `suggested_remove` (judge prune suggestions, audit-only — not applied) |
| `hops/<from>__<to>/` | per-hop agent session (stdout, stderr, verdict.json) |
| `judge/` | judge agent session |

## Comparing against GT

GT (`causal_graph.json`) uses `component_to_service` to map span-level
nodes to services. When comparing:

1. Extract GT service set via `component_to_service`, minus root causes
2. Extract verifier service set from `final_propagation.json` nodes,
   minus injection seeds
3. Classify discrepancies:
   - **Verifier has, GT doesn't** — either verifier found real
     propagation GT missed, or verifier false positive
   - **GT has, verifier doesn't** — either GT over-labeled (e.g.
     "fewer spans" ≠ service degradation), or verifier capability gap
     (BFS didn't reach it, agent rejected incorrectly)

These are discrepancies to investigate, not accuracy metrics. Do NOT
compute precision/recall treating GT as ground truth — the whole point
is that GT may be wrong.

## Known limitations

- **Infra sinks:** Confirmed infra nodes (mysql, redis, etc.) do not
  fan out — a localized fault on one service's DB path does not
  implicate all DB consumers.
- **Co-deployed subtlety:** Network faults affect all pods on the
  target's k8s node via tc netem, not just the target pod. Modeled
  via `co_deployed` edges but only fires if node-level resource
  degradation is detected.
- **Composition artifacts:** When abnormal window has drastically fewer
  spans with different type mix, percentile changes may reflect
  composition shift rather than real degradation.
