# verifier — offline FPG candidate construction

The verifier builds a candidate fault propagation graph (FPG) from one case's
normal/abnormal telemetry. It is an offline labeling aid: the workflow gathers
and gates evidence, emits a schema-valid candidate FPG, and writes a review
packet for human adjudication. It does not use ground truth while constructing
the graph.

## Scope

- Multi-fault cases are first-class. Each confirmed seed keeps `source_seed`
  attribution through propagation and review metadata.
- Traces, metrics, and logs are all primary evidence. Missing or sparse telemetry
  limits confidence; it is not silently ignored.
- Baseline noise is allowed. If a signal is noisy before and after with little
  material shift, the inventory marks it as stable background rather than an
  injected effect.
- The verifier avoids hard-coded mechanism templates. Relationship type and fault
  docs help scope checks, but gates judge evidence coverage and consistency, not
  whether a result matches a preset signature.

## Flow

1. `prepare.py` normalizes injections, builds the relationship graph, and creates
   deterministic context:
   - `data_profile`: table schemas, row counts, services, relationships, and
     bounded service-level trace/metric/log statistics.
   - `anomaly_inventory`: normal/abnormal signal changes, including stable
     background/noisy signals.
   - seed observation surfaces: nearby services, link endpoints, modalities, and
     relevant anomalies for each injection.
2. `seed_phase` runs `verify_seed` for every injection. Confirmed service seeds
   become propagation roots; link seeds may add a link node plus an observed
   service-side effect target.
3. `propagate` scans candidate edges per source seed. Candidates come from the
   structural frontier and anomaly-inventory hints; audit can also request
   out-of-graph rechecks. Every accepted hop comes from a gated `verify_hop`
   result.
4. `audit_loop` checks reachability, anomaly coverage, and free exploration until
   a round surfaces no re-dispatch request and no unexplained anomaly, or until
   `max_audit_rounds` is reached. The audit never edits the graph directly; it
   only emits `seed_recheck` or `hop_recheck` requests.
5. `cli.py` writes:
   - `fpg_scenario.json`: candidate FPG conforming to the local FPG profile.
   - `run_meta.json`: execution ledger and audit metadata.
   - `review_packet.json`: candidate FPG, data profile, anomaly inventory,
     evidence ledger, candidate edges, node attribution, and human-review notes.

## Key Design Points

- `HopRecheckRequest` carries optional `rel_type` and `source_seed`. If an audit
  proposes an out-of-graph edge, the verifier uses the provided relationship or
  marks it as `other`; it no longer defaults unknown edges to a caller/callee
  direction.
- `GraphState` stores node attribution separately from FPG nodes. When several
  faults reach the same node, the FPG remains schema-valid while
  `review_notes` flags the node for manual gate/OR/AND adjudication.
- Candidate-edge checks are per source seed. Rechecking one source's edge does
  not automatically invalidate another source's confirmed support for the same
  structural edge.
- Edge removal is verdict-driven: an audit requests a hop recheck; only a gated
  rejected hop can remove an edge, and confirmed support from another source seed
  prevents removing the shared edge.
- The gate prompt is evidence-coverage oriented. It asks whether traces,
  metrics, and logs were discovered and used or shown unavailable, and whether
  timing/magnitude/path alignment and competing faults were handled.

## Module Map

| Module | Role |
|---|---|
| `prepare.py` | Case input assembly plus deterministic profile/anomaly/surface context. |
| `workflow.py` | Thin orchestration: seed verification, candidate propagation, audit fixpoint, redispatch. |
| `state.py` | Pure graph/evidence state, attribution, pruning, final output assembly. |
| `discovery.py` | Effectful adapters for seed/hop agents and gate retries. |
| `audit_map.py` | Gated audit agents for reachability, coverage, and exploration. |
| `lib/case_profile.py` | Bounded telemetry profile and anomaly inventory builders. |
| `lib/candidates.py` | Structural/anomaly-informed candidate-edge enumeration. |
| `lib/fpg.py` | FPG vocabulary mapping and scenario serialization. |
| `lib/schema.py` | TypedDict/Pydantic workflow contracts. |

## Running

```bash
uv run python -m contrib.scenarios.verifier.cli run <case_dir> --out <out_dir>
```

For model selection, pass `--model <profile>` and optionally
`--judge-model <profile>`. `--gate-retries` controls focused retries after a
seed/hop/audit gate rejects an investigation.

## Human Review Packet

`review_packet.json` is the intended handoff artifact. Reviewers should check:

- whether each changed anomaly is covered by the candidate FPG, unrelated, or a
  missed branch;
- whether stable baseline noise was correctly treated as pre-existing;
- whether multi-fault nodes require OR, AND, precondition, or split-node modeling;
- whether any `other` relationship needs a clearer FPG mechanism after manual
  inspection.
