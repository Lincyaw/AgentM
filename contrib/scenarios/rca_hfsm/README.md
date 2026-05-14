# agentm-rca-hfsm

A hypothesis-driven, falsification-gated RCA scenario. Sibling — not
replacement — of `contrib/scenarios/rca/`: the two coexist so they can be
A/B'd on the same eval set.

The design lives in
[`.claude/designs/hypothesis-driven-rca.md`](../../../.claude/designs/hypothesis-driven-rca.md);
this README is only orientation.

## Mental model

Root-cause analysis is process-of-elimination: propose hypotheses, predict
observables, gather evidence, eliminate the ones contradicted by evidence,
accept the one that explains every recorded symptom and has at least one
credible refutation attempt that failed.

Three structural commitments back the framing:

1. **Hypothesis lifecycle is a DAG of update operators**, not a confirm/refute
   binary. See design §3.3.
2. **Falsification discipline is enforced by the FSM gate**, not asked for by
   the prompt. See design §7.
3. **Semantic structure (the FSM) is orthogonal to context structure
   (sessions)**. The scenario uses an L1 + L2 + L3 layered topology to keep
   L2 token growth proportional to hypothesis events instead of tool calls.
   See design §5.

## Phase 1 status

Phase 1 (this commit set) lands the L1 schema and the single-writer
hypothesis-graph store. Subsequent commits land the falsification gate, the
evidence-tool surface, the FSM policy + brief builder, and the full
manifest + investigator persona. Phasing detail in
[`.claude/plans/2026-05-13-rca-hfsm-phase1.md`](../../../.claude/plans/2026-05-13-rca-hfsm-phase1.md).

## Layout

```
contrib/scenarios/rca_hfsm/
├── pyproject.toml                 — workspace member
├── src/agentm_rca_hfsm/
│   ├── schema.py                  — Symptom, Prediction, Hypothesis, ...
│   └── atoms/
│       └── rca_hgraph_store.py    — single-writer L1 store atom
└── tests/                         — scenario-local fail-stop tests
```
