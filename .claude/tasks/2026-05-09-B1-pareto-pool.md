# Task: B-1 — Pareto candidate pool with split inclusion vs deployment

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §11.1](../designs/per-task-evolution-loop.md)
**Reference**: [GEPA summary §3.3, §6.2.1, §6.2.2](../../knowledge/summary_gepa_reflective_evolution.md)
**Assignee**: tdd → implementer

## Objective

Implement the Pareto candidate pool. Inclusion criterion: a candidate is
retained iff it is **best on ≥1 task** vs all other retained candidates
(strict-dominance pruning). This **replaces** "beats incumbent" as the
inclusion gate. The four-floor gate from §8 narrows in scope: it becomes
the **deployment gate** (which pool member is live in production), not the
inclusion gate.

Add new tier-1 atom `tool_query_candidates(scenario) → ParetoFrontier`
that returns the current frontier sorted by per-task win count.

Maintain `.agentm/decisions/<scenario>/tree.jsonl` (parent → child edges)
and a regenerable cache `.agentm/decisions/<scenario>/pareto.json`.

## Inputs

- A-4 candidates schema and writer.
- `tool_propose_change.py` (split inclusion vs deployment).

## Outputs

- New file `src/agentm/extensions/builtin/tool_query_candidates.py`:
  - `MANIFEST.name = "tool_query_candidates"`, registers `tool:query_candidates`.
  - Walks `.agentm/decisions/<scenario>/candidates/*.json`, builds the
    frontier in-process (per-task argmax across candidates), returns
    `[{candidate_id, change_spec, win_tasks: [task_id], score_summary}]`.
- `tool_propose_change.py`:
  - On any successful eval result (whether or not it activates), write a
    candidate record (lift from A-4 to *every* eval-passed call, not only
    `decision="activate"`).
  - **Inclusion** check: candidate is retained iff it wins on ≥1 task.
    Pruned strict-dominators are moved to `candidates/_pruned/<id>.json`
    (kept on disk for audit; not in the frontier).
  - **Deployment** check: existing four-floor gate, now scoped to "is this
    candidate worth making the live one" — the question is unchanged but
    its name in the code becomes `_apply_deployment_gate`.
  - `tree.jsonl` append on every new candidate: `{child, parent, at}`.

## Acceptance Conditions

- [ ] **Fail-stop test**: a candidate that scores 0.30 globally but is the
  unique winner on task 5 is **retained** when a candidate scoring 0.80
  globally is also added. (Property test: pool size > 1 after this scenario.)
- [ ] `tool_query_candidates(scenario="format_fix")` returns the frontier
  with non-empty `win_tasks` per entry.
- [ ] Re-running `tool_query_candidates` is idempotent (cache regenerable).
- [ ] mypy + ruff clean.
- [ ] §11 atom contract validator passes for `tool_query_candidates.py`.

## Notes

- Justification for new fail-stop: GEPA's 35×-rollout efficiency hinges on
  diversity preservation. Wrong → search collapses to greedy → user loses
  the headline property of Phase 2.
- Estimated diff: ~250 lines (one new atom + ~80 lines of edits).
