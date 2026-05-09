# Task: Layer B Foundation Completion Report (B-3, B-1, B-6)

**Date**: 2026-05-09
**Status**: COMPLETE
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)

## Tasks completed

- B-3 — Generalized ChangeSpec kinds (validator dispatch) — DONE
- B-1 — Pareto candidate pool with split inclusion vs deployment — DONE
- B-6 — Rollout budget tracking — DONE

Order: B-3 first (structural foundation for kind dispatch), B-1 second
(reorganization sharing the same `_execute`), B-6 last (cap-check
plumbing layered on top).

## Lines changed per file

| File | Insertions | Deletions |
|---|---:|---:|
| `src/agentm/extensions/builtin/tool_propose_change.py` | ~280 | ~70 |
| `src/agentm/extensions/builtin/tool_eval_run.py` | ~38 | 0 |
| `src/agentm/extensions/builtin/tool_query_candidates.py` (new) | 196 | 0 |
| `contrib/extensions/changespec_validators/__init__.py` (new) | 12 | 0 |
| `contrib/extensions/changespec_validators/atom_source.py` (new) | 73 | 0 |
| `contrib/extensions/changespec_validators/system_prompt.py` (new) | 109 | 0 |
| `contrib/extensions/changespec_validators/manifest_field.py` (new) | 178 | 0 |
| `contrib/extensions/changespec_validators/manifest_extensions.py` (new) | 144 | 0 |
| `tests/integration/test_pareto_inclusion.py` (new) | 219 | 0 |
| `tests/integration/test_per_task_evolution.py` | 9 | 5 |
| **Approx total** | **~1258 insertions** | **~75 deletions** |

(Diff size came in above the 1000 LoC heuristic primarily because the
four validator files account for ~500 LoC of comment-heavy single-purpose
modules. The substantive change footprint in `tool_propose_change.py` is
~280 LoC across three task scopes — within plan estimates.)

## New files

- `src/agentm/extensions/builtin/tool_query_candidates.py` — read-only
  Pareto-frontier projection atom (B-1).
- `contrib/extensions/changespec_validators/{atom_source,system_prompt,manifest_field,manifest_extensions,__init__}.py`
  (B-3).
- `tests/integration/test_pareto_inclusion.py` — fail-stop test for
  Pareto inclusion correctness (B-1).

## Key design decisions

1. **Validator location** — under `contrib/extensions/changespec_validators/`,
   not `src/agentm/`. They encode scenario-shape policy, not SDK
   mechanism, and are discovered via importlib at dispatch time. They
   are not atoms (no MANIFEST), so the §11 "no atom-to-atom imports"
   rule does not bind them.
2. **Inclusion vs deployment split** — candidate is written to
   `candidates/<id>.json` *before* the deployment-gate check. Even when
   the four-floor gate rejects the swap, the candidate stays in the
   pool for future search (the GEPA-defining property).
3. **Pareto pruning is strict** — a candidate must be the strict argmax
   on >=1 task to be on the frontier. Ties on a task contribute no
   inclusion claim; this avoids the degenerate case where two identical
   candidates would tie indefinitely.
4. **Pruning is non-destructive** — dominated candidates get a
   `<id>.json.pruned` sidecar (zero-byte flag) instead of being deleted.
   The .json is preserved for git history and audit. The reverse
   transition (re-include after a later candidate changed who wins on
   a task) drops the flag.
5. **Budget split between atoms** — `tool_eval_run` continues to be the
   sole writer of `usd_used` (it owns child-trace cost). `tool_propose_change`
   now writes `rollouts_used` (one per call). Both atoms read the file
   and refuse before doing work when their respective cap is hit. No
   double-counting of cost.
6. **`scenario_compose` remains rejected** — the existing
   `not_yet_implemented` rejection moved from a generic "non-atom_source"
   guard to a specific scenario_compose-only branch. The
   `test_propose_change_rejects_without_evidence` test was updated to
   point at scenario_compose so the assertion still has a referent.

## Blockers

None. No `agentm.core.*` modification was required. The constitution
glob `.agentm/decisions/**` already covers `tree.jsonl`,
`candidates/*.json.pruned`, and `budget.json`.

## Test status

`uv run pytest --tb=short` -> **99 passed, 1 skipped, 14 deselected**
(was 97 passed; +2 new tests in `test_pareto_inclusion.py`).

New fail-stop count: **+1** — Pareto inclusion correctness
(`test_pareto_inclusion_retains_niche_winner`). The companion
`test_pareto_dominated_candidate_excluded` is the complement-sanity
counterpart pinning the inclusion criterion from both sides.

`uv run ruff check` and `uv run mypy` both clean on every touched file.
