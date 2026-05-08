# Task: V0 integration smoke + .claude/index.yaml registration

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-cognitive-audit-v0.md)
**Design**: [design](../designs/llmharness-cognitive-audit.md)
**Assignee**: reviewer + implementer

## Objective

Run lint / type-check / index-validate, do a manual end-to-end
smoke against the real AgentM CLI, and register this plan + tasks
under the `cognitive_audit` concept in the top-level
`.claude/index.yaml`. **Pytest gating is intentionally NOT part of
this round** — tests are deferred to a later pass.

## Inputs

- Read: completed Phase 1–4 deliverables.
- Read: scenarios/llmharness/CLAUDE.md "Dev-loop stages" table.
- Read: AgentM CLAUDE.md "End-to-end testing methodology" (for the
  manual trajectory inspection procedure, even though no pytest e2e
  exists this round).

## Outputs

- Modified: `.claude/index.yaml` — under
  `concepts.cognitive_audit`:
  - `plans: ["plans/2026-05-08-llmharness-cognitive-audit-v0.md"]`
  - `tasks:` list every task file from this plan (already populated
    in a prior step; re-verify it matches on-disk task files after
    Phase 1–4 land)
- Manual smoke run report (informal, not a committed file):
  - `bash scripts/tick_worker.sh` with `LLMHARNESS_PROVIDER=agentm`
    in a temp sandbox confirms a child session spawns and a verdict
    persists
  - Inspect `<sandbox>/.llmharness/verdicts/` (or wherever
    `store.append_verdict` writes) and the child's
    `.agentm/observability/<child_id>.jsonl` for the audit trace

## Acceptance Conditions

- [ ] `uv run ruff check src` — clean
- [ ] `uv run ruff format src --check` — clean
- [ ] `uv run mypy src/llmharness` — strict-clean
- [ ] `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml` — 0 violations
- [ ] Manual smoke shows: `TurnEndEvent` triggers child session;
      child emits a Verdict; if drift, next-prompt `system` field
      gains the `[harness] ...` line
- [ ] `.claude/index.yaml` updated and consistent (no broken refs);
      `tasks:` list matches the 5 on-disk task files
      (01-schema, 02-cards-module, 03-agentm-adapter,
      04-scenario-manifest, 06-integration)
- [ ] Reviewer sign-off that the implementation matches design §2
      authority constraints (advisor not controller, never mutates
      trajectory/plan/tools/model, default silent)

## Notes

- If the manual smoke fails because the AgentM `ExtensionAPI`
  surface used by `adapters/agentm.py` differs from what the design
  sketch assumed, file a feedback note and pause for architect
  review — do NOT silently widen the design.
- No `pytest` step in V0 acceptance: testing was deliberately
  deferred (see plan deliverable 5 / Risk row "Schema additions
  break rca-autorl serializer"). When the testing pass lands, it
  should target the fail-stop positions enumerated in the prior
  test scoping conversation (schema additivity / cards axis_hint
  coverage / adapter trigger correctness / e2e injection chain).
- Out of scope: any V1 work — Claude Code three-axis migration,
  training-data export script, pre-filter, cross-session memory,
  evaluation harness.
