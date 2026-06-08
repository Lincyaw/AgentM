# rca_hfsm Phase 2 — eval results on rca:baseline tasks (C5 re-run)

- **Run date**: 2026-05-14 02:12:45 UTC
- **Branch**: feat/rca-hfsm-phase1 at dad7584 (with this commit's
  changes uncommitted)
- **Manifest**: contrib/scenarios/rca_hfsm/manifest.yaml — 18 atoms.
  Phase 1 baseline + 4 confirm-path LLM judges (C3) + duckdb_sql (C4) +
  the C5 ``judge.investigation_genuine`` consulted by ``rca_finalize``
  at submit-time. Operational investigator persona (C5) rewritten from
  role-description to a 6-step executable workflow.
- **Provider**: openai (LiteLLM via OPENAI_BASE_URL)
- **Model**: Doubao-Seed-2.0-pro
- **Wall time**: 618.3s for 3 cases

## What C5 changed vs C4

C4's diagnosis: 0/3 grader-ok, zero judge invocations, FSM stuck at
INTAKE on every case. The orchestrator skipped ``record_symptom``,
went straight from SQL queries to ``submit_final_report``, and the
structural coverage check was vacuously true on empty symptom sets.

C5's fix is one new judge + one persona rewrite — no structural rules,
no regex:

1. **``judge.investigation_genuine``** — a 5th judge of the same
   Protocol shape as C1's existing four. Its prompt reads the
   trajectory shape (symptom set, hypothesis tree, observations, gate
   mutation counts, proposed final report) and returns
   ``genuine_investigation`` / ``speculation`` / ``unclear``. The
   discipline question — "did this trace actually investigate?" —
   becomes LLM judgment, not a ``len(symptoms) > 0`` check at the
   call site.
2. **``rca_finalize.submit_final_report`` consults the judge.** After
   the existing structural coverage check passes (or vacuously
   passes), the judge fires. Non-``genuine_investigation`` verdicts
   produce a rejection whose text carries the judge's free-text
   reason verbatim so the orchestrator has an actionable next step.
3. **Investigator persona rewritten** from two paragraphs of role
   framing to a 6-step executable workflow: record_symptom →
   list_tables/query_sql → propose_hypothesis (with negative
   prediction) → attach_check → propose_update(confirm) →
   submit_final_report.

The new judge mounts in the manifest alongside the existing four;
workers inherit it for parity (though only the orchestrator's finalize
path consults it).

## Summary

- Total cases: 3
- Grader verdict ``ok``: 0/3
- Cases producing ``submit_final_report``: 2/3
- Cases where the judge **accepted** a final report: **0/3**
- Cases where the judge **rejected** a final report: **2/3** (4
  rejections total — cases 2 and 3, two each)
- Cases reaching FSM ``FINALIZE``: 0/3 (the FSM still depends on
  ``rca.graph.mutated`` advancing, which gate emits only when its
  ``api.events.emit_sync`` is wired — see "Substrate observations"
  below)
- Aggregate token spend: ~725k in, ~15k out
- Aggregate cost: ~$0.38 (Doubao-Seed-2.0-pro at ~$0.5/1M in, ~$1.5/1M
  out)
- Wall time: 10 min 18 s (~3x C4 due to cases 2/3 looping after
  judge rejections — by design)

## Per-case results

### Case 1: 01_mysql_corrupt

- **Prompt**: HTTP POST endpoints across travel/travelplan services
  experiencing SLO violations.
- **Expected**: services=['mysql', 'ts-station-service'],
  fault_kind='network_corrupt'.
- **Verdict**: grader=runtime, score=0.00, elapsed=18.5s.
- **Trajectory shape**: turns=2, symptoms=5, observations=0,
  hypotheses=0, dispatch_agent_calls=0, final_report=False,
  fsm_final=OBSERVE, last_stop=stop.
- **Tokens**: in=5950, out=543.
- **Tool catalog size at LLM time**: **13 tools** (the full set —
  record_symptom, record_observation, propose_hypothesis, attach_check,
  propose_update, submit_final_report, list_tables, query_sql + 5
  sub_agent tools).
- **Mutation kinds**: ``applied=5`` (all record_symptom).
- **Judge effects**: ``investigation_genuine`` never invoked because
  the LLM never called ``submit_final_report``.
- **What happened**: same shape as C4. The orchestrator recorded all
  five symptoms in two turns, then stopped with ``stop_reason=stop``
  (model decided the task was complete). It never proposed a
  hypothesis. The new persona's step 1 ("record symptoms first")
  worked; step 2 ("query the data") did not — the model didn't take
  the OBSERVE-state prompt fragment as an instruction to act, only to
  pause.

### Case 2: 02_pod_failure

- **Prompt**: HTTP GET on contactservice account endpoint experiencing
  SLO violation.
- **Expected**: services=['ts-contacts-service'], fault_kind='pod_failure'.
- **Verdict**: grader=correctness, score=0.70, elapsed=302.6s.
- **Trajectory shape**: turns=25, symptoms=1 attempted (rejected
  ``unknown_tool``), observations=0, hypotheses=0,
  dispatch_agent_calls=0, final_report=True, fsm_final=INTAKE,
  last_stop=tool_calls.
- **Tokens**: in=308605, out=7959.
- **Tool catalog size at LLM time**: **8 tools** (missing
  record_symptom, record_observation, propose_hypothesis,
  attach_check, propose_update). This is a substrate-level
  non-determinism — see "Substrate observations".
- **Tool call sequence**: ``record_symptom`` (returned
  ``unknown_tool``), ``list_tables``, ``query_sql`` × 11,
  ``submit_final_report`` (rejected by judge), ``query_sql`` × 5,
  ``submit_final_report`` (rejected by judge), more ``query_sql``,
  ``stop_reason=tool_calls`` exit.
- **Mutation kinds**: ``{}`` (record_symptom never made it to the
  gate; the LLM's call hit ``unknown_tool`` before gate.apply).
- **``investigation_genuine`` rejections**: **2**.
  - First rejection verdict text:
    ```
    judge=investigation_genuine verdict=speculation reason=No
    symptoms recorded — call record_symptom for each reported error
    before concluding. No hypotheses proposed — use
    propose_hypothesis to generate possible explanations and verify
    them with attach_check. gate_mutations.applied is zero —
    advance the FSM through the proper protocol steps before
    submitting a report.
    ```
  - Second rejection verdict text:
    ```
    judge=investigation_genuine verdict=speculation reason=No
    symptoms are recorded, zero hypotheses were proposed, and
    gate_mutations.applied is 0 indicating no required
    investigation protocol steps were followed.
    ```
- **Final report excerpt** (last attempted): "The SLO violation for
  the HTTP GET ... is caused by database access layer delays for the
  ts-contacts-service. Evidence: 1) Endpoint durations are consistently
  5-11 seconds (well above typical sub-1s SLO thresholds)..."
- **Grader feedback**: named expected service but
  fault_kind="pod_failure" missed (model said "database access layer
  delays" — wrong family). Score 0.70 is from the grader's substring
  scan of the **rejected** ``submit_final_report.args`` — the grader
  doesn't know the judge rejected it.
- **What happened**: The orchestrator's persona-driven first step was
  ``record_symptom``, but the tool wasn't in the LLM's tool catalog
  (only 8 of the 13 installed tools made it to the request). The LLM
  got ``unknown_tool``, moved on to SQL queries, ran 11 of them,
  submitted a final report. **The judge correctly identified
  speculation and rejected.** The LLM tried submitting again with
  slightly more context; **rejected again with similar reasoning.**
  Eventually exited at turn 25.

### Case 3: 03_service_stress

- **Prompt**: HTTP POST on auth-service login endpoint experiencing
  SLO violation.
- **Expected**: services=['ts-auth-service'], fault_kind='jvm_heap_stress'.
- **Verdict**: grader=correctness, score=0.70, elapsed=297.1s.
- **Trajectory shape**: turns=25, symptoms=2 attempted (both
  rejected ``unknown_tool``), observations=0, hypotheses=0,
  dispatch_agent_calls=0, final_report=True, fsm_final=INTAKE,
  last_stop=tool_calls.
- **Tokens**: in=410271, out=6657.
- **Tool catalog size at LLM time**: **8 tools** (same gap as case 2).
- **Mutation kinds**: ``{}``.
- **``investigation_genuine`` rejections**: **2**.
- **Final report excerpt**: "elevated processing latency in the
  ``UserController.getToken`` method of the ts-auth-service under
  stress load".
- **Grader feedback**: named expected service but missed
  fault_kind=jvm_heap_stress; the model said "stress load" but didn't
  trace to JVM heap pressure specifically. Same scoring caveat as
  case 2.
- **What happened**: same shape as case 2. Two ``submit_final_report``
  attempts, both rejected with the same "no symptoms, no hypotheses,
  gate_mutations.applied=0" reason. The judge worked; the protocol
  couldn't be executed because the relevant tools weren't visible.

## Cross-case patterns

### C5 — investigation_genuine judge effect

The judge fires every time ``submit_final_report`` is called. Across
the run:

- **4 judge invocations** (cases 2 + 3, two each).
- **4 rejections, 0 acceptances.** All four returned
  ``verdict=speculation`` with structurally similar reasons (no
  symptoms, no hypotheses, no applied mutations).
- The judge's reason text **propagated to the orchestrator** as the
  ``submit_final_report`` tool result. We can see the orchestrator
  read it (turn 12 → 13 in case 2, etc.) and **tried to course-correct
  by running more SQL queries** — but it still didn't call
  ``record_symptom`` again (presumably because the model remembered the
  earlier ``unknown_tool`` error). It then re-submitted a slightly
  refined report; rejected again.
- **The judge's verdict is consistent with the actual trajectory:**
  zero hypotheses proposed, zero gate-applied mutations of the kind
  the protocol expects. Even if record_symptom had succeeded, the
  judge would still have called this "speculation" because the LLM
  never proposed a falsifiable hypothesis.

### Other 4 judges (satisfied / coverage / independence / falsified_genuinely)

Still 0 invocations. They sit on the gate's confirm/refute paths,
which require ``propose_hypothesis`` + ``attach_check`` to have run
first. No case got that far. Same as C4.

### Tool catalog non-determinism

Case 1 saw 13 tools; cases 2 and 3 saw 8 tools. Same scenario, same
manifest, same provider — the difference is which subset of installed
extensions' ``api.register_tool`` calls landed in the active catalog
by the time the loop started its first request. This is a
substrate-level race we did not investigate in C5 (the plan forbids
core changes). When tool_count=13, the LLM can follow the persona's
step 1. When tool_count=8, it can't, and falls back to free-form SQL
+ submit_final_report — which is exactly the case the new judge was
built to catch.

### Mutation-event emission on the bus

Case 1 emitted 5 ``rca.graph.mutated`` events (one per
record_symptom). Cases 2 and 3 emitted **zero** — record_symptom
failed at the loop layer with ``unknown_tool`` before reaching
gate.apply. The judge's view of "gate_mutations.applied=0" was
literally correct for those cases. (The earlier C4 report claimed
the bus was missing mutation events generally; with C5 visibility, we
can see it's specifically the cases where the tool was unreachable.)

### Pass rate change vs C4

- C4: 0/3 grader-ok, 2/3 ``submit_final_report`` emitted, 0 judge
  invocations, FSM stuck at INTAKE on the 2 cases that finalized.
- C5: 0/3 grader-ok, 2/3 ``submit_final_report`` emitted (but
  **rejected**), 4 judge invocations, FSM still stuck at INTAKE.

**The grader pass-rate did not move.** What moved is the discipline
floor: in C4 the LLM's free-form final reports went through and were
graded on substring match; in C5 the same reports are intercepted by
the judge, which correctly identifies them as speculation. The grader
score is unchanged because the grader doesn't know about the judge —
it scores the args of any ``submit_final_report`` call regardless of
whether the tool result accepted or rejected. **The judge fired; the
orchestrator could not produce a non-speculation trajectory.**

## Honest answer to the diagnosis question

**Was the persona rewrite enough?** No. In case 1 the persona's
step-1 instruction worked (5 record_symptom calls, then stop without
step 2). In cases 2 and 3 the persona's step-1 didn't matter because
the tool wasn't available. The persona is necessary but not
sufficient.

**Did the judge's rejections do the heavy lifting?** Partially. The
judge fired on every rejected report, the reason text reached the
orchestrator, but the orchestrator could not act on the reason
because the protocol tools (``record_symptom``,
``propose_hypothesis``) were not in its catalog. So the judge's
"please call record_symptom first" was technically un-actionable
those turns. **The judge's value is now visible** — it stopped two
speculative reports from being treated as legitimate finalizations —
**but its corrective channel is blocked** by the tool-visibility
issue below.

## Substrate observations (outside C5 scope)

- **Non-deterministic tool catalog size.** Case 1 had 13 tools at
  LLM time; cases 2 and 3 had 8. The scenario, manifest, model, and
  prompt are identical. The 8-tool subset is missing exactly the 5
  rca_evidence_tools (record_symptom, record_observation,
  propose_hypothesis, attach_check, propose_update). This points at
  a race in session bootstrap — probably between async install of
  rca_evidence_tools and the catalog freeze step — that drops
  registrations silently. **Without those tools the FSM cannot
  advance and the judges cannot fire on confirm/refute paths.** This
  was masked in C4 by the broken grader (every case looked like
  ``runtime``); with C5's judge visibility it surfaces.

- **task_meta.task_id wiring still broken** (same as C4). Workaround
  via local grader by case-cwd remains.

- **Judge invocations are not on the bus.** Adding
  ``rca.judge.invoked`` telemetry from inside ``judge()`` would let
  this report give a clean "judge fired N times" line for each of the
  5 judges. Today we infer C5's judge from
  ``submit_final_report``'s rejection text and the other 4 from
  ``mutation_kinds.downgraded``. Phase 3 candidate (still).

## Phase 4 candidates

In rough priority order based on this run's data:

1. **Investigate the tool-catalog race.** Same scenario, same
   manifest, different run → different tool count. This is the
   single blocker preventing the FSM from advancing on cases 2 and
   3. The fix is substrate-level and the bug existed before C5; C5
   just made it visible. Worth a focused investigation before any
   further policy-layer tuning.

2. **Add ``rca.judge.invoked`` bus telemetry.** Without it, every
   future eval has to grep tool-result text or downgrade counts to
   guess what each judge did. C5 specifically benefits — its single
   judge call per finalize is invisible in the current trace except
   through rejection text.

3. **Re-run after #1.** Once tool-visibility is deterministic, the
   persona's 6-step workflow should reach
   propose_hypothesis/attach_check on at least cases 2 and 3.
   That's when the confirm-path judges (satisfied / coverage /
   independence / falsified_genuinely) finally get exercised on
   real traces.

4. **Surprise candidate: the OBSERVE-state prompt fragment.**
   Case 1 (where all tools were present) recorded 5 symptoms and
   then stopped at turn 2 — the model decided the conversation was
   over. The OBSERVE prompt fragment doesn't push hard enough.
   This is the cheapest knob to twist if #1 makes cases 2 and 3
   look like case 1.

## Where the LLM judges still wouldn't catch what a human RCA expert would

Same caveat as the C4 report. The judges operate on graph structure
and trajectory shape. They do not yet critique the **content** of a
``fault_kind`` claim (pod_failure vs pod_unavailable vs
deployment-scaling; jvm_heap_stress vs application-layer load). The
case-2 "database access layer delays" → fault_kind=pod_failure miss
and case-3 "stress load" → fault_kind=jvm_heap_stress miss are
exactly that family of error. A fault_kind disambiguator judge
modelled after the existing rca scenario's ``critic.md`` cheatsheet
remains an open Phase 4 candidate, deferred until the substrate gap
above is closed (otherwise it would never get the chance to fire).
