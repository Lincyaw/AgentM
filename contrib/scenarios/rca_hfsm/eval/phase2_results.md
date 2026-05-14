# rca_hfsm Phase 2 — eval results on rca:baseline tasks (C4 re-run)

- **Run date**: 2026-05-14 01:22:10 UTC
- **Branch**: feat/rca-hfsm-phase1 at cfcf5eb (C4 manifest fix landing in
  this commit)
- **Manifest**: contrib/scenarios/rca_hfsm/manifest.yaml — 17 atoms.
  Phase 1 baseline + 4 LLM-mode judges (C3) + duckdb_sql at orchestrator
  level (C4) + worker inheritance for store/judges/gate/evidence-tools/
  duckdb_sql/worker_finalize.
- **Provider**: openai (LiteLLM via OPENAI_BASE_URL)
- **Model**: Doubao-Seed-2.0-pro
- **Wall time**: 273.3s for 3 cases

## Selection criteria

Same as the C3 run. The only YAML-defined task suite is
`contrib/scenarios/rca/eval/baseline/tasks/` (three cases). The 50-case
`ops-lite-fixed-50` set lives in HuggingFace-dataset form and is driven
by `rca llm-eval run` from `rcabench-platform`, not this runner. The
three baseline tasks span: a multi-service propagation fault (mysql
network_corrupt) for judge.independence/judge.coverage stress, a
single-service pod-failure for judge.satisfied focus, and a JVM-level
stress case for judge.falsified_genuinely on container-vs-JVM
disambiguation.

## Manifest baseline (C4 fix)

C3 ran on a manifest that was missing the data-access tools
(`agentm_rca.tools.duckdb_sql`) and sub-agent inheritance entries for
`rca_falsification_gate` plus the four judge atoms. Workers refused to
start because `rca_evidence_tools` declares
`requires=("rca_falsification_gate",)` and the gate wasn't in the
worker's inheritance list. **C4 fixes both gaps**: the orchestrator now
carries `list_tables` / `query_sql` directly, and workers inherit the
store → judges → gate → evidence-tools → duckdb_sql → worker_finalize
chain in the same dependency order as the orchestrator. This eval
re-run is the first time the rca_hfsm orchestrator has had SQL access
under the LLM-native gate; compare the per-case rows below against the
C3 results in git history (commit cfcf5eb).

`AGENTM_RCA_DATA_DIR` is now wired per-case from each YAML's
`input.fixtures[0]` so `query_sql` resolves to the right dataset
directory.

## Summary

- Total cases: 3
- Grader verdict `ok`: 0/3
- Cases producing `submit_final_report`: 2/3
- Cases reaching FSM `FINALIZE`: 0/3
- Cases with infrastructure errors: 0/3
- Aggregate token spend: ~245k in, ~8k out
- Aggregate cost (rough, Doubao-Seed-2.0-pro at ~$0.5/1M in,
  ~$1.5/1M out): ~$0.13

## Per-case results

### Case 1: 01_mysql_corrupt

- **Prompt summary**: HTTP POST endpoints across travel/travelplan
  services experiencing SLO violations.
- **Expected**: services=['mysql', 'ts-station-service'],
  fault_kind='network_corrupt'.
- **Verdict**: grader=runtime, score=0.00, elapsed=26.5s.
- **Trajectory shape**: turns=2, symptoms=5, observations=0,
  hypotheses=0, dispatch_agent_calls=0, final_report=False,
  fsm_final=OBSERVE, last_stop=stop.
- **Tokens**: in=5914 out=982.
- **Mutation kinds**: `applied=5` (all `record_symptom`).
- **Grader feedback**: no `submit_final_report` reached.
- **What happened**: The orchestrator recorded all five reported
  symptoms in two turns, then stopped with `stop_reason=stop` (model
  decided the conversation was complete). It never called `list_tables`
  or `query_sql`, never proposed a hypothesis, never dispatched a
  worker. The OBSERVE-state prompt fragment told it to gather L1 facts
  but did not surface that SQL tools were available — under the old
  fragment the model treated "the symptoms are recorded" as "the task
  is done".

### Case 2: 02_pod_failure

- **Prompt summary**: HTTP GET on contactservice account endpoint
  experiencing SLO violation.
- **Expected**: services=['ts-contacts-service'], fault_kind='pod_failure'.
- **Verdict**: grader=correctness, score=0.70, elapsed=123.3s.
- **Trajectory shape**: turns=10, symptoms=0, observations=0,
  hypotheses=0, dispatch_agent_calls=0, final_report=True,
  fsm_final=INTAKE, last_stop=tool_calls.
- **Tokens**: in=101216 out=3717.
- **Mutation kinds**: `{}` — zero graph mutations.
- **Final report excerpt**: "The root cause … is insufficient replica
  count for the ts-contacts-service. The service is running with only
  1 replica, which is unable to handle the incoming request volume."
- **Grader feedback**: named expected service (ts-contacts-service)
  but called the fault `insufficient replica count` rather than
  `pod_failure`.
- **What happened**: The orchestrator bypassed the FSM entirely. It
  never called `record_symptom`, so the FSM stayed in INTAKE and the
  coverage check on `submit_final_report` was vacuously satisfied
  (zero unexplained symptoms because zero symptoms recorded). 101k
  input tokens over 10 turns strongly implies it spent the budget
  running `query_sql` and reading data — but those tool calls do not
  produce `rca.graph.mutated` events, so the mutation-kinds row is
  empty. The model went straight from "look at data" to "submit
  report" with no symptom / observation / hypothesis trail. Service
  was named correctly; fault_kind was named in human language
  ("insufficient replicas") rather than the rcabench enum
  (`pod_failure`).

### Case 3: 03_service_stress

- **Prompt summary**: HTTP POST on auth-service login endpoint
  experiencing SLO violation.
- **Expected**: services=['ts-auth-service'], fault_kind='jvm_heap_stress'.
- **Verdict**: grader=correctness, score=0.70, elapsed=123.3s.
- **Trajectory shape**: turns=12, symptoms=0, observations=0,
  hypotheses=0, dispatch_agent_calls=0, final_report=True,
  fsm_final=INTAKE, last_stop=tool_calls.
- **Tokens**: in=137935 out=3501.
- **Mutation kinds**: `{}` — zero graph mutations.
- **Final report excerpt**: "The root cause … is excessive load/stress
  on the ts-auth-service. The service is processing a high volume of
  login requests, leading to increased latency in the
  `UserController.getToken` method (which accounts for the majority
  of the …)."
- **Grader feedback**: named expected service (ts-auth-service) but
  attributed the fault to load/stress on the application layer rather
  than `jvm_heap_stress` specifically. The container-vs-JVM
  disambiguation this case was meant to stress never reached the
  judge.
- **What happened**: same shape as case 2. Bypassed FSM, large input
  token count consistent with extensive SQL probing, jumped to final
  report without graph mutations.

## Cross-case patterns

### Judge call patterns

**Zero judge invocations across the run.** The judges fire from inside
`gate.apply` on the `propose` / `attach_check` / `refute` / `confirm`
update paths. Cases 2 and 3 made zero such calls (no graph mutations
of any kind); case 1 made only `record_symptom` calls which the gate
applies without consulting any judge (record_symptom is a pure L1
write, never gate-judged). So the LLM-native judges are wired
correctly — the manifest smoke test confirms each `rca.judge.*`
service resolves to a `_LlmJudge` instance — but the orchestrator
never reached any state in which the gate would consult them.

This is the same result as C3 in terms of judge-call counts, but the
reason has shifted. In C3 the workers couldn't launch (manifest
missing the gate + judges). In this run the orchestrator simply chose
not to use the protocol, despite having every tool available.

### Downgrade patterns

No `downgraded` mutation events across the run. Vacuously: nothing
got past `propose` so no `confirm` / `refute` could be downgraded.

### FSM state distribution

Final FSM states: `INTAKE=2`, `OBSERVE=1`. Zero traces reached
HYPOTHESIZE / VERIFY / JUDGE / FINALIZE. The FSM machinery did its
job — it never advanced to FINALIZE — but the orchestrator submitted
final reports anyway via the coverage-check loophole (no symptoms
recorded → no symptoms unexplained → coverage passes).

### Honest observations

This run measures one thing cleanly: **whether the C4 manifest fix
unblocks the orchestrator that C3 documented as stuck.** It does not.
The orchestrator's failure mode shifted from "blocked by missing
worker extension" (C3) to "ignores the FSM protocol and submits
free-form final reports without graph mutations" (this run). Both
failure modes prevent the LLM-native judges from firing on a
confirm/refute path, so the Phase 2 acceptance question (§8 of the
design doc — "do the LLM-native judges *behave reasonably* when
exercised?") cannot be answered from baseline behaviour alone. The
substrate is correct (services register, smoke tests pass, install
order respected); the policy layer that should drive the LLM through
INTAKE → OBSERVE → HYPOTHESIZE → VERIFY → JUDGE → FINALIZE is too
weak under the current per-state prompt fragments.

### Where an LLM-driven judge would clearly help over regex

Not exercised in this run because the gate paths that consult judges
were never reached. The closest thing observed: in case 2, the
orchestrator's free-form final report says "insufficient replica
count" rather than the rcabench enum `pod_failure`. A Phase-1 regex
would never have caught this — it would have to know every fault_kind
synonym. A `judge.coverage` invocation on the (would-be) confirm path
could in principle catch "your conclusion uses a vocabulary the
symptoms can't validate" — but again, this is hypothetical; the gate
was never asked.

### Where the LLM judges still wouldn't catch what a human RCA expert would

The same case 2 failure — "insufficient replicas" vs `pod_failure` —
is a fault-kind disambiguation problem, not a coverage problem. The
correct disambiguator is the one the existing rca scenario's critic
persona spells out in `agents/critic.md` ("pod_failure → pod restart
count / phase transitions visible; pod_unavailable → pod status
reachable but readiness=0"). None of the four current judges know
about pod_failure vs pod_unavailable vs deployment-scaling. They
operate on graph structure (was the falsification real, are the
checks independent, is every symptom covered) — they don't critique
the *content* of a fault_kind claim. A human RCA expert would
immediately challenge "the service is running with only 1 replica" by
asking for `kubectl get pod -w` output showing the restart-count
delta. The current judges have no analogue.

### Phase 3 candidates

1. **Strengthen per-state prompt fragments** to (a) name the data-access
   tools available in each state and (b) make explicit that
   `submit_final_report` requires a non-empty symptom set. The OBSERVE
   fragment in particular should require at least one `query_sql` call
   before allowing transition to HYPOTHESIZE. This is the cheapest
   change and the one most likely to unblock the next layer.
2. **Tighten the FINALIZE-coverage check** to require symptoms recorded
   > 0 (rather than only "every recorded symptom is explained"). This
   plugs the loophole cases 2 and 3 used. Belongs in `rca_finalize`,
   not in any judge — it's a structural precondition, not a semantic
   one.
3. **Add `rca.judge.invoked` bus telemetry** so the eval runner can
   count judge calls directly rather than via the mutation-kinds proxy.
   Design §3.4 envisaged this; Phase 2 deferred it. With the
   fragmentation / FSM-bypass issues above, this becomes urgent —
   without it we cannot distinguish "judges fired and approved" from
   "judges were never consulted because the gate path wasn't taken".
4. **Add a fault_kind disambiguator judge** modeled after the rca
   critic persona's family cheatsheet. This would address the case-2
   "insufficient replicas" failure mode that the four current judges
   structurally cannot catch.

### Substrate notes (unchanged from C3)

- `task_meta.task_id` wiring in core is still broken (`SessionReadyEvent`
  declares the field but the emit at
  `src/agentm/core/runtime/session_factory.py` L366 does not forward
  `eval_task_id`). This runner works around the gap with a local
  scoring helper that locates the trace by case-cwd. Independent
  substrate fix; out of scope here.
- Judge telemetry is still missing on the bus (see Phase 3 candidate
  #3).
