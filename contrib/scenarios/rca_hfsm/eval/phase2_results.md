# rca_hfsm Phase 2 — eval results on rca:baseline tasks

- **Run date**: 2026-05-13 16:22:20 UTC
- **Branch**: feat/rca-hfsm-phase1 at b86fe49
- **Manifest**: contrib/scenarios/rca_hfsm/manifest.yaml (4 LLM-mode judges mounted before the gate)
- **Provider**: openai
- **Model**: Doubao-Seed-2.0-pro
- **Wall time**: 267.9s for 3 cases

## Selection criteria

The plan asked for up to 10 representative cases from ``contrib/scenarios/rca/eval/tasks/``. The only YAML-defined task suite in the existing rca scenario is ``contrib/scenarios/rca/eval/baseline/tasks/`` and it holds exactly three cases. The 50-case ``ops-lite-fixed-50`` set lives in HuggingFace-dataset form, is driven by ``rca llm-eval run`` from ``rcabench-platform``, and is not a YAML task list this scenario can read directly. Per the plan's explicit fall-back ("If ``tasks/`` has fewer than 10, runs all available"), this report covers all three baseline tasks: a multi-service propagation fault (mysql network_corrupt) that stresses judge.independence and judge.coverage, a tight single-service pod-failure that stresses judge.satisfied, and a JVM-level stress case that stresses judge.falsified_genuinely on a container-vs-JVM disambiguation.

## Important caveat on rca_hfsm capability gap

The rca scenario this eval suite was built for has the ``duckdb_sql`` tool wired in and reads parquet fixtures (``abnormal_traces.parquet``, ``metrics_sum.parquet``, etc.) to investigate. **The rca_hfsm scenario has NO data-access tools** — its orchestrator only sees the 5 graph-mutation tools (``record_symptom`` / ``record_observation`` / ``propose_hypothesis`` / ``attach_check`` / ``propose_update``) plus ``submit_final_report`` and ``dispatch_agent``. Workers spawned via ``dispatch_agent`` inherit ``operations_local`` but no LLM-callable SQL/bash tools either. This means the eval here is **not measuring diagnostic accuracy** — accuracy is bounded by whatever the orchestrator can reason out from the user message alone. What this eval IS measuring is **judge behaviour and FSM trajectory shape** when LLM-mode judges replace the Phase-1 structural rules. Treat pass-rate numbers below as lower-bound trajectory-completion signals, not as RCA competence signals.

## Summary

- Total cases: 3
- Grader verdict ``ok``: 0/3
- Cases producing ``submit_final_report``: 2/3
- Cases reaching FSM ``FINALIZE``: 0/3
- Cases with infrastructure errors: 0/3

## Per-case results

### Case 1: 01_mysql_corrupt

- **Prompt summary**: The following API endpoints are experiencing possible SLO violations and need investigation:
- **Expected**: services=['mysql', 'ts-station-service'] fault_kind='network_corrupt'
- **Verdict**: grader=runtime  score=0.00  elapsed=20.7s
- **Trajectory shape**: turns=2, symptoms=5, observations=0, hypotheses=0, dispatch_agent_calls=0, final_report=False, fsm_final=OBSERVE, last_stop=stop
- **Tokens (orchestrator only)**: in=5642 out=606
- **Mutation kinds**: applied=5
- **Judge calls**: none observed on bus (see Phase 3 note on judge telemetry)
- **Trace**: ``contrib/scenarios/rca_hfsm/eval/traces/01_mysql_corrupt/.agentm/observability/315e2ddb7248455a.jsonl``
- **Grader feedback**: No submit_final_report tool_call observed in the case trace; orchestrator never reached FINALIZE.

### Case 2: 02_pod_failure

- **Prompt summary**: The following API endpoints are experiencing possible SLO violations and need investigation:
- **Expected**: services=['ts-contacts-service'] fault_kind='pod_failure'
- **Verdict**: grader=correctness  score=0.00  elapsed=110.9s
- **Trajectory shape**: turns=5, symptoms=1, observations=0, hypotheses=0, dispatch_agent_calls=3, final_report=True, fsm_final=INTAKE, last_stop=tool_calls
- **Tokens (orchestrator only)**: in=8500 out=4147
- **Mutation kinds**: {}
- **Final report root_cause** (excerpt): A possible SLO violation is reported for the HTTP GET endpoint http://ts-ui-dashboard:8080/api/v1/contactservice/contacts/account/{accountId}. Further root cause investigation is blocked because the required 'rca_falsification_gate' extension is not loaded, which prevents spawning child diagnostic a
- **Judge calls**: none observed on bus (see Phase 3 note on judge telemetry)
- **Trace**: ``contrib/scenarios/rca_hfsm/eval/traces/02_pod_failure/.agentm/observability/146069900c114111.jsonl``
- **Grader feedback**: missed services (expected one of ['ts-contacts-service']); missed fault_kind (expected substring 'pod_failure').

### Case 3: 03_service_stress

- **Prompt summary**: The following API endpoints are experiencing possible SLO violations and need investigation:
- **Expected**: services=['ts-auth-service'] fault_kind='jvm_heap_stress'
- **Verdict**: grader=correctness  score=0.00  elapsed=136.2s
- **Trajectory shape**: turns=6, symptoms=1, observations=0, hypotheses=0, dispatch_agent_calls=3, final_report=True, fsm_final=INTAKE, last_stop=tool_calls
- **Tokens (orchestrator only)**: in=9644 out=5140
- **Mutation kinds**: {}
- **Final report root_cause** (excerpt): Unable to determine the root cause of the HTTP POST http://ts-ui-dashboard:8080/api/v1/users/login SLO violation due to a critical RCA platform configuration error. The required `rca_falsification_gate` extension, which is a dependency for the `rca_evidence_tools` extension needed to spawn child inv
- **Judge calls**: none observed on bus (see Phase 3 note on judge telemetry)
- **Trace**: ``contrib/scenarios/rca_hfsm/eval/traces/03_service_stress/.agentm/observability/dadc3e620f8e43ee.jsonl``
- **Grader feedback**: missed services (expected one of ['ts-auth-service']); missed fault_kind (expected substring 'jvm_heap_stress').

## Cross-case patterns

### Judge call patterns

No ``rca.judge.invoked`` events were observed on the bus. The judges in Phase 2 do not currently emit a structured-bus event from inside ``judge()`` — judge calls happen inside ``gate.apply`` and are observable only through their effect on ``rca.graph.mutated`` (applied vs downgraded). Adding an explicit ``rca.judge.invoked`` event is Phase 3 work (design §3.4 envisaged observability JSONL emission per judge call but Phase 2 did not yet wire it up). The mutation-kinds row above is the proxy.

### Downgrade patterns

No ``downgraded`` mutation events observed across the run. Either (a) the orchestrator never proposed a ``confirm`` whose preconditions failed, or (b) traces terminated before reaching ``_apply_confirm``. Inspect ``mutation_kinds`` per case to disambiguate.

### FSM state distribution

Final FSM states: INTAKE=2, OBSERVE=1.

### Surprises and Phase 3 candidates

Observations are recorded honestly per the design doc's acceptance gate (§8): the question for Phase 2 was whether the refactor *behaves reasonably*, not whether it matches a target pass-rate on the first run. Surprises noted below describe what we saw, not a target.

- **2/3 ``submit_final_report`` calls were platform-error reports, not RCA verdicts.** The workers failed to launch because the production manifest's ``sub_agent.inherit_extensions`` block does not list ``rca_falsification_gate``, yet ``rca_evidence_tools`` (also inherited) declares ``requires=("rca_falsification_gate",)``. This is a **pre-existing Phase 1 manifest bug**, not a regression introduced by Phase 2 C3 — the C3 commit added only the four judge atoms before the gate. Fix is one manifest line; outside this commit's surgical scope. Without the fix the orchestrator never reaches HYPOTHESIZE/VERIFY/JUDGE on tasks that require dispatch_agent — every case stops at INTAKE or OBSERVE.
- **Zero ``propose_hypothesis`` calls across all cases.** Because workers couldn't run, the orchestrator never received observations that would seed hypotheses, and the investigator persona's discipline ("every hypothesis needs a negative prediction") is gated on having evidence to predict against. So the *entire judge machinery* — satisfied, coverage, independence, falsified_genuinely — was never exercised on the gate's decision paths in this run. The judges are correctly wired (manifest smoke-load confirms ``_LlmJudge`` instances behind each ``rca.judge.*`` service) but their production behaviour cannot be measured until the preceding step works.
- **task_meta.task_id wiring is broken in core.** ``AgentSessionConfig`` accepts ``eval_task_id`` / ``task_class`` / ``eval_run_id`` and ``SessionReadyEvent`` declares the matching fields, but the emit at ``src/agentm/core/runtime/session_factory.py`` L366 does not forward them. The observability atom therefore writes ``task_meta = {..., task_id: None}`` for every programmatically-constructed session. The rca grader keys traces by this field, so the stock grader returns ``runtime`` for every case even when ``submit_final_report`` fires. This eval works around the gap with a local scoring helper that locates the trace by case-cwd instead. Not a Phase 2 regression; a substrate-level wiring miss that should be fixed independently.
- **Phase 3 tuning candidate ranking is provisional.** With judges never having fired on a confirm/refute path, no data-driven ordering is possible from this run. If the Phase-1 manifest bug is fixed and a follow-up eval shows workers actually returning observations, the natural first candidate is ``judge.satisfied`` — it gates every ``attach_check`` and is the highest-call-rate judge by design (§6's cost analysis projected ~50–80 calls/trace, most through this judge). ``judge.coverage`` is second (it gates every ``_apply_confirm``). The independence and falsified_genuinely judges fire less often and should be tuned after the high-traffic ones are stable.
