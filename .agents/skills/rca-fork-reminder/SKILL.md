---
name: rca-fork-reminder
description: >
  Use when an RCA scenario agent failed or is suspected to have failed, and
  Codex needs to run AgentM fork experiments with targeted reminders to help the
  agent notice reasoning gaps, missed evidence, wrong assumptions, premature
  narrowing, root/effect confusion, or output synthesis errors. This skill is
  for research on when to inject hints and what to hint: compare multiple fork
  points and reminder strategies, using only information available to the forked
  agent at that point, then evaluate whether later forks recover the correct RCA
  answer.
---

# RCA Fork Reminder

Use this skill to study and improve a failed RCA trajectory by forking the
session at selected points and injecting evidence-bounded reminders. The goal is
not to hand the answer to the agent. The goal is to test whether a reminder about
its own visible omissions or weak reasoning helps it recover.

## Non-negotiable constraints

1. Use `agentm trace` for trajectory inspection. Do not hand-parse raw OTLP
   JSONL or artifacts.
2. Run forked RCA agents in a scratch cwd such as `/tmp/<case>-fork-<label>`,
   not in the main repo.
3. A reminder may cite only facts the source agent could already access before
   the fork point:
   - tool outputs already shown before that turn,
   - user/scenario instructions already in the conversation,
   - general RCA reasoning principles,
   - schema/tool affordances visible to the agent.
4. Never leak ground truth, judge output, injection metadata, or conclusions from
   a later fork into an earlier fork reminder.
5. Use ground truth or judge output only after a fork finishes, to score the
   result and choose the next experiment.

## Inputs

Collect what is available:

- failed or baseline session id,
- RCA case data directory, if needed for reruns via `AGENTM_RCA_DATA_DIR`,
- optional judge/result file for post-run scoring,
- user budget or maximum fork count, if provided.

If the case has no existing failed session, first run a baseline RCA scenario and
record the session id. If the user already provided a failed session, start from
that.

## Inspect the failed trajectory

Start with cheap trace commands:

```bash
uv run agentm trace turns --session <sid> --format ndjson
uv run agentm trace tools --session <sid> --tool submit_final_report --format ndjson
uv run agentm trace usage --session <sid> --format ndjson
```

Use `messages` only when you need the chronological context around a candidate
fork point:

```bash
uv run agentm trace messages --session <sid> --format text
```

Identify:

- the final submitted root services and fault kinds,
- the first point where the agent narrows to a wrong hypothesis,
- evidence it observed but did not reconcile,
- candidates it discovered but demoted without enough justification,
- assumptions such as single-root bias, "no error logs means healthy", or
  treating missing spans as absence of failure,
- output synthesis problems such as malformed graph ids or empty root causes.

## Choose fork points

Prefer testing several positions. Good positions are:

1. **Early divergence**: before the first confident wrong narrowing. Use this to
   redirect investigation breadth.
2. **Evidence discovery**: after the agent has seen useful anomaly data but
   before it ignores or misclassifies it. Use this to force reconciliation.
3. **Pre-submit synthesis**: immediately before `submit_final_report`. Use this
   to audit root/effect classification, multi-root recall, and graph structure.
4. **Repair after validation failure**: after a failed final-report tool call.
   Use this only to remind it to fix structure, not to change evidence claims.

If an early fork partially improves the answer, fork from that improved session
at a later point. Multi-generation fork trees are expected.

## Reminder design

Write reminders as short `<system_reminder>...</system_reminder>` blocks. A good
reminder points at a reasoning operation the agent should perform, not at a
hidden answer.

### Strategy A: Reconcile observed anomalies

Use when the agent observed anomalies but is about to choose a different root.

```text
<system_reminder>
Before narrowing to one root, reconcile the anomalies you already observed:
<fact 1 from prior tool output>, <fact 2 from prior tool output>. Missing or
normal-only spans are evidence that must be explained, not proof of health.
Run targeted comparisons on those candidates and their callers before finalizing.
</system_reminder>
```

### Strategy B: Challenge a weak assumption

Use when the agent relies on a brittle shortcut.

```text
<system_reminder>
Do not assume <assumption> from the current evidence. Check whether an alternate
explanation fits the observations: <visible observation>. If you keep your
current hypothesis, explicitly explain why it accounts for that observation.
</system_reminder>
```

Common RCA assumptions to challenge:

- only one root cause exists,
- the user-facing slow service must be the root,
- zero abnormal traces means the service is healthy,
- sparse ERROR logs rule out a local fault,
- latency increase is stronger evidence than complete path disappearance,
- resource metrics are irrelevant when traces are missing.

### Strategy C: Root/effect audit

Use near final submission when the agent found real signals but may demote a
root to an effect.

```text
<system_reminder>
Before submitting, audit root/effect classification. For each candidate, decide
whether it has direct local evidence or is only downstream propagation. If you
demote a candidate, explain which upstream change fully accounts for both its
trace/log behavior and its service-local metrics. The RCA task may have multiple
independent roots.
</system_reminder>
```

### Strategy D: Missing-check prompt

Use when the trajectory shows it never ran a necessary comparison.

```text
<system_reminder>
Before finalizing, run one targeted check that is still missing: compare
<candidate services/endpoints/edges/metrics> across normal and abnormal windows.
Use the result to decide whether the current hypothesis is local root cause or
downstream effect.
</system_reminder>
```

Do not name a candidate that the source agent had no basis to investigate unless
the reminder is phrased generically, such as "return to the top candidates from
your broad diff".

### Strategy E: Output repair

Use only after the agent has already attempted `submit_final_report` and the
tool rejected the structure.

```text
<system_reminder>
Your previous final-report tool call failed schema validation. Keep the evidence
claims you can support, but repair the graph structure: every edge endpoint and
every root_cause id must reference an existing node id.
</system_reminder>
```

## Run a fork

Use the same RCA data directory as the source run when applicable:

```bash
AGENTM_RCA_DATA_DIR=<case-data-dir> \
AGENTM_DUCKDB_THREADS=1 \
uv run agentm fork <source-session-id> \
  --cwd /tmp/<case-id>-fork-<turn-or-strategy> \
  --turn-index <n> \
  --prompt '<system_reminder>...</system_reminder>'
```

Monitor until the command exits. Record the new session id from stdout. Do not
leave required agent processes running.

## Score and iterate

After each fork:

```bash
uv run agentm trace tools --session <new-sid> --tool submit_final_report --format ndjson
uv run agentm trace usage --session <new-sid> --format ndjson
uv run agentm trace turns --session <new-sid> --format ndjson
```

Classify the result:

- **worse**: more hallucination, less evidence, or no valid final report,
- **same**: same wrong root/fault type,
- **partial**: one root fixed, fewer hallucinations, better evidence, but still
  not correct,
- **success**: correct enough for the target metric or judge.

If partial, inspect the improved fork and choose a later fork point where the
remaining error becomes visible. Example progression:

1. early reminder moves the agent from a hallucinated user-facing root to one
   real root,
2. mid-run reminder prevents it from ignoring a second candidate,
3. pre-submit reminder fixes root/effect classification and multi-root recall,
4. repair reminder fixes a malformed final graph.

Stop when the fork succeeds, the user-provided budget is exhausted, or additional
forks repeat the same failure without producing new evidence.

## Experiment log

Report results as a compact table:

| Source | Fork turn | Reminder strategy | Evidence basis | New session | Outcome |
|---|---:|---|---|---|---|

Include token usage for successful or notable forks. In the summary, explain
which reminder timing and content helped, which did not, and what this suggests
about the RCA agent's failure mode.

## Guardrails for research integrity

- Keep the reminder text in the final report so the experiment is reproducible.
- Separate "information available at fork time" from "post-run scoring
  information".
- If a reminder relies on a tool output, name the observed signal, not the hidden
  true cause.
- Prefer minimal hints first. Escalate from generic reasoning reminders to more
  pointed evidence reminders only after a fork fails or partially succeeds.
- A successful fork proves the agent could use that reminder, not that the
  reminder is generally valid. Re-test on more cases before generalizing.
