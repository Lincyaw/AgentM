# Policy-Guided Context Recovery

Status: design note and experiment protocol

This note records a proposed closed-loop response to structurally detected
agent investigation churn. The motivating case is Harbor session
`e3a4838f995540e6`, which completed normally but solved an adjacent OAuth
header/status problem instead of the requested generic request-context response
problem.

## Motivation

The policy engine detected investigation churn well before the first production
edit:

| Turn | Observation |
| --- | --- |
| 76 | `repeated-region-reading` fired |
| 78, 79, 80, 89 | `unchanged-anchor-reentry` fired |
| 93 | `policy_stats` reported 7 rules, all in observe mode |
| 159 | First production edit |
| 207 | Token-budget compaction occurred, after the wrong implementation and tests |

At turn 84 the agent had already established several useful facts: the ordinary
`auth.handler(Request)` path worked, the leaked wrapper was associated with an
`asResponse=false` path, and a framework adapter might call an endpoint without
setting `asResponse`. It did not turn those facts into a discriminating probe.
Instead, it continued expanding the investigation and eventually substituted a
separate `ctx.json` header/status defect for the reported problem.

This is a control failure rather than a detection failure. Observe-only IFG
signals recorded the churn, but they did not cause the agent to re-ground its
task model. The eventual generic compaction happened too late and was designed
to preserve current decisions, not audit their causal support.

## Objective

When structural investigation evidence shows that an agent is repeatedly
re-entering unchanged anchors, cause the agent to:

1. recover the original task invariants;
2. separate observations from interpretations;
3. identify evidence that contradicts the current hypothesis;
4. choose the smallest experiment that distinguishes surviving hypotheses; and
5. delay production edits until the reported primary symptom is reproduced.

The policy remains a structural observer. It must not infer the correct code
location or inject a task-specific solution.

## Recovery ladder

Use an escalation ladder instead of compacting on the first weak signal.

### Stage 1: causal-review injection

On the first adaptive `investigation-anchor-churn` transition, inject a concise
recovery contract at the next model boundary:

```text
Pause the current investigation before using another tool.

1. Restate the original task as changed variables and invariants.
2. Separate confirmed observations from explanatory hypotheses.
3. Name evidence that contradicts or fails to support the current hypothesis.
4. Identify the smallest untested case that distinguishes the surviving
   hypotheses.
5. Run one discriminating reproduction that changes only that variable.
6. Do not edit production code until the primary reported symptom is reproduced.
```

The diagnostic must include the structural IFG evidence that triggered it. A
generic message such as "you are repeating reads" is too easy to dismiss as
unrelated to the task.

### Stage 2: recovery compaction

If churn recurs in the same repository generation after the Stage 1 injection,
request one recovery compaction. This compaction is a semantic checkpoint, not
only a token-budget operation.

The recovery summary must contain these sections:

```text
Original objective and invariants
Confirmed observations
Disconfirmed hypotheses
Current hypothesis and missing causal links
Contradictions in the current approach
Untested discriminating cases
Smallest next probe
Conditions required before editing
```

The summarizer must treat prior decisions as hypotheses rather than facts. It
must not invent a solution or use hidden verifier information.

After compaction, inject the same causal-review contract and require the next
meaningful action to be a discriminating probe. Limit recovery compaction to
once per repository generation and apply a cooldown so repeated structural
events cannot create a compaction storm.

### Stage 3: persistent non-convergence

If investigation churn persists after recovery compaction, escalate with the
specific unmet condition: no new discriminating evidence, no reproduced primary
symptom, or continued re-entry into an already rejected path. Do not resend the
same generic warning. A future enforcement policy may gate production edits on
a task-derived failing reproduction, but that requires stronger evidence than
the current structural IFG provides.

## Mechanism and policy boundary

`compact` is a context mechanism; `recover` is the policy intent. The policy DSL
should therefore request recovery rather than directly manipulate trajectory
state:

```text
IFG policy
  -> recovery request
     -> context recovery service
        -> optional forced compaction
        -> causal-audit summary prompt
        -> post-compaction recovery injection
```

Today the policy effects are limited to `notify`, `block`, `escalate`, and
`abort`, while LLM compaction is triggered only by `max_messages` or
`reserve_tokens`. Implementing Stage 2 requires a typed, session-scoped recovery
request consumed by the compaction/context layer. This preserves the repository
rule that the SDK supplies mechanism and atoms supply policy.

## Fork experiment: `e3a4838f995540e6`

The first experiment isolates the causal-review message from compaction. The
completed original suffix acts as the control, so only one intervention fork is
required.

### Anchor

Fork the session after committed turn 89. This is the last observed
`unchanged-anchor-reentry` before the agent manually inspected policy state, and
the trajectory still contains the useful turn-84 observation about
`asResponse=false`. No production edit has occurred.

Optional follow-up anchors:

- turn 76 tests intervention at the earliest structural signal;
- turn 93 tests intervention immediately after the agent has seen
  `policy_stats` and dismissed it as unrelated.

### Intervention message

Send the following message without mentioning `auth.api.ok`,
`to-auth-endpoints.ts`, the oracle patch, or verifier results:

```text
Pause before changing code. Your current investigation has revisited unchanged
areas several times, and some observations may already contradict the path you
are following.

Re-read the original task and write a short checkpoint containing:
- the one condition that changes between the working and failing cases;
- confirmed facts versus hypotheses;
- which hypotheses the current reproductions have ruled out;
- the smallest still-untested reproduction that changes only the task's stated
  condition.

Then run that reproduction. Do not edit production code until you can reproduce
the reported response-shape or header behavior through that minimal contrast.
```

This prompt supplies a reasoning procedure, not the expected answer.

### Success criteria

The intervention succeeds if the fork, without ground-truth hints:

1. recognizes that `auth.handler(Request)` is not a discriminating probe because
   that path already returns a `Response`;
2. compares a direct endpoint/API call with and without a `Request` while holding
   the endpoint constant;
3. reproduces the response-materialization difference before editing production
   code;
4. redirects investigation toward the generic endpoint dispatcher rather than
   OAuth-specific endpoint files; and
5. explains how the same generic mechanism accounts for both response shape and
   response headers such as `Set-Cookie`.

Classify the result as:

- **self-corrected**: derives the discriminating request-context probe and causal
  mechanism;
- **partially corrected**: abandons the old path but still needs a task-specific
  hint;
- **not corrected**: continues endpoint-specific searching or edits without a
  primary-symptom reproduction.

### Measurements

Record:

- turns and tool calls from the fork anchor to the first discriminating probe;
- whether the probe changes only request presence;
- turns to first production edit;
- production files touched before validation;
- verifier outcome;
- input/output tokens; and
- whether the intervention itself was repeated or dismissed.

For the later Stage 2 experiment, fork from the same anchor and add recovery
compaction. Keeping the anchor and intervention contract constant isolates the
incremental value of compaction from the value of the causal-review message.

## Evaluation across sessions

Do not graduate automatic recovery from observe mode based on this case alone.
Replay it against both failed and successful Harbor sessions and measure:

- detection before the first wrong-scope edit;
- probability of producing a new discriminating probe within the next ten tool
  calls;
- task success rate;
- false-positive recovery rate on sessions that were converging;
- additional summarization cost; and
- total token and wall-time change.

Promote the Stage 1 injection before Stage 2 compaction. It is cheaper,
reversible, and directly tests whether a causal-review instruction is sufficient
to recover the agent's task model.
