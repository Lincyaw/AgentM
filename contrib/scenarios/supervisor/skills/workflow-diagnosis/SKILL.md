---
name: workflow-diagnosis
description: >
  How to diagnose workflow failures by reading structured results and
  traces. Load this skill when a workflow run fails or produces
  low-quality output and you need to identify the root cause.
---

# Workflow Diagnosis

## Reading the result

A workflow returns a structured JSON object. Start here — it tells you
what happened without needing the full trace.

Key signals in a devloop-style result:
- `success: false` + `test_result.failures` — read the failure messages.
  Are they test bugs (wrong assertions) or implementation bugs?
- `code_review.approved: false` — read `findings`. Are they spec gaps
  or implementation mistakes?
- `rounds == max_rounds` with failures remaining — the coder is stuck.
  Check if the failures are the same across rounds (fundamental
  misunderstanding) or changing (thrashing).

## When to read the trace

The structured result is usually enough to decide what to fix. Read the
trace only when:
- The failure is ambiguous (test passes but output is wrong)
- The agent seemed to ignore its instructions (check if context was
  actually injected)
- You need to understand what the agent tried and why it failed

## Trace analysis pattern

Don't read full traces yourself — they're large. Spawn a sub-agent with
a focused question:

- "Read the trace at {path} and tell me: did the coder agent receive
  the spec in its context? What was its first implementation approach?"
- "Read the test-writing agent's trace. Are the tests actually testing
  the acceptance criteria, or are they testing implementation details?"

## Common failure patterns

**Spec too vague** — the coder has freedom to interpret, and interprets
wrong. Fix: improve the spec prompt or add a skill about the domain.

**Tests test implementation, not behavior** — tests break on valid
alternative implementations. Fix: add a skill about writing
behavior-focused tests.

**Coder ignores context** — the skill or spec was injected but the
coder didn't follow it. This is usually a model capability issue.
Fix: simplify the instruction, or switch to a more capable model.

**Same failure across rounds** — the coder doesn't understand the
problem. The fix prompt isn't giving enough signal. Fix: add a skill
with a worked pattern for this type of problem.

**Thrashing across rounds** — the coder fixes one test but breaks
another. The implementation approach is fundamentally wrong. Fix: the
spec needs to be more prescriptive about the approach, or add a skill.

## Deciding what to fix

Most fixes should be skills. The hierarchy:
1. Missing domain knowledge → write a skill
2. Bad prompt wording → edit the prompt (rare — prompts should be
   principle-based)
3. Wrong workflow structure → edit the workflow (very rare)
4. Model limitation → switch model or simplify the task
