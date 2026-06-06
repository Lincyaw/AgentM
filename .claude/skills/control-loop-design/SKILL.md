---
name: control-loop-design
description: >
  Guide for designing closed-loop control in agent-driven DevOps
  pipelines — from spec writing through parallel development, test
  verification, code review, and merge. Use when designing a new
  agent workflow, adding quality gates between pipeline stages,
  debugging an agent that fails to converge, or when stages produce
  errors that cascade downstream. Also trigger on: 控制闭环, 反馈设计,
  怎么让 agent 不跑偏, 纠偏, observer, 级联错误, 闭环设计, 质量门,
  pipeline 设计, spec 验收, 并行开发, 多次采样.
---

# Control Loop Design for Agent DevOps Pipelines

A method for designing feedback control in multi-stage agent
pipelines. The core insight: errors in early stages cascade
downstream and corrupt everything that follows. Each stage
transition needs a quality gate — observe what was actually
produced, compare to what was required, reject early if it
doesn't pass.

This document walks through a concrete pipeline — spec → design
review → parallel development → test verification → code review →
merge — and shows where control theory applies at each transition.
The principles generalize to any multi-stage agent workflow.

## The pipeline

```
Requirement
    │
    ▼
Spec Writing ──── design review gate ────┐
    │                                     │ reject: spec incomplete
    │ pass                                │
    ▼                                     │
Test Generation (from spec)               │
    │                                     │
    ▼                                     │
Parallel Development ×N                   │
    │                                     │
    ▼                                     │
Test Verification ──── test gate ─────────┤ reject: tests fail
    │                                     │
    │ pass                                │
    ▼                                     │
Code Review ──── review gate ─────────────┘ reject: doesn't fulfill spec
    │
    │ pass
    ▼
Merge to main
```

Every arrow pointing right is a feedback loop. Every "reject"
sends the agent back with a specific delta — not "try again" but
"these specific conditions are unmet."

## Stage 1: Spec Writing

The spec is the objective function for everything downstream.
A vague spec produces vague code and untestable results. The
spec must be detailed enough that:

1. **Tests can be derived from it.** If the spec says "add user
   authentication," you can't write a test. If it says "POST /login
   with valid credentials returns 200 + JWT; invalid credentials
   returns 401; missing fields returns 422," you can.

2. **Interfaces are defined.** Function signatures, API endpoints,
   data structures, error types. The dev agent shouldn't be inventing
   interfaces — it should be implementing specified ones.

3. **Acceptance criteria are verifiable.** Each criterion maps to
   a machine-checkable condition: test name, CLI exit code, file
   existence, API response shape.

A good spec looks like:

```
## Feature: Rate Limiting

### Interface
- RateLimiter(max_requests: int, window_seconds: int)
- RateLimiter.check(client_id: str) -> bool
- RateLimiter.reset(client_id: str) -> None

### Acceptance Criteria
AC-1: check() returns True for the first max_requests calls
      within window_seconds for the same client_id
AC-2: check() returns False for the (max_requests + 1)th call
AC-3: After window_seconds elapse, check() returns True again
AC-4: reset() immediately restores the full quota
AC-5: Different client_ids have independent counters
```

Each AC becomes a test case. Each test case becomes a gate
condition. This is where the objective function lives — not in
a generic Protocol, but in the spec itself.

### Design review gate

Before any code is written, review the spec:

```
pre-conditions:
  spec_exists: true
  has_interface_definitions: true
  has_acceptance_criteria: true
  each_ac_is_testable: true    # no "should be fast" without a threshold

post-conditions:
  reviewer_approved: true
```

If the spec fails review, send it back with specific feedback:
"AC-3 is not testable — 'after window_seconds elapse' needs a
concrete test mechanism (mock clock? sleep?). AC-5 doesn't specify
thread safety — is concurrent access in scope?"

This is the highest-leverage gate in the pipeline. A rejected spec
costs one LLM call. A bad spec that passes costs N parallel dev
sessions + test runs + review cycles, all producing the wrong thing.

## Stage 2: Test Generation

Tests are written from the spec, not from the code. This is
critical — tests derived from code test what was built, not what
was required. Tests derived from the spec test what should have
been built.

```
input:  spec (interface definitions + acceptance criteria)
output: test file(s) that exercise each AC

post-conditions:
  one_test_per_ac: true         # every AC has a corresponding test
  tests_compile: true           # tests are syntactically valid
  tests_fail_without_impl: true # tests aren't trivially passing
```

The last condition matters: if the tests pass without any
implementation, they're not testing anything. Run the tests against
a stub/empty implementation and verify they fail. This is the
"test the test" step.

The test file is the ground-truth verifier for all subsequent
stages. It doesn't change when the code changes — it changes only
when the spec changes.

## Stage 3: Parallel Development

Spawn N independent dev sessions working on the same spec. Each
session:
- Gets the spec + test file as input
- Works on its own branch
- Has no knowledge of the other sessions

Why parallel? LLM output is stochastic. One session might take
a clean approach; another might get stuck in a loop. Running N
sessions and selecting the best result is cheaper than running
one session and hoping it produces good code.

Each session's turn-level control loop handles behavioral issues:

```
turn-level observer (inside each session):
  if consecutive_identical_tool_errors >= 3:
    inject("You've hit the same error 3 times. Try a different
           approach — the current one isn't working.")
  if turns_since_last_file_write >= 10:
    inject("You haven't written any code in 10 turns. Focus on
           producing the implementation.")
```

This is Tier 2 observation (behavioral patterns). It catches
"agent is stuck" without knowing whether the agent's approach is
correct — that's the test gate's job.

## Stage 4: Test Verification

Run the pre-written tests against each dev session's output.
This is the primary quality gate — pure Tier 1 observation.

```
for each session's branch:
  git checkout branch
  result = pytest test_rate_limiter.py
  record:
    tests_passed: int
    tests_failed: int
    failed_test_names: list[str]
    lint_clean: bool
    type_check_clean: bool
```

### Decision logic

```
if all tests pass AND lint clean AND types clean:
  → advance to code review
if some tests pass:
  → resume same session with delta:
    "3 of 5 tests pass. Failing:
     test_rate_limit_exceeded: AssertionError at line 42
     test_window_expiry: TimeoutError — mock clock not advancing
     Fix these specific failures."
if zero tests pass:
  → check if tests even ran (import errors? missing deps?)
  → if tests can't run: resume with "tests fail to import: <error>"
  → if tests run but all fail: consider discarding this session
```

### Per-test tracking across rounds

Don't just count "3 of 5 pass." Track which specific tests pass
and fail each round:

```
Round 1: AC-1 ✓  AC-2 ✓  AC-3 ✗  AC-4 ✗  AC-5 ✗
Round 2: AC-1 ✓  AC-2 ✓  AC-3 ✓  AC-4 ✗  AC-5 ✗  (converging)
Round 3: AC-1 ✗  AC-2 ✓  AC-3 ✓  AC-4 ✗  AC-5 ✗  (regression on AC-1!)
```

Round 3 shows a regression — AC-1 was passing, now it's not. The
feedback must say "AC-1 regressed" not just "2 tests fail." The
agent needs to know it broke something that was working.

### Convergence and escalation

Track delta magnitude per round:

- Converging (more tests pass each round) → continue
- Stalled (same tests fail for 2+ rounds) → escalate: change the
  feedback from "test X fails" to "test X has failed for 3 rounds.
  The error is always <same error>. Your current approach to this
  test isn't working — reconsider the algorithm, not just the
  implementation details."
- Diverging (previously-passing tests now fail) → stop this session.
  The agent is making things worse. Prefer other parallel sessions.

### Session selection

When multiple parallel sessions reach the test gate:

```
rank sessions by:
  1. number of tests passing (primary)
  2. lint cleanliness (secondary)
  3. code simplicity / LOC (tiebreaker)

select: top session(s) that pass all tests
if none pass all tests:
  select: session with most tests passing, resume it
if multiple pass all tests:
  advance all to code review, let reviewer pick
```

## Stage 5: Code Review

A review agent verifies two things:
1. Does the code actually fulfill the spec? (not just "do tests pass"
   — tests can have gaps)
2. Is the code quality acceptable? (architecture, readability,
   maintainability)

```
input:  spec + code diff + test results
output: per-AC verdict (pass/fail/cannot-judge) + code quality findings

post-conditions:
  all_ac_pass: true           # every acceptance criterion is met
  no_critical_findings: true  # no security/correctness issues
  review_comment_posted: true # structured verdict on the issue
```

### Structured verdict format

The reviewer must produce a per-AC verdict, not a free-text opinion:

```
AC-1: PASS — test_first_n_requests covers this, implementation
      uses a counter dict keyed by client_id
AC-2: PASS — test_rate_limit_exceeded verifies the (n+1)th call
AC-3: FAIL — test_window_expiry passes but the implementation
      uses time.time() instead of the injected clock; this will
      be flaky in CI
AC-4: PASS — test_reset verifies quota restoration
AC-5: CANNOT JUDGE — no concurrency test exists; spec doesn't
      specify thread safety but the implementation uses a plain
      dict (not thread-safe)
```

FAIL and CANNOT JUDGE findings go back to the dev agent as
specific, actionable feedback. The dev agent gets the verdict
and the failing AC description, not a vague "needs work."

### Parallel session selection at review

When the reviewer sees multiple passing implementations:

```
compare on:
  - correctness (AC verdicts — any FAIL disqualifies)
  - code quality (simpler > complex, idiomatic > clever)
  - test coverage beyond spec (bonus, not required)
  - maintainability (clear naming, reasonable structure)

select: the implementation that is correct AND simplest
```

The reviewer picks, not the orchestrator. The orchestrator only
checks "did the reviewer produce a verdict?" The quality judgment
is the reviewer's job.

## Stage 6: Merge

The final gate. Conditions:

```
pre-conditions:
  all_tests_pass: true
  reviewer_approved: true
  no_merge_conflicts: true
  ci_green: true (if CI exists)

action:
  squash merge to main
  close the issue
  delete the branch
```

This is the simplest gate — entirely mechanical. If pre-conditions
are met, merge. If not, identify which condition fails and route
feedback to the appropriate stage (test failure → dev agent, review
rejection → dev agent, merge conflict → dev agent with conflict
details).

## Control theory summary

Where each control concept lives in this pipeline:

| Concept | Where | Concrete form |
|---|---|---|
| **Objective** | The spec | AC-1 through AC-N, interface definitions |
| **Observation** | Test results | pytest exit code, per-test pass/fail |
| **Feedback** | Delta message | "AC-3 fails: expected X, got Y" |
| **Controller** | Orchestrator | policy table: pass → advance, fail → resume/block |
| **Stability** | Budget caps | max rounds per stage, max parallel sessions |
| **Ground truth** | Tests from spec | tests written before code, not from code |
| **Convergence** | Per-test tracking | which specific tests improve/regress each round |
| **Gain control** | Escalation ladder | factual delta → strategy hint → scope narrowing → block |
| **Multi-sampling** | Parallel dev | N sessions, reviewer selects best |

## Closure checklist

Before deploying a pipeline, verify:

```
[ ] Spec has testable acceptance criteria (not just prose goals)
[ ] Tests are derived from spec, not from code
[ ] Each stage transition has explicit pre/post conditions
[ ] Feedback names specific failing conditions, not "try again"
[ ] Individual findings are tracked across rounds (not just counts)
[ ] Convergence is monitored: stall/diverge triggers escalation,
    not just more of the same feedback
[ ] Feedback escalates on repeated failure (factual → strategy →
    scope narrowing → block)
[ ] At least one gate uses external ground truth (test results),
    not just LLM judgment
[ ] Parallel sessions are ranked by objective criteria, not by
    which finished first
[ ] Feedback is concise and deduped — not accumulated indefinitely
    in the agent's context
```

## Anti-patterns

**Vague spec, precise code review.** Reviewing code against a vague
spec produces subjective, inconsistent verdicts. The reviewer argues
about design choices that were never specified. Fix the spec, not
the review process.

**Tests from code.** Writing tests after seeing the implementation
tests what was built, not what was required. The tests become a
mirror of the code's bugs. Write tests from the spec before any
dev session starts.

**Sequential-only development.** Running one dev session, waiting
for it to fail, retrying — wastes wall-clock time and gets the same
stochastic output distribution. Parallel sessions with selection
are strictly better when budget allows.

**Aggregate feedback.** "2 of 5 tests fail" is not actionable. The
agent doesn't know which 2, what the errors are, or whether they're
the same 2 as last round. Always name the specific failing tests
and their error messages.

**Same feedback on repeat.** If the agent failed to fix test_login
after seeing "test_login: AssertionError on line 42" twice, sending
the same message a third time won't help. Escalate: explain why the
current approach isn't working, suggest a different angle, or narrow
scope to just that one test.

**LLM reviewer with no test anchor.** A reviewer that only reads
code and spec (no test results) can be fooled by plausible-looking
code that doesn't actually work. Always give the reviewer the test
results alongside the code. Tests are the ground truth; the review
is the judgment layer on top.
