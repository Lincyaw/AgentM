---
name: coding-guidelines
description: >
  General coding guidelines for the devloop coder agent. Load this skill
  when writing implementation code, tests, or fixing failures. Covers
  code quality, testing discipline, and common pitfalls.
---

# Coding Guidelines

## Implementation

- Read the spec before writing any code. Match the interface signatures exactly.
- Write the minimal code that satisfies the acceptance criteria.
- Prefer standard library over third-party dependencies unless the spec requires them.
- Keep functions short. If a function exceeds ~30 lines, split it.

## Testing

- One test function per acceptance criterion.
- Test the public API, not internal implementation details.
- Use descriptive test names: `test_<what>_<condition>_<expected>`.
- Include edge cases: empty inputs, boundary values, error paths.
- For thread safety tests, use barriers to synchronize thread starts.

## Fixing Failures

- Read the error message carefully before changing code.
- Make one focused change per fix attempt.
- Never modify test files to make tests pass — fix the implementation.
- After fixing, run a targeted smoke check, then return control to the workflow.

## Floating-Point and Time-Dependent Testing

- **Never assert exact equality on time-dependent values.** `time.monotonic()` and
  `time.sleep()` have inherent jitter; a token count that should be 0.0 may be
  `1.19e-06` due to sub-millisecond clock advance between refill and assertion.
- Use `pytest.approx()` for float comparisons: `assert bucket.tokens == pytest.approx(0.0, abs=1e-4)`.
- For "tokens should be exactly N" assertions after a consume, prefer `abs=1e-4`
  to absorb micro-refills.
- For throughput/rate tests, use a generous relative tolerance (e.g., `rel=0.15`
  for ±15%) and run for at least 5–10 seconds to average out scheduling jitter.
- When draining a bucket to 0, accept that `tokens` may be a tiny positive
  epsilon rather than exactly 0.0 — the refill clock never stops.
- Structure tests so that the implementation's refill-on-every-call pattern
  doesn't cause spurious failures: check `tokens <= expected + epsilon`, not
  `tokens == expected`.

## Common Pitfalls

- Forgetting to handle the empty/zero/nil case.
- Off-by-one errors in boundary checks.
- Race conditions in concurrent code — use locks or atomic operations.
- Forgetting to close resources (files, connections, locks).
- Using exact float equality (`== 0.0`, `== 9.0`) on values derived from
  `time.monotonic()` — always use `pytest.approx()` with an absolute tolerance.
