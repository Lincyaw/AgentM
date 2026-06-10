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

## Common Pitfalls

- Forgetting to handle the empty/zero/nil case.
- Off-by-one errors in boundary checks.
- Race conditions in concurrent code — use locks or atomic operations.
- Forgetting to close resources (files, connections, locks).
