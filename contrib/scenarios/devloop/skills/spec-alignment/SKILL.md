---
name: spec-alignment
description: >
  Ensure implementation matches the spec's interface contract. Load this
  skill when writing implementation code or fixing review findings. Covers
  contract fidelity, spec-implementation consistency, and avoiding
  test-implementation collusion.
---

# Spec-Alignment

## Contract Fidelity

- **The spec is the source of truth.** Read the interface signatures,
  return types, and exception clauses in the spec before writing code.
- Match every detail: return type (`bool` vs `None` vs raises), default
  values, parameter names, and exception types (`ValueError` vs
  `TimeoutError` vs `RuntimeError`).
- If the spec says "raises X on condition Y", the implementation MUST
  raise X — returning a sentinel value (e.g., `False`) is a spec violation.

## Test-Implementation Collusion

- **Never let the test match the implementation when the implementation
  violates the spec.** If the spec says "raises TimeoutError" but your
  implementation returns `False`, the test must test for `TimeoutError`,
  not `False`. Fix the implementation, not the test.
- After writing tests, cross-check: does every test assertion match the
  spec's contract, not just what the code currently does?

## Self-Review Before Submission

- Before declaring code complete, re-read the spec's interface section
  and verify each method's behavior against the spec line by line.
- Check: return values, exception paths, edge cases (n=0, n<0, timeout=0).
- If the spec is ambiguous, clarify it — don't guess and implement.
