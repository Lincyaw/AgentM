---
name: reproduce-bug
description: >
  Step-by-step procedure to reproduce a bug. Use this FIRST before trying to
  fix anything. Produces a minimal script that demonstrates the failure.
---

# Reproduce Bug

You are reproducing a bug. Follow these steps exactly in order. Do not skip
steps or read additional files beyond what each step specifies.

## Step 1: Identify the test framework

```bash
ls package.json 2>/dev/null && cat package.json | grep -E "vitest|jest|mocha|tap" | head -5
ls *.toml 2>/dev/null | head -3
```

Note which test runner is available (vitest, jest, etc.).

## Step 2: Find existing related tests

```bash
find . -name "*.test.*" -o -name "*.spec.*" | grep -i "<keyword from bug description>" | head -10
```

If you find relevant test files, read them to understand how the codebase
tests similar functionality.

## Step 3: Write a minimal reproduction script

Create a file called `repro.ts` (or `repro.js` / `repro.py` depending on
the project language) that:

1. Imports the minimum necessary from the project
2. Sets up the simplest possible environment (in-memory DB, mock server, etc.)
3. Calls the function/endpoint described in the bug
4. Prints ACTUAL output clearly
5. Prints EXPECTED output clearly
6. Prints PASS or FAIL

Keep it under 50 lines. Use the `write` tool to create the file.

## Step 4: Run the reproduction

```bash
npx tsx repro.ts 2>&1
```

Or the appropriate runner for the language. Report:
- Exit code
- ACTUAL vs EXPECTED output
- Whether the bug is confirmed

## Step 5: Report result

State clearly:
- "BUG CONFIRMED: [one-line description of what goes wrong]"
- OR "BUG NOT REPRODUCED: [what happened instead]"

If not reproduced, adjust the repro script (maybe the setup is wrong) and
try ONE more time. If still not reproduced after 2 attempts, report that.

## Constraints

- Do NOT read source code in this skill. You are only writing and running
  a reproduction script.
- Do NOT try to fix the bug. Only reproduce it.
- Maximum 8 tool calls for this entire skill.
