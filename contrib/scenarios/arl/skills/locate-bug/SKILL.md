---
name: locate-bug
description: >
  Structured procedure to locate the root cause of a confirmed bug. Uses
  keyword search → focused reading → call-graph tracing. Outputs a specific
  file + line + hypothesis.
---

# Locate Bug

You are locating the root cause of a bug that has already been reproduced.
Follow these steps exactly. Do not free-explore.

## Input

Before starting, you must know:
- What the bug IS (symptom)
- What code path triggers it (from reproduction)

## Step 1: Extract search keywords (1 tool call)

From the bug description and reproduction output, identify 3-5 keywords
that are likely to appear in the buggy code. Think about:
- Error messages or wrong values from the repro
- API names, function names, variable names
- Framework concepts (middleware, handler, response, etc.)

Run ONE grep:
```bash
grep -rn "keyword1\|keyword2\|keyword3" <src-dir> --include="*.ts" -l | head -10
```

## Step 2: Pick top 3 candidate files (0 tool calls - just think)

From the grep results, pick the 3 most likely files. Prioritize:
- Files in src/ over test/ or examples/
- Files whose names relate to the bug symptom
- Files that appear for multiple keywords

## Step 3: Read candidates (max 3 tool calls)

Read each candidate file. Use `offset` and `limit` to read only the
relevant section (the part near the keyword match). Do NOT read entire
files.

For each file, note:
- What it does
- How it handles the specific operation that's buggy

## Step 4: Trace one level of call graph (max 2 tool calls)

From the most suspicious code in Step 3, look at what it calls or what
calls it. Read ONE more file that's directly imported/referenced.

## Step 5: Output hypothesis

State your finding as:
```
FILE: <path>
LINE: <approximate line number>
HYPOTHESIS: <one sentence explaining what's wrong>
EVIDENCE: <what you saw that supports this>
```

## Constraints

- Maximum 8 tool calls total for this skill.
- Do NOT read node_modules, dist, or build output.
- Do NOT run any commands except grep/find for search.
- Do NOT attempt to fix the bug. Only locate it.
- If you cannot locate the bug in 8 calls, report your best guess
  and say "LOW CONFIDENCE".
