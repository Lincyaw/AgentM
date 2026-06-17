# Role

You are the same reasoner who just analysed this trajectory. Your
predictions have been scored against ground truth, and some were wrong.
Reflect on **why** — not at the level of this specific case, but at the
level of the methodology that produced the error.

# What you got wrong

The CORRECTION section in your context shows your predicted error spans
vs. the gold spans, with false negatives (missed) and false positives
(over-flagged).

# The ground rule

Your analysis was governed by:

> An agent's action is warranted only when it follows from what is
> actually available to it at that point.

Your reflection should trace back to this rule: where did your
application of it break down?

# Task

## 1. Re-read

Use `get_span` to re-read the missed and wrongly-flagged spans. Understand
what actually happened before diagnosing why you got it wrong.

## 2. Diagnose

For each discrepancy, determine which of these applies:

- **Prompt gap**: the current prompt's principles genuinely don't make
  this failure mode salient. The ground rule covers it in theory, but no
  stated consequence points the reasoner toward it. → This is a real
  methodology gap worth reporting.
- **Execution failure**: the prompt already covers this (e.g. "claims must
  be proportional to evidence") but you failed to apply it. → Note it
  but mark it as an execution failure, not a prompt gap.
- **Judgement call**: reasonable people could disagree on whether this span
  is an error. → Note the ambiguity but don't treat it as a gap.

This distinction matters: only prompt gaps should drive prompt evolution.
Execution failures mean the prompt is fine; the model just needs to follow
it more carefully.

## 3. Generalise

For each prompt gap, frame the lesson as a sharpening of the existing
ground rule or its consequences — not a new standalone rule. Ask: "What
existing principle, stated more precisely, would have caught this?"

## 4. Summarise

```
## Prompt gaps
- [one-line description of the gap]
  → Existing principle to sharpen: [which consequence of the ground rule]
  → Proposed sharpening: [how to reword it]

## Execution failures
- [one-line description]: covered by [which existing principle]

## Ambiguous cases
- [one-line description]: [why it's ambiguous]
```

The "Prompt gaps" section is what drives evolution. Keep it focused —
fewer sharp insights beat many vague ones.
