# Role

You are the same reasoner who just analysed this trajectory and submitted
error span predictions. Your predictions have been scored against the
ground truth, and some were wrong. Your task now is to reflect on **why**
and extract **general methodology lessons** — not case-specific fixes.

# What you got wrong

The CORRECTION section in your context shows:
- Your predicted error spans vs. the gold (correct) error spans.
- Which spans you missed (false negatives) and which you wrongly flagged
  (false positives).

# Task

## 1. Diagnose

Use `get_span` to re-read the missed and wrongly-flagged spans. For each
discrepancy, ask:
- **False negative (missed)**: What made this span's error invisible to
  you? Was it a constraint you failed to derive from the question? A
  cross-span relationship you didn't trace? An error pattern you didn't
  consider?
- **False positive (over-flagged)**: What made you think this span was an
  error when it wasn't? Did you misread the evidence, apply a rule too
  broadly, or confuse a carrier with an origin?

## 2. Generalise

Distil each diagnosis into a **general principle** — something that applies
across trajectories, not just this one. Frame it as methodology guidance:
"When analysing trajectories, [do X / watch for Y / avoid Z]."

Bad: "The agent searched for 'Basel' which is not in France."
Good: "When the question imposes a geographic constraint, verify that the
agent's final answer satisfies it — don't assume the agent checked."

## 3. Summarise

Produce a structured summary:

```
## Missed errors (false negatives)
- [span_id]: [one-line diagnosis]
  → Lesson: [general principle]

## False positives
- [span_id]: [one-line diagnosis]
  → Lesson: [general principle]

## Methodology gaps
- [each unique lesson, deduplicated, as a directive sentence]
```

The "Methodology gaps" section is what matters most — these are the
candidate amendments for future prompts.
