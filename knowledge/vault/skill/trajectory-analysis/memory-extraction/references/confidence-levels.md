---
type: reference
tags: [trajectory-analysis, memory-extraction]
---

# Confidence Levels

The vault accepts three confidence levels. Use these exact values in frontmatter.

## fact

Observed in 3+ trajectories with clear causal evidence. The pattern is well-established
and can be relied upon.

## pattern

Observed in 2+ trajectories, or in 1 trajectory with very strong evidence (clear causal
chain, unambiguous signal pattern).

## heuristic

Observed in 1 trajectory with plausible generalization. The pattern seems transferable
but has limited supporting evidence.

## Writing Threshold

Only write standalone entries with confidence >= heuristic.

Very speculative observations should not be standalone entries. Instead, note them
in existing related entries as "possible related pattern" to be confirmed by
future trajectories.
