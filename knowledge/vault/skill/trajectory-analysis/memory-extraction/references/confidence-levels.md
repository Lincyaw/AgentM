---
type: reference
tags: [trajectory-analysis, memory-extraction]
---

# Confidence Levels

## fact

Observed in 3+ trajectories with clear causal evidence. The pattern is well-established
and can be relied upon.

## high

Observed in 2+ trajectories, or in 1 trajectory with very strong evidence (clear causal
chain, unambiguous signal pattern).

## medium

Observed in 1 trajectory with plausible generalization. The pattern seems transferable
but has limited supporting evidence.

## low

Speculative inference that needs more evidence before acting on it.

## Writing Threshold

Only write standalone entries with confidence >= medium.

Low-confidence observations should not be standalone entries. Instead, note them
in existing related entries as "possible related pattern" to be confirmed by
future trajectories.
