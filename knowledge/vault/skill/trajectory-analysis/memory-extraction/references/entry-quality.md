---
type: reference
tags: [trajectory-analysis, memory-extraction]
---

# Entry Quality

A good entry teaches a reasoning APPROACH, not a specific RULE.

## Good Examples

### Causal relationship with diagnostic move

"When a service shows high CPU but low memory usage, these metrics may be connected
through garbage collection. Frequent GC reclaims memory (keeping usage low) while consuming
CPU cycles. The diagnostic move is to check GC activity before concluding on either metric
independently -- they may be two faces of the same problem."

### Explaining the WHY behind a pattern

"A service can appear healthy in average latency while being the root cause.
Resource exhaustion problems (GC pauses, page fault stalls, connection pool waits) cause
intermittent long pauses -- a few requests take 10x longer while most are fast. The average
barely moves, but the tail (p99) explodes. If only average latency is checked, the sick
service looks healthy and the investigation chases its upstream victims instead."

### Reasoning pattern with blind spot identification

"When traces show unexplained latency (no slow child spans), the instinct is to blame
'internal processing.' But absence of child spans means traces cannot explain the latency --
it's a blind spot, not an answer. The diagnostic move is to switch signals: check resource
metrics and logs for the service. The explanation often lives outside the trace data."

## Bad Examples

### Brittle rule without reasoning

"If CPU > 3x normal and memory.used < normal, it's GC thrashing. Check jvm.gc.duration."

Why bad: A weaker model will apply this mechanically without understanding why,
and miss cases that don't exactly match the pattern.

### True but unexplained

"Always check p99 latency, not just average."

Why bad: True but doesn't teach WHY, so the agent doesn't know when this matters
more or less.

### Rule without reasoning (variant)

"If no slow child spans, check metrics instead of blaming internal processing."

Why bad: A weaker model will follow this mechanically without understanding why,
and fail when the situation is slightly different.

### Case fact (not transferable)

"ts-ui-dashboard had no internal child spans for the login endpoint."

Why bad: Not transferable -- only useful for this specific case.

## Quick Self-Check

Before finalizing an entry, ask:
1. Does it explain **why** a pattern occurs, not just **what** to do?
2. Would it apply to services and metrics not mentioned in the source trajectory?
3. Does it teach reasoning that adapts to variations, or a rigid rule that breaks?
