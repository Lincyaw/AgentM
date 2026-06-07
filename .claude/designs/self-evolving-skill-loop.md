# Design: Self-Evolving Skill Loop

**Status**: ACTIVE (implementation at `contrib/scenarios/rca/src/agentm_rca/evolution/`)
**Created**: 2026-06-07
**Builds on**: [per-task-evolution-loop.md](per-task-evolution-loop.md), [llmharness-cognitive-audit.md](llmharness-cognitive-audit.md)

---

## 1. Summary

Offline loop: read historical RCA trajectories + GT → identify recurring
failure patterns → distill into SKILL.md → validate on held-out cases →
keep or discard. Agent improves over time without code changes.

## 2. The Loop

```
train cases ─▶ observe ─▶ distill ─▶ backtest ─▶ select ─▶ skill库
                (GT+traj)   (LLM)     (test split)  (ΔAcc)
```

- **Observe**: GT-aware LLM analysis identifies where agent diverged.
- **Distill**: Cluster failure categories, synthesize dominant into SKILL.md.
- **Backtest**: Run with/without skill on held-out cases, compare accuracy.
- **Select**: Accept if accuracy up, reject otherwise. Retire stale skills.

## 3. Constraints

- Only output: SKILL.md. Only channel: prompt injection via skill_loader.
- Train/test: no overlap. Evaluation: direct GT comparison.
- Quality > quantity. Skills compete for prompt budget.

## 4. Orthogonality

Skill library (axis S) is independent of strategy axes (FSM, harness, etc.).
