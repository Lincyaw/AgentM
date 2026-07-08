# AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems

arXiv 2605.08715 (Rutgers/UT Austin/Purdue, NeurIPS 2026 preprint).
Source: `~/.cache/nanochat/knowledge/2605.08715/`.

## What the paper does

Reframes agent failure attribution from **post-hoc** (diagnose the failed
trajectory after it ends — Who&When, AgenTracer, AgentDebug) to **online
auditing**: at every step k, an auditor sees only the prefix τ₀:ₖ and must
emit CONTINUE or ALARM(step k̂, agent â) — no access to future steps or the
outcome. An alarm at the *decisive error* (earliest step whose correction
would flip the outcome, Def. 2.1) opens an intervention window before
propagation locks in failure.

Two artifacts:

1. **AFTraj-2K** (~2.3K trajectories; Math/Coding/Agentic on
   AutoGen/MetaGPT/Smolagents). Safe runs pass three filters (outcome
   equivalence, tool-call integrity, per-turn coherence) — success alone is
   NOT safety, silent errors get masked by downstream recovery. Unsafe runs
   from two streams: *constructive* (inject a faulty turn into a safe run,
   re-roll, keep only if outcome flips → ground-truth k* by construction)
   and *diagnostic* (natural failures, propose-and-verify multi-judge
   ensemble with 4 binary criteria: exists/substantive/decisive/earliest).

2. **AgentForesight-7B** (Qwen2.5-7B-Instruct), coarse-to-fine RL:
   - Stage 1 **BPPO**: DPO variant on *boundary pairs* — prefix ending one
     turn before k* (optimal verdict CONTINUE) vs prefix including k*
     (optimal ALARM). Same trajectory, one-turn difference, flipped label →
     the failure boundary is the only learnable signal.
   - Stage 2 **GRPO** with a three-axis reward: format gate (*what*),
     Gaussian step-proximity (*where*), agent attribution (*who*);
     class-symmetric ±1 to prevent SAFE-collapse; KL anchored to the
     Stage-1 checkpoint.

## Key results

- Exact-F1 66.4 vs best proprietary judge (DeepSeek-V4-Pro) 46.6 (+19.9);
  step-localization error (ASS) 3× lower (0.6 vs 1.77).
- **The headline number for us: False Alarm Rate.** Generic LLM judges are
  terrible online auditors — DeepSeek-V4-Pro FAR = 43.2%, GPT-4.1 and
  open-source 7-8B models worse, 3-4B collapse to near-universal false
  alarms. Only the trained 7B sits in the "deployable region"
  (FAR ≤ 20% ∧ Step-Acc ≥ 50%), at FAR 2.4%.
- Ablation: Stage 2 alone wins on Math/Coding (sharply localizable errors)
  but *degrades* on Agentic (fuzzy boundary); Stage 1's risk-anticipation
  prior is what suppresses false alarms.

## What it does NOT do

- **No closed-loop intervention experiment.** The eval halts the walk at
  first alarm and scores localization; nobody reruns the task after an
  alarm to measure recovered success rate. "Enabling deployment-time
  intervention" is aspirational.
- Broader-impact section explicitly warns: don't use verdicts as hard kill
  switches; expose calibrated confidence; tiered intervention or
  human-in-the-loop. Never feed auditor verdicts back as agent training
  signal (obfuscated reward hacking, citing Baker et al.).
- Appendix "external-vs-self": why an external auditor beats agent
  self-reflection — generator-verifier asymmetry, self-reflection inherits
  the generator's prior/blind spots, competes with the primary objective,
  and external is hot-swappable/tunable/inspectable.

## Relevance to llm-as-harness (AgentM)

Our 2026-07-08 trace analysis found the llmharness auditor contributed
zero outcome on LongCLI (goal-only == full): reminders were either
redundant with the goal checker (which holds the stop gate) or factually
wrong (pgtbl USYSCALL false alarm, p2 test-file flip-flop), and correct
ones were ignored when the agent could refute them from the repo (p0, 11+
CRITICALs). This paper explains the mechanism and quantifies it:

1. **Our auditor is a prompted generic judge doing online auditing — the
   exact configuration the paper measures at 27-47 F1 / 40-90% FAR.** Its
   noise is not an implementation bug; it's the documented failure mode of
   untrained auditors on prefix-restricted evidence.
2. **FAR is the metric that matters for an advisory channel.** Every false
   alarm devalues the channel (our p0 fatigue effect). A warning-only
   auditor must be tuned for precision, not recall.
3. **Decisive-error framing** is the right alarm criterion: alarm when
   correction-now would flip the outcome — not process nagging ("you
   haven't tested yet"), which our traces show the checker enforces anyway.
4. **Their data recipe is replicable on our infra**: we have thousands of
   scored LongCLI/SWE sessions in ClickHouse; constructive injection +
   re-roll is exactly what rescue-window forking already does. An
   AFTraj-style corpus (and a small trained auditor) is within reach.
5. **Their deployment advice matches our design conclusion**: auditor
   should not decide (no hard kill switch); it should emit calibrated,
   evidence-anchored warnings; enforcement stays with the goal checker.
   What the paper solves by training (grounding), we currently lack; what
   we already have (checker with gate authority), the paper lacks.
