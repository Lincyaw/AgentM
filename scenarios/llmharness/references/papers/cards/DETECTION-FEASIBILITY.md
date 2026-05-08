# Detection feasibility: 37 AFC cards classified by what it takes to fire them

The cards in this directory describe *what can fail*. This document
classifies *how cheaply you can detect each one* — the answer drives the
diagnosis-harness layering: rule-based detectors run on every step
basically for free; LLM-as-judge detectors cost real tokens and should
only fire on candidates the cheaper layers couldn't decide.

## The four tiers

| Tier | Rough cost | What's needed | Where it works |
|---|---|---|---|
| **T1 — Pure rule** | µs / step | regex, schema check, counter, log-string match | the signal is in the trajectory format, not in the meaning |
| **T2 — Structural** | ms / step | pre-trained embedding similarity, NER + set-diff, graph metric, action-whitelist | the signal is in *form* — text similarity, dependency structure, type-checking — but no understanding-of-meaning is required |
| **T3 — Hybrid** | rule-cheap + 1 LLM call when rule fires | rule narrows to a small set of suspect steps; LLM confirms the rest | the symptom is detectable, the diagnosis isn't |
| **T4 — Pure semantic** | 1+ LLM calls per trajectory | LLM-as-judge, abductive reasoning, counterfactual replay | only an LLM that *understands* the task can decide whether the failure occurred |

T1+T2 = "rule-solvable" in the user's framing. T4 = "needs semantic". T3 is the middle ground — most useful operationally because a cheap rule layer eliminates 80%+ of negatives before paying for LLM tokens.

## Cross-tab — all 37 cards

| AFC | Name | Tier | Detection signal |
|---|---|---|---|
| 0001 | Memory over-simplification | **T4** | Need to know what detail in original observation got elided AND that the elided detail mattered for the next decision — semantic compare. |
| 0002 | False memory / fabricated recall | **T3** | Rule: extract entities the agent claims came from a tool call, check if any matching tool_call span exists. LLM confirms paraphrase vs fabrication. |
| 0003 | Memory retrieval failure | **T4** | "Relevant context exists but agent didn't fetch it" — *relevant* is task-conditional. |
| 0004 | Conversation history wipe | **T1** | Token-count / field-presence on prompt input: history is empty / truncated. |
| 0005 | Progress misassessment | **T3** | Rule: agent emits "task done" string + env state. LLM confirms whether subgoal was actually pending. |
| 0006 | Outcome misinterpretation | **T4** | Need to compare semantic content of tool result vs agent's restatement. |
| 0007 | Causal misattribution | **T4** | Need to know real root cause to know agent named the wrong one. |
| 0008 | Reflection hallucination | **T3** | Rule: NER on cited events, search upstream spans. LLM confirms whether they're truly absent vs paraphrased. |
| 0009 | Constraint ignorance | **T3** | Rule when constraints are structured (numeric budget, schema). LLM when constraints are in NL. |
| 0010 | Impossible action | **T2** | Tool returns "object not found" / "precondition failed" — string match in tool_result. |
| 0011 | Inefficient plan / wrong decomposition | **T4** | "Wasteful vs minimal expert plan" — needs understanding of the task. |
| 0012 | Backtracking absence | **T3** | Embedding similarity on consecutive revision attempts (T2-style). LLM confirms strategy is genuinely isomorphic. |
| 0013 | Planning–action disconnect | **T4** | Reasoning-vs-action and step-N-vs-step-M semantic compare. |
| 0014 | Action format error | **T1** | JSON / schema parse error from runtime. |
| 0015 | Action parameter error | **T2** | Tool returns "invalid_param" / "no such metric" — string match in tool_result. |
| 0016 | Task specification deviation | **T4** | Final answer addresses different question than task. |
| 0017 | Role specification deviation | **T2** | If roles have action-whitelists (typical in agentic systems), rule. Lookup actor → allowed actions. |
| 0018 | Goal drift | **T2** | Embedding similarity between current step and task statement, monotonic decrease over time, threshold-based. |
| 0019 | Repeated handled work | **T2** | Span-pair embedding similarity + subgoal-target identity check. |
| 0020 | Ambiguous request to peer | **T4** | Need to understand whether peer's response addressed the right question. |
| 0021 | Information withholding | **T2** | NER + set-diff: entities mentioned in A's reasoning but absent from A's outgoing message. |
| 0022 | Peer feedback ignored | **T2** | Embedding similarity between agent A's pre-feedback action and post-feedback action. |
| 0023 | Verification step skipped | **T1** | Regex over reasoning spans for `verify` / `check` / `test` / `validate`. |
| 0024 | Incorrect verification | **T4** | Need ground truth or strong oracle to know verifier passed wrong work. |
| 0025 | Premature termination | **T3** | Rule: TERMINATE/final-answer emitted. LLM confirms subgoals were still incomplete. |
| 0026 | Non-termination / endless loop | **T2** | Step-similarity matrix ridge detection + length-cap proximity. |
| 0027 | Step / round limit exhausted | **T1** | trajectory_length == cap. |
| 0028 | Wrong-location fixation | **T3** | Rule: count consecutive actions targeting same entity after a failure signal on that entity. LLM confirms the entity was actually wrong. |
| 0029 | Tool-output override | **T4** | Tool result and subsequent reasoning span have *opposite* claims — semantic compare. |
| 0030 | Critical-evidence miss | **T4** | Need to know what's *critical* in the input context. |
| 0031 | Tool execution error | **T1** | 5xx / timeout / exception in tool_call span. |
| 0032 | LLM API limit | **T1** | Provider error string `rate_limit_exceeded` / `context_length_exceeded` / `content_policy_violation`. |
| 0033 | Environment error | **T1** | Env span has crash / inconsistent-state log. |
| 0034 | Under-specified objective | **T1** | Regex on task statement — absence of deliverable verbs (`return` / `output` / `fix` / `verify`). |
| 0035 | Defective generated artifact | **T2** | tool_call succeeded + downstream test/verifier span failed — structural mismatch. |
| 0036 | Stated-reasoning unfaithfulness | **T4** | PromptExp attribution + LLM-judge alignment check (Watson §III-D); inherently semantic. |
| 0037 | Sibling-source contradiction unresolved | **T3** | Rule (graph): detect IDG conflict edge `c_ij=1` via NER on sibling outputs. LLM confirms synthesis span did or didn't arbitrate. |

## Tier counts

| Tier | Count | % of corpus |
|---|---|---|
| **T1 pure rule** | 8 | 22% |
| **T2 structural** | 9 | 24% |
| **T3 hybrid** | 8 | 22% |
| **T4 semantic** | 12 | 32% |
| Rule-solvable (T1+T2) | **17** | **46%** |
| Semantic-required (T3+T4) | **20** | **54%** |

## What goes in each tier

### T1 — Pure rule (8 cards)

These fire from log strings, schema validation, or counters. Implementable in <100 LOC each, no model.

- AFC-0004 Conversation history wipe
- AFC-0014 Action format error
- AFC-0023 Verification step skipped
- AFC-0027 Step / round limit exhausted
- AFC-0031 Tool execution error
- AFC-0032 LLM API limit
- AFC-0033 Environment error
- AFC-0034 Under-specified objective

Note: half of T1 is "infrastructure layer" (0031/0032/0033/0027) — these are the *easiest* to detect because the runtime / provider / OS already raises a structured error. Production observability stacks already capture them.

### T2 — Structural (9 cards)

Need a pre-trained embedding (sentence-transformer or similar), or NER, or graph property — but no LLM call per detection.

- AFC-0010 Impossible action — tool string match
- AFC-0015 Action parameter error — tool string match
- AFC-0017 Role specification deviation — actor → action-whitelist lookup
- AFC-0018 Goal drift — embedding similarity over time
- AFC-0019 Repeated handled work — span-pair similarity
- AFC-0021 Information withholding — NER + set-diff
- AFC-0022 Peer feedback ignored — pre-vs-post-feedback action similarity
- AFC-0026 Non-termination / endless loop — similarity-ridge detection
- AFC-0035 Defective generated artifact — tool-call-success + verifier-fail mismatch

EAGER's Reasoning-Scoped Contrastive Learning (§3.2) is the most relevant prior art for this layer: a single trained encoder gives Action-Consistent / Progressive-Aligned / Cross-Agent-Negative scores that cover several of these cards at once.

### T3 — Hybrid: rule narrows, LLM confirms (8 cards)

Cheap rule eliminates the bulk of negatives; LLM is called only on the small candidate set.

- AFC-0002 False memory — rule: NER + history search; LLM: paraphrase vs fabrication
- AFC-0005 Progress misassessment — rule: completion-string + env state; LLM: subgoal status
- AFC-0008 Reflection hallucination — rule: cited events vs upstream spans; LLM: confirm absence
- AFC-0009 Constraint ignorance — rule when structured; LLM when NL
- AFC-0012 Backtracking absence — rule: revision similarity; LLM: strategy isomorphism
- AFC-0025 Premature termination — rule: TERMINATE token; LLM: subgoal completeness
- AFC-0028 Wrong-location fixation — rule: target-entity repeat after failure; LLM: confirm wrong target
- AFC-0037 Sibling-source contradiction — rule: NER on siblings → conflict edge; LLM: arbitration check

### T4 — Pure semantic (12 cards)

No useful rule layer; only an LLM-as-judge or abductive reasoning step decides.

- AFC-0001 Memory over-simplification
- AFC-0003 Memory retrieval failure
- AFC-0006 Outcome misinterpretation
- AFC-0007 Causal misattribution
- AFC-0011 Inefficient plan / wrong decomposition
- AFC-0013 Planning–action disconnect
- AFC-0016 Task specification deviation
- AFC-0020 Ambiguous request to peer
- AFC-0024 Incorrect verification
- AFC-0029 Tool-output override
- AFC-0030 Critical-evidence miss
- AFC-0036 Stated-reasoning unfaithfulness

These are the cards where literature step-level accuracy on Who&When peaks at ~50% (A2P, GraphTracer-8B). They're hard *for everyone*, including SOTA reasoners. AgentLens' AgentLens 2.4.2 sketch — Sibling Agent + structured trajectory preprocessing — is most needed here.

## Implications for AgentLens harness design

1. **Cheap-first layered pipeline**: run T1 on every step, T2 on every trajectory tail, T3 only when T1+T2 left a *suspicion* (e.g., task failed but no rule fired, or low-confidence T2 hit), T4 only when T3 returned ambiguous. This matches the "粗筛 → 精诊" framing in the AgentLens 2.4.2 outline. Concretely: with T1+T2 covering 17/37 cards (46%), the LLM-judge layer only has to decide on the remaining 20 — substantial cost reduction vs running judge per step.
2. **Half of T1 is already in production observability**: AFC-0031/0032/0033/0027 are runtime errors most stacks capture. Make sure the AgentLens harness *consumes* these signals rather than re-implementing them.
3. **T2 needs one good encoder, not eight separate ones**: EAGER's contrastive encoder gives 4-pair semantics covering AFC-0013, AFC-0019, AFC-0021, AFC-0022 in a single forward pass. Investing in one trained encoder pays back across the whole row.
4. **T3 is where rule + LLM cooperation matters most**: each T3 card has a rule that reduces 1000s of trajectory steps to ~5 candidates, then 1 LLM call confirms. This is the operationally cheapest form of LLM-judge use. Bake the candidate-selection step into the harness as a first-class primitive, not an ad-hoc prompt prefix.
5. **T4 cards bound the achievable accuracy**: 32% of the corpus needs semantics. No amount of rule engineering will close the gap on AFC-0006 (outcome misinterpretation), AFC-0007 (causal misattribution), AFC-0011 (inefficient plan), AFC-0024 (incorrect verification), AFC-0036 (unfaithful CoT). The realistic question is *which T4 cards a sibling agent must handle in real-time vs which can be deferred to offline post-mortem*.
6. **The asymmetry between detection and attribution**: many T2 cards (e.g., AFC-0026 endless loop, AFC-0019 repeated work) detect the *symptom* cheaply but say nothing about the *cause* — which lives in T4 (e.g., AFC-0011 inefficient plan caused the loop). Production telemetry should surface T1+T2 alarms; AgentLens' value-add is climbing the chain to T3+T4 root causes when an alarm fires.

## Caveats

- Tier ratings assume *standard* trajectory shape (structured tool calls + free-form reasoning). Heavily structured agents (typed actions only) push some T2/T3 cards down to T1; conversely, very free-form agents (no schema) push T1 cards up.
- "Pure rule" doesn't mean "rule alone is the right detector". Rules are noisy — they need calibration on labeled data. The point is they don't require an LLM.
- Cards can sit in multiple tiers depending on what *level* of detection you want: AFC-0026 endless loop is T2 to *flag* the loop, but T4 to attribute *why* the agent looped.
