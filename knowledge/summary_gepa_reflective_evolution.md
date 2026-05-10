# GEPA: Reflective Prompt Evolution Can Outperform RL — Summary & AgentM Implications

**Paper**: arxiv.org/abs/2507.19457 (Agrawal et al., ICLR 2026 submission)
**Tag**: `gepa_reflective_evolution`
**Read on**: 2026-05-09 (in context of AgentM's per-task evolution loop, PR #67)

---

## 1. The thesis in one paragraph

For optimizing **compound AI systems** (multi-LLM-call pipelines: agents, multi-agent
systems, language programs), the dominant approach has been RL methods like
GRPO that derive policy gradients from sparse scalar rewards over thousands of
rollouts. GEPA argues that **rollouts already contain natural-language traces**
(reasoning chains, tool calls, error messages, eval rubric outputs) and that
those traces are a much richer learning signal than the scalar reward they're
collapsed into. By **reflecting on traces in natural language**, GEPA can
extract large quality gains from a handful of rollouts — beating GRPO by up
to 20% with **35× fewer rollouts** on six benchmarks, and beating MIPROv2
(prior SOTA prompt optimizer) by +13% aggregate.

## 2. Formal framing

A compound AI system is `Φ = (M, C, X, Y)`:
- `M = ⟨M_1, ..., M_|M|⟩` — language modules, each `M_i = (π_i, θ_i, X_i, Y_i)` with prompt `π_i`, weights `θ_i`, I/O schemas
- `C` — control flow code, can call modules in any order, may invoke tools
- Learnable params: `⟨Π, Θ⟩_Φ` (prompts + weights)

Optimization problem (sample-efficient form):

```
⟨Π*, Θ*⟩ = argmax  E_(x,m)~T  [μ(Φ(x; ⟨Π,Θ⟩), m)]
            ⟨Π,Θ⟩
       s.t.  #rollouts ≤ B
```

GEPA evolves only `Π` (prompts). Weights stay frozen. The cost unit is a **rollout** (one Φ call + one μ eval).

## 3. The three pillars

### 3.1 Genetic optimization loop

Maintain a **candidate pool** `P` (each candidate = a concrete instantiation of all module prompts). Loop:
1. Select a promising candidate (Pareto-based; see §3.3)
2. Mutate it (reflective; see §3.2) → variant Φ'
3. Evaluate Φ' on a **minibatch** of training tasks
4. If minibatch score improves vs parent → run on `D_pareto` (validation set), add to P with ancestry record
5. Repeat until budget B exhausted, return best on `D_pareto`

Crucially, each child **inherits its parent's accumulated lessons via the prompt** (the parent's prompt becomes the starting point). The candidate pool is a tree, not a linear chain.

### 3.2 Reflective prompt mutation

The core trick. Given a candidate to mutate:
1. Run on a minibatch, capture **execution traces** (each module's I/O + reasoning) AND **evaluation traces** (compiler errors, rubric checks, etc. — natural-language byproducts of computing μ)
2. Pass `(current_prompt, traces, score, feedback_text)` to a **reflection LM**
3. Reflection LM proposes an updated prompt for **one** module per iteration (round-robin module selection)
4. New candidate = old candidate with that one module's prompt replaced

The key shift: **`μ` becomes `μ_f`**, a feedback function returning `(score, feedback_text)` rather than just a scalar. The text is what the reflection LM reads.

> "The text that LLMs produce is the *execution trace* of the AI system. The text that the environment produces to compute the reward (e.g. compiler error messages before giving reward 0) is the *evaluation trace*."

### 3.3 Pareto-based candidate selection (anti-local-optimum)

For each training instance `i`, record the highest score achieved by any candidate. Candidates that are **best on at least one instance** form the Pareto frontier. Sample the next candidate to mutate **stochastically from the frontier**, weighted by the number of tasks each is best at. Strictly dominated candidates are pruned.

Why this matters: greedy "always mutate the global best" gets stuck — once you find a strategy that scores 80% overall, all mutations are evaluated against that incumbent and most fail. Pareto preserves a candidate that scores 30% overall but solves *one task no other candidate does*, because that candidate carries a lesson worth recombining.

This is borrowed from MAP-Elites/quality-diversity literature: **illuminate the search space**, don't just climb to the nearest peak.

## 4. The hidden gem: Section commented out but real

Lines 435–442 of the main TeX (commented out, but in the appendix/results): **GEPA search-tree nodes don't have to be prompts. They can be the entire system code** — modules, prompts, control flow, all of it.

> "While we previously used GEPA search-tree nodes to represent sets of prompts in an AI system, these nodes can be extended to the code describing the *whole* AI system, including its subcomponents (modules including their prompts, control flow, etc.). Without any other changes, GEPA can then reflectively evolve the whole AI system through code, including the evolution of the number of modules, detailed prompts to each of the modules, task decomposition across modules, and the control flow orchestrating the system."

Concrete result: GEPA + Gemini-2.5-Pro evolved a **single-step ChainOfThought** agent into a **6-step self-refining agent** (induce rule → write Python → execute on training samples → debug → refine system → apply) that lifted ARC-AGI-1 from 44% → 49.5%.

This is what the user is asking for: the optimization target is not "the prompt", it is **the whole system**, including its harness.

## 5. Sample-efficiency receipts

- HotpotQA, IFBench, PUPA, HoVer, FSABench (Qwen3 8B): GEPA matches GRPO's peak using **1.8% of GRPO's rollouts**; +7.8% better than GRPO at **12.4%** of its budget
- Up to 20% better than GRPO with 35× fewer rollouts
- One reflective update often produces large gains; the speedup comes from rich signal per rollout, not algorithmic cleverness

The arithmetic: a single rollout's traces, parsed as language, contain orders of magnitude more bits than the eventual scalar reward. RL throws those bits away; GEPA reads them.

---

# 6. Implications for AgentM's per-task evolution loop

This is the section that matters. Mapping each pillar onto the current PR-67 design (`.claude/designs/per-task-evolution-loop.md`).

## 6.1 What we already have

| GEPA concept | AgentM equivalent (today) |
|---|---|
| Compound system Φ | A scenario manifest (atoms + system_prompt + control via the agent loop) |
| Module M_i prompt π_i | Atom source / `system_prompt` text / scenario manifest |
| Metric μ | `eval/grader.py` (scalar) |
| Training set | `eval/tasks/*.yaml` (with `holdout` flag) |
| Rollout budget B | `max_turns` per task; `max_cost_usd` (proposed §11) |
| `D_pareto` | Same eval suite, used for full evaluation |

## 6.2 What we're missing — and what to change

### 6.2.1 Promote candidate pool from "linear git history" to **explicit search tree with Pareto frontier**

**Today**: One activated atom version per scenario; older versions live only in git log. `decisions.jsonl` is a flat audit trail. The "incumbent vs proposed" gate is binary — accept or discard.

**Problem**: This is exactly the greedy-best strategy GEPA shows gets stuck. Our `exploratory` channel is a band-aid; the underlying issue is we don't preserve diverse winners.

**Change**:
- Add `.agentm/decisions/<scenario>/candidates/<id>.json` — one record per candidate, storing `{parent_id, atom_changes: dict[name, sha], per_task_scores: dict[task_id, float], holdout_scores: ...}`
- `tool_propose_change` no longer requires "beats incumbent" — it requires "wins on at least one task vs all retained candidates". Pareto-dominated candidates are pruned.
- Activation (which version production loads) becomes a **separate decision** from inclusion in the pool. The pool is the search frontier; activation picks one of its members.
- `tool_query_candidates(scenario)` atom — returns the Pareto frontier so the tuner can pick a parent stochastically.

This decouples *search* (Pareto pool) from *deployment* (which member to activate).

### 6.2.2 Replace the 4-floor gate's role: gate **deployment**, not **inclusion in pool**

The 4-floor gate (threshold + 2σ + guards + holdout) we just added in `3a59e9e` is the right idea, but it's gating the wrong thing.

- **Pool inclusion**: should be loose — any candidate that wins on any task or finds a new strategy is kept. (GEPA's pool has dozens of candidates; ours has 1.)
- **Production activation**: should be tight — the 4 floors apply *here*, when picking which pool member to make live for end users.

This separation is the difference between an exploration algorithm and a deployment policy. We've conflated them.

### 6.2.3 Make `μ_f` (feedback function) the contract, not `μ` (scalar)

**Today**: `grader.py` returns `{score, rationale}`. The `rationale` is for humans; the tuner LLM reads it but the framework doesn't structurally pipe it to the mutation prompt.

**Change**:
- Formalize the feedback function: `grade(task, output, trace) -> {score: float, dimensions: dict, feedback_text: str, module_feedback: dict[str, str]}`
- `feedback_text` is fed verbatim into the tuner's reflection prompt as `<failure_diagnosis>`. Currently the tuner has to fish it out of `tool_query_traces`; structurally pipe it.
- Add **module-level feedback** (e.g., per-atom): GEPA shows multi-hop systems benefit when the grader can say "hop 2 retrieval was wrong". For us this means the grader can attribute failure to a specific atom or to the system_prompt. Then the round-robin module-selection policy from GEPA §3.2 maps cleanly: tuner mutates one (atom/prompt) per iteration, biased toward modules the grader fingered.

### 6.2.4 Add a `tool_reflect` atom — separate the diagnosis step from the mutation step

**Today**: Tuner does diagnosis and mutation in one prompt (the system prompt at `tuner/prompt.md`). GEPA's reflection LM is a *separate call* that produces the mutation proposal.

**Change**:
- New tier-1 atom `tool_reflect(failures: list[TraceSummary], target_module: str) -> {diagnosis: str, proposed_mutation: str}`
- The tuner's loop becomes: `query_traces → reflect → eval_run(baseline) → eval_run(proposed with override) → propose_change`
- `tool_reflect` itself uses a meta-prompt template that's separately versioned (and itself mutable via GEPA — meta-meta path the paper notes).

This makes the reflection step swappable: someone can drop in a more sophisticated reflection mechanism without touching the tuner's outer loop.

### 6.2.5 Expand the mutation space — **everything mutable, not just atoms**

User's explicit ask: prompt + code + tool harness, "any optimizable place". GEPA paper backs this with the ARC-AGI result.

**Today's `tool_propose_change`** mutates one thing: an atom file. It cannot:
- Edit `system_prompt` text
- Add/remove atoms from the manifest extension list
- Change tool exposure (which tools are registered)
- Change the scenario's behavioral control flow

**Change**: Generalize `target` from `target_atom: str` to `target: ChangeSpec`:

```python
class ChangeSpec(TypedDict):
    kind: Literal["atom_source", "system_prompt", "manifest_extensions",
                  "manifest_field", "scenario_compose"]
    path: str          # where in scenarios/<name>/ to write
    new_content: str   # full replacement OR structured patch for manifest
    rationale: str
```

The §11 atom contract (one file, no atom-to-atom imports, etc.) still bounds atom-shaped changes. Manifest-shaped changes go through the same ResourceWriter; new validators per change kind.

This is the user's "tool harness 也可改动" made concrete: a mutation can register a new tool, retire one, or change the tool's name.

### 6.2.6 Add System Aware Merge (crossover)

**Today**: We have mutation only. Two improvements that fix different failure modes have no way to combine — they get evaluated head-to-head and one is discarded.

**Change**:
- New decision channel `decision="merge"` taking `parents: list[candidate_id]`
- Tuner reads two non-dominated candidates, calls `tool_reflect` with their combined per-task scores, generates a unified change spec that takes lessons from both
- Merge is gate-checked the same way activation is

GEPA's results show merge contributes meaningfully on multi-module tasks. For us this is most valuable when one mutation fixes the atom and another fixes the prompt — a merge gives the union.

### 6.2.7 Track rollouts as the cost unit

GEPA's headline (35× fewer rollouts) hinges on counting rollouts cleanly. Today we have `max_cost_usd` (USD) and `max_turns` (per task) but no scenario-wide rollout counter.

**Change**: `tool_eval_run` and `tool_propose_change` both increment `.agentm/decisions/<scenario>/budget.json: {rollouts_used, rollouts_budget, usd_used, usd_budget}`. Rollouts cap is per-tuning-session; the budget is what the activation gate's noise-floor compensates for.

### 6.2.8 Re-examine the "exploratory" channel

With Pareto pool: most of what `exploratory` exists for (insufficient evidence but I want to ship anyway) becomes "candidate is in the pool but not activated". The activate decision can wait for evidence to accumulate without losing the candidate.

`exploratory` then narrows to its real use case: "I want to ship this version to production traffic to gather more eval signal, even though my offline gate isn't satisfied." Distinct from search-time exploration.

## 7. Suggested redesign skeleton (replaces parts of PR #67)

```
.agentm/
├── decisions/<scenario>/
│   ├── candidates/<candidate_id>.json   # one per pool member
│   ├── tree.jsonl                       # parent → child edges
│   ├── pareto.json                      # current frontier (cached, regenerable)
│   ├── activations.jsonl                # which candidate is live, when
│   └── budget.json                      # rollout / cost counters
└── observability/...                    # unchanged

contrib/scenarios/<name>/
├── manifest.yaml                        # ← also mutable now
├── system_prompt.md                     # ← extracted out so it's a mutation target
├── <atom>.py                            # mutable
├── eval/
│   ├── tasks/*.yaml
│   ├── grader.py                        # returns μ_f, not just μ
│   └── reflection_template.md           # ← new: μ_f → diagnosis → mutation prompt
└── tuner/
    ├── manifest.yaml
    └── prompt.md                        # ← mostly the outer loop now
```

New atoms (replace the trio):
- `tool_query_traces` — unchanged
- `tool_query_candidates(scenario)` — returns Pareto frontier ★ new
- `tool_reflect(failures, target_module) → {diagnosis, mutation}` ★ new
- `tool_eval_run` — unchanged signature, but receives `feedback_text` from `μ_f`
- `tool_propose_change(target: ChangeSpec, parent_ids, scores, decision)` — generalized
- `tool_merge(parent_ids: list, change: ChangeSpec, ...)` ★ new

## 8. What this means for the current PR #67 doc

The doc as it stands (post-3a59e9e) is internally consistent, **but it commits to a single-incumbent gated-promotion model that the GEPA evidence says is suboptimal**.

Three options:

**A. Keep PR #67 as MVP, do GEPA-style overhaul as Phase 2.**
Pros: ships what's working; current implementation just needs the `holdout` field plus the four-floor gate, both already merged.
Cons: every Phase-2 change touches the contract again.

**B. Pause PR #67, redesign around Pareto pool + ChangeSpec + reflect atom.**
Pros: future-proof; matches user's "终极目标是真正的自进化" (true self-evolution).
Cons: 2–3 weeks to redo MVP; format_fix scenario already validates the simpler model.

**C. Hybrid: keep MVP single-incumbent, but design the data structures (`candidates/`, `tree.jsonl`, `ChangeSpec`) so adding Pareto is a non-breaking schema extension.**
Pros: lowest regret. The candidate pool with a single member IS a degenerate Pareto pool.
Cons: requires renaming current `decisions.jsonl` → `activations.jsonl` and adding the `candidates/` directory now even if only one member exists.

**My read**: C is the right call. Specific concrete edits to PR #67:

1. Rename `decisions.jsonl` → `activations.jsonl` (semantic clarity); add `candidates/` dir to the schema even if MVP only writes one entry per activation.
2. Generalize `target_atom: str` → `target: ChangeSpec` in `tool_propose_change` signature, but allow only `kind="atom_source"` in MVP. Other kinds reserved.
3. Document the future Pareto pool in §11 Phase 2 and §13 Open Questions; state that the four-floor gate's role becomes "deployment gate" once Pareto exists.
4. Add `μ_f` distinction to §3.2 (grader returns `feedback_text` not just `rationale`) — costs nothing today and unblocks reflect-atom later.
5. Note in §5.2 that the tuner system prompt is itself mutable by a meta-tuner (the paper's "tuner can tune its own prompt — meta-meta if you want it" hint).

These five tweaks turn PR #67 from "the system" into "the MVP slice of the GEPA system" without breaking anything.

## 9. Connection to AgentM's stated direction

From `MEMORY.md`:
- "Self-Modifiable + Evidence-Driven Evolution: core/extensions split by 'what agent can safely self-modify'; **evolution needs catalog of versioned atoms paired with observation data, not just version snapshots**"
- "AgentM Multi-Agent Direction: user plans to implement similar agent-team philosophy in AgentM"

Both align with GEPA. The catalog + observation requirement is exactly the candidate pool with `per_task_scores` per candidate. The multi-agent angle aligns with mutating the tool harness / control flow as GEPA does in the ARC-AGI experiment.

---

## 10. One-liner takeaway for the design discussion

> The current per-task evolution loop is **hill climbing with a noise-aware gate**. GEPA's evidence says the right shape is **MAP-Elites style illumination over a Pareto pool of candidate systems**, where mutations come from natural-language reflection on traces and evaluation rubrics, and where the entire system (prompts, atom code, tool harness, control flow) is the search space. Same primitives we have now, reorganized so that diversity is preserved instead of greedily collapsed.
