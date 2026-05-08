# Agent-Failure Cards (AFC) — schema-compatible with chaoschain

This directory aggregates **named failure patterns** observed across the 10
surveyed papers in the parent directory (see `../INDEX.md`,
`../FAILURE-PATTERNS-SYNTHESIS.md`). Each card describes one mechanism by
which an LLM agent / multi-agent system fails.

The format is **adapted from
`/home/ddq/AoyangSpace/chaoschain/docs/schema-v0.1.md`** — same shape
(defect / activation / observable / downstream_effects / evidence), but
the controlled vocabularies have been forked for the LLM-agent domain.
chaoschain's vocabulary targets infrastructure faults (connection pools,
GC, latency); ours targets cognitive / coordination faults (context drop,
plan inefficiency, peer ignorance).

Cards are research summaries, not yet validated by chaos experiments.
`reproducibility.confirmed_in_chaos` is therefore `false` everywhere.

## Card ID

`AFC-NNNN` (Agent-Failure Card). IDs are stable; never reused.

## Forked controlled vocabularies

### Defect class — `defect.class`

| Class | What it covers |
|---|---|
| `memory_fault` | Past context misused: forgotten, fabricated, summarized away. |
| `reflection_fault` | Self-monitoring fails: progress / outcome / cause misread. |
| `planning_fault` | Strategy itself is wrong: infeasible, inefficient, ignores constraints. |
| `action_fault` | Plan-to-action translation breaks: format / parameter / disconnect. |
| `specification_fault` | Agent abandons assigned task / role / goal. |
| `coordination_fault` | Multi-agent handoff fails: ignore, withhold, repeat, ambiguous. |
| `verification_fault` | Self-checking is skipped or wrongly approves bad work. |
| `termination_fault` | Stop condition misfires: too early, never, or budget cap. |
| `cognitive_fault` | Reasoning-level error not localized to one module: fixation, override, miss. |
| `system_fault` | External / infrastructure layer fails (tool, model API, env). |

### Predicate vocabulary — `<scope>.<object>.<operation>[.modifier]`

Same shape as chaoschain. `scope` is a controlled enum; `object` is free
text within an agent-domain set; `operation` is a controlled enum.

**Scope (enum)**

| Scope | Examples (object) |
|---|---|
| `agent_state` | context, memory, role, goal, plan |
| `trajectory` | step, action, tool_call, observation, reasoning |
| `coordination` | peer, message, request, handoff |
| `quality` | plan, observation, decomposition, verification |
| `outcome` | task, subtask, termination |
| `system` | tool, llm_api, environment, budget |

**Operation (enum)**

| Op | Meaning |
|---|---|
| `dropped` / `wiped` | context/state removed (FM-1.4-style). |
| `fabricated` | invented content not in input (false memory, hallucinated event). |
| `not_retrieved` | needed context exists but agent didn't fetch it. |
| `misread` / `misjudged` / `misattributed` | observation/progress/cause interpretation wrong. |
| `inefficient` / `infeasible` / `disconnected` | plan structurally bad / impossible / unimplementable. |
| `malformed` / `invalid_param` | action shape wrong. |
| `repeated` / `looped` | step or handled work redone. |
| `ignored` / `withheld` / `ambiguous` | coordination message degraded. |
| `skipped` / `incorrect_pass` | verification step missing or wrongly accepts. |
| `terminated_early` / `unbounded` / `exhausted` | termination malfunctions or budget hits cap. |
| `fixated` / `overridden` / `missed` | cognitive: anchored on wrong location, ignored tool feedback, missed evidence. |
| `error` / `limit` / `corrupted` | system layer broke. |
| `amplified` | downstream blow-up (cascade). |

### Detection difficulty (same as chaoschain)

`easy` / `medium` / `silent`. "Silent" failures are the high-value class:
trajectory looks plausible, end task fails, root cause is upstream and
invisible to surface checks.

### Confidence / timescale (same as chaoschain)

- `confidence`: `low` / `medium` / `high`
- `timescale`: `sub_second` / `seconds` / `minutes_to_hours` / `hours_to_days`
  (for agents, "minutes_to_hours" maps to long autonomous trajectories,
  "seconds" maps to within a single LLM turn).

## Layout

Cards are grouped by `defect.class` into subdirectories:

```
cards/
├── memory/           # AFC-0001..0004 + AFC-0039 (post-obs fabrication) + AFC-0040 (abductive drift)
├── reflection/       # AFC-0005..0008
├── planning/         # AFC-0009..0012 + AFC-0038 (hypothesis-irrelevant tool)
├── action/           # AFC-0013..0015 + AFC-0035 (defective artifact)
├── specification/    # AFC-0016..0018 + AFC-0034 (under-specified objective)
├── inter_agent/      # AFC-0019..0022
├── verification/     # AFC-0023..0024 + AFC-0037 (sibling-source contradiction)
├── termination/      # AFC-0025..0027 + AFC-0042 (coarse-granularity)
├── cognitive/        # AFC-0028..0030 + AFC-0036 (stated-reasoning unfaithful) + AFC-0041 (rationalizing pseudo-backtrack)
└── system_level/     # AFC-0031..0033
```

## Index

| ID | Name | Class | Sources |
|---|---|---|---|
| AFC-0001 | Memory over-simplification | memory | AgentDebug |
| AFC-0002 | False memory / fabricated recall | memory | AgentDebug |
| AFC-0003 | Memory retrieval failure | memory | AgentDebug |
| AFC-0004 | Conversation history wipe | memory | AEGIS FM-1.4 |
| AFC-0005 | Progress misassessment | reflection | AgentDebug |
| AFC-0006 | Outcome misinterpretation | reflection | AgentDebug, AgentDiagnose |
| AFC-0007 | Causal misattribution | reflection | AgentDebug |
| AFC-0008 | Reflection hallucination | reflection | AgentDebug |
| AFC-0009 | Constraint ignorance | planning | AgentDebug |
| AFC-0010 | Impossible action planned | planning | AgentDebug |
| AFC-0011 | Inefficient plan / wrong decomposition | planning | AgentDebug, EAGER, AgentDiagnose |
| AFC-0012 | Backtracking absence | planning | AgentDiagnose |
| AFC-0013 | Planning–action disconnect | action | AgentDebug, AEGIS FM-2.6 |
| AFC-0014 | Action format error | action | AgentDebug |
| AFC-0015 | Action parameter error | action | AgentDebug, EAGER |
| AFC-0016 | Task specification deviation | specification | AEGIS FM-1.1 |
| AFC-0017 | Role specification deviation | specification | AEGIS FM-1.2 |
| AFC-0018 | Goal drift | specification | AEGIS FM-2.3 |
| AFC-0019 | Repeated handled work | inter_agent | AEGIS FM-2.1 |
| AFC-0020 | Ambiguous request to peer | inter_agent | AEGIS FM-2.2 |
| AFC-0021 | Information withholding | inter_agent | AEGIS FM-2.4 |
| AFC-0022 | Peer feedback ignored | inter_agent | AEGIS FM-2.5 |
| AFC-0023 | Verification step skipped | verification | AEGIS FM-3.2, AgentDiagnose |
| AFC-0024 | Incorrect verification | verification | AEGIS FM-3.3 |
| AFC-0025 | Premature termination | termination | AEGIS FM-3.1 |
| AFC-0026 | Non-termination / endless loop | termination | AEGIS FM-1.3 + FM-1.5 |
| AFC-0027 | Step / round limit exhausted | termination | AgentDebug, EAGER |
| AFC-0028 | Wrong-location fixation | cognitive | Watson, EAGER (Localization Error) |
| AFC-0029 | Tool-output override | cognitive | ELPO |
| AFC-0030 | Critical-evidence miss | cognitive | EAGER (Trace Miss), AgentDiagnose |
| AFC-0031 | Tool execution error | system_level | AgentDebug |
| AFC-0032 | LLM API limit | system_level | AgentDebug |
| AFC-0033 | Environment error | system_level | AgentDebug |
| AFC-0034 | Under-specified objective | specification | AgentDiagnose |
| AFC-0035 | Defective generated artifact | action | EAGER, AgenTracer |
| AFC-0036 | Stated-reasoning unfaithfulness | cognitive | Watson |
| AFC-0037 | Sibling-source contradiction unresolved | verification | GraphTracer |
| AFC-0038 | Hypothesis-irrelevant tool invocation | planning | GoS |
| AFC-0039 | Post-observation evidence fabrication | memory | GoS |
| AFC-0040 | Long-horizon abductive context drift | memory | GoS |
| AFC-0041 | Rationalizing pseudo-backtrack | cognitive | GoS |
| AFC-0042 | Coarse-granularity termination | termination | GoS |

## Future schema extensions (not yet applied)

Reviewer agents proposed three orthogonal axes worth adding once a card
needs them. None applied yet — listed here so the next pass knows where
to look:

1. **`defect.scope: intra_agent | inter_agent | cross_scope`** — sourced
   from EAGER §3 / Figure 1's intra- vs inter-agent reasoning split.
   Would cleanly tag AFC-0017 / AFC-0019..0022 as `inter_agent`,
   AFC-0001..0015 + AFC-0028..0030 as `intra_agent`, and let
   AFC-0011 / AFC-0023..0027 be `cross_scope` (their meaning depends
   on whether the system is single- or multi-agent).
2. **`position.is_first_irrecoverable: bool`** — a *position* attribute
   ELPO §4.1 / Algorithm 1 BEL operationalizes via binary-search
   rollout trees. Cross-cuts the taxonomy: any card can host such a
   step under tight rollout budget.
3. **`localization_granularity: agent_step | agent | trace`** — sourced
   from EAGER §3.1's fine-grained vs coarse-grained failure-knowledge
   split. AFC-0011 (decomposition) is a clean case — agent-step in
   single-agent ReAct, trace-level in AutoGen-Code where no single
   agent owns the decomposition.

A *non-failure* observation also worth documenting (not card-worthy):
Watson §II-C cites Liu et al. and Su et al. that fast-thinking decoding
itself differs structurally from CoT-mode generation, with up to 36%
accuracy drop when forcing reasoning. This is a design tradeoff, not a
runtime fault — captured here so future readers don't try to make a
card out of it.

## What this collection is NOT

- Not a chaoschain submission. chaoschain models infrastructure cascades;
  these cards model agent reasoning/coordination. Future work could
  attempt a unified cross-domain vocabulary, but for now keep these in
  the AgentM research tree.
- Not validated. Each card is grounded in published descriptions, not
  in a chaos experiment.
- Not exhaustive. The corpus is 10 papers; new patterns from MAST or
  follow-up work should be added as new AFC IDs.
