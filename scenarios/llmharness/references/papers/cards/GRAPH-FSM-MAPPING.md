# Mapping AFC cards to a GoS-style graph + state-machine abstraction

> Question: for the cards that needed *semantic judgment* (T3+T4 in
> `DETECTION-FEASIBILITY.md`), can we represent them as state-machine
> invariants on a structured belief graph — replacing LLM-as-judge with
> graph property checks?

Answer: **roughly half — 10-12 of the 20 semantic cards collapse cleanly
to graph/FSM invariants**. The other half remains genuinely semantic
because the violation is in *content interpretation*, not *state
topology*.

This doc reuses the abstraction proposed in **Graph of States (GoS)**
— `~/.cache/nanochat/knowledge/2603.21250/sections/Methodology.tex` —
and maps each AFC onto a concrete invariant where one exists.

## The GoS abstraction in one paragraph

GoS represents the agent's reasoning state as a tuple **B = (G, S)**
plus a *reasoning focus* h\*:

- **Causal graph G = (V, E)**. Nodes have three types: symptom `v_sym`,
  evidence `v_evi`, hypothesis `v_hyp`. Each `v_hyp` has a confidence
  score `P(v_hyp)` and a granularity level `L(v_hyp)`. Edges are typed:
  `derive(v_sym → v_hyp)`, `refine(v_hyp_coarse → v_hyp_fine)`,
  `support(v_evi → v_hyp)`, `refute(v_evi → v_hyp)`.
- **State machine S_t = current investigation level (∈ ℕ⁺)**. State
  advances by *drill-down* (S_t+1 = S_t + 1, deeper) or *backtracking*
  (S_t+1 = l\*, shallower-than-current ancestor that got demoted).
- **Drill-down trigger**: `P(v_hyp_top) − P(v_hyp_2nd) > δ` AND
  `|E_sup(v_hyp_top)| ≥ η`. Both confidence gap and evidence count must
  pass.
- **Backtracking trigger**: any ancestor at level l < S_t loses its
  highest-confidence-among-siblings status due to a `refute` edge.
- **Reasoning focus**: `h*_t = argmax_{h∈V_hyp, L(h)=S_t} P(h | G_t)`.

GoS itself targets four deficiencies that map directly to AFCs:

| GoS deficiency | AFC card |
|---|---|
| Evidence Fabrication | AFC-0002 (false memory), AFC-0008 (reflection hallucination), AFC-0030 (critical-evidence miss inverse) |
| Context Drift | AFC-0011 (inefficient plan), AFC-0019 (repeated work), AFC-0026 (endless loop) |
| Failed Backtracking | AFC-0012 (backtracking absence) |
| Early Stopping | AFC-0025 (premature termination) |

So even *before* extending the abstraction, GoS already covers 8 of the
T3+T4 cards by construction.

## Cards-as-invariants: the 20 semantic cards

Reading order: **AFC-NNNN — invariant** (✓ = fits cleanly, ◐ = partial,
✗ = doesn't fit, needs LLM).

### T3 hybrid — 8 cards

- **AFC-0002 False memory** ✓ — *invariant*: `∀ v_evi ∈ V . provenance(v_evi) ≠ ∅`. Every evidence node must trace back to a tool_call span. Fabricated evidence has no provenance edge to any tool output. Pure graph property check.
- **AFC-0005 Progress misassessment** ◐ — partly: the agent's claim of "task done" must coincide with `S_t = max(L(v_hyp))` AND drill-down threshold satisfied. If TERMINATE emitted while either condition fails → invariant violation. Doesn't catch "agent thinks completion is closer than it is" without LLM.
- **AFC-0008 Reflection hallucination** ✓ — same invariant as 0002 but applied to nodes referenced in a `reflection` span: `∀ entity_cited_in_reflection . ∃ matching_node ∈ V`.
- **AFC-0009 Constraint ignorance** ◐ — fits if constraints are *modeled as edge predicates* (e.g., budget edges with cost annotations). GoS doesn't natively have constraints; needs schema extension `v_constraint` node type. Then invariant = `cost(plan) ≤ budget`.
- **AFC-0012 Backtracking absence** ✓ — *literally* GoS's backtracking trigger: ancestor at l < S_t demoted but state hasn't reset to l\*. Detectable by inspecting the (G_t, S_t) pair and the confidence-recalibration history. **GoS targets this card by design.**
- **AFC-0025 Premature termination** ✓ — *literally* GoS's drill-down threshold: termination emitted while `P_top − P_2nd ≤ δ` OR `|E_sup| < η`. **GoS targets this card by design.**
- **AFC-0028 Wrong-location fixation** ✓ — *invariant*: focus `h*_t` cannot persist for more than K iterations without either confidence increasing or a `refute` edge appearing. If `h*_t` is unchanged for K steps and `P(h*_t)` is flat → fixation. Pure FSM property.
- **AFC-0037 Sibling-source contradiction** ✓ — *invariant*: `∀ pair (v_evi_a, v_evi_b) with edges to same v_hyp where polarity(a) ≠ polarity(b) . ∃ resolution_node`. Two evidence siblings with opposing support/refute polarity must produce a resolution span. Detectable as graph-local pattern match.

**T3 fit: 6/8 clean, 2/8 partial.** Big win — these all become rule-firing on graph state, no LLM needed once the graph is constructed.

### T4 semantic — 12 cards

- **AFC-0001 Memory over-simplification** ✗ — the violation is in the *content* of a summary node vs the source observation. Comparing detail loss requires semantic understanding. Graph cannot represent "what specificity got dropped".
- **AFC-0003 Memory retrieval failure** ✗ — requires knowing what's *relevant*, which is task-conditional and outside the graph schema. (Could partially catch via "evidence node exists but has no edges": that's AFC-0030 not 0003.)
- **AFC-0006 Outcome misinterpretation** ◐ — fits if the tool result has a *machine-readable polarity tag* (success/failure/value). Then invariant = the resulting `support`/`refute` edge polarity must match the tool's polarity. If tool returned failure but agent created `support` edge → violation. Without polarity tags, semantic.
- **AFC-0007 Causal misattribution** ✓ — *invariant*: GoS encodes causal links as typed edges. Misattribution = wrong `support`/`refute` edge target. Detectable *if* you have a separate oracle/golden graph to compare against. Without an oracle, you can only flag inconsistencies (e.g., `support` edge to a hypothesis that already has `refute` from same evidence) — that's a weaker check but still graph-local.
- **AFC-0011 Inefficient plan / wrong decomposition** ✗ — plan structure isn't natively in GoS. Could add `v_plan` nodes with `decompose` edges, but "inefficient" requires comparison to a reference plan, which is task-conditional.
- **AFC-0013 Planning–action disconnect** ◐ — fits if you add an `action` node type and require `∀ v_action . parent(v_action) ∈ {v_hyp_active, v_plan_active}` and `target(v_action) ∈ allowed_targets(parent)`. Catches "reasoning says A, action does B" as a graph property. Cross-step temporal inconsistency (FM-2.6 second shape) needs LLM.
- **AFC-0016 Task specification deviation** ✗ — requires comparing the *meaning* of final answer to task statement.
- **AFC-0020 Ambiguous request to peer** ✗ — peer interaction semantics (sender intent vs receiver interpretation) not capturable in graph alone.
- **AFC-0024 Incorrect verification** ✗ — needs ground truth oracle.
- **AFC-0029 Tool-output override** ✓ — invariant: `∀ v_evi with refute(v_evi → h) . ¬∃ subsequent reasoning span asserting support(v_evi → h)`. The agent contradicting a refute-evidence shows up as polarity mismatch in the local subgraph. Same backbone as AFC-0006.
- **AFC-0030 Critical-evidence miss** ✓ — invariant: `∀ v_evi ∈ V . deg_out(v_evi) ≥ 1`. Evidence nodes registered in the observation context but unconnected to any hypothesis are unattended. Pure graph property.
- **AFC-0036 Stated-reasoning unfaithfulness** ✗ — requires PromptExp-style attribution; the violation is between *stated reasoning* and *actual decision factors*, neither of which reduces to a graph property.

**T4 fit: 4/12 clean, 2/12 partial, 6/12 don't fit.**

## Tally

| Category | Count | Cards |
|---|---|---|
| Clean FSM/graph invariant | **10** | 0002, 0008, 0012, 0025, 0028, 0037 (T3) + 0007, 0029, 0030 (T4) + 0026 (T2 already) |
| Partial — needs schema extension | **5** | 0005, 0009, 0006, 0013 + 0011 (with `v_plan` extension) |
| Genuinely semantic, doesn't reduce | **6** | 0001, 0003, 0016, 0020, 0024, 0036 |

So if we adopt the GoS abstraction:

- **8 T3 cards collapse from "rule + LLM" to "rule on graph"** (no LLM call needed once graph is constructed). 
- **4 T4 cards collapse to graph invariants** (0007 weakly, 0029, 0030 cleanly, plus 0026 which was T2 anyway).
- **6 T4 cards stay semantic** — these are the actual irreducible AgentLens workload.

## What it costs to get there

The trade is: you pay an LLM-as-judge call **once per turn to construct/update the graph** (parse tool output → register evidence node, parse reasoning → create support/refute edge), and then **all subsequent invariant checks are free** (graph property checks in microseconds).

This is a much better economic shape than the naïve "judge each step against each card" loop:

| Approach | LLM calls per trajectory of T turns | Detectors covered |
|---|---|---|
| Naïve LLM-judge per card per step | T × 37 ≈ 100s-1000s | all |
| **GoS-style: build graph once, check invariants** | T (one per turn) | 16 cards via invariants + 6 via judge = 22 / 37 |
| Hybrid: layered T1 → T2 → graph → judge | T1+T2 free, then ~T graph LLM calls + ~5-10 judge calls | all 37 |

The GoS-style middle row is the right baseline for the 22 cards it covers; the harness still needs an LLM-judge fallback for the 6 T4 stragglers (memory over-simplification, retrieval relevance, task deviation, ambiguous peer request, incorrect verification, unfaithful CoT).

## Concrete schema extension this implies for cards

Reasonable proposal: add an optional **`fsm_invariant`** field to each
card whose check reduces to a graph property. Format:

```yaml
fsm_invariant:
  state_repr: "GoS"        # which abstraction the invariant assumes
  predicate: "∀ v_evi ∈ V . provenance(v_evi) ≠ ∅"
  check_kind: graph_property  # | fsm_transition | dual_threshold
  source_paper: "GoS arXiv 2603.21250 §Methodology"
```

Cards that don't fit (the genuine semantic 6) get no `fsm_invariant`
field — the absence itself signals "must use LLM-judge".

Then a single graph-aware detector can scan all `fsm_invariant`-bearing
cards in one pass per turn instead of per-card prompts.

## Implications for AgentLens

1. **The Sibling Agent's first job is graph construction, not failure detection.** Once it maintains B = (G, S) faithfully, ~60% of the failure surface (22/37 cards) becomes invariant-checkable without further LLM calls.
2. **Graph-construction LLM calls are reusable**: same call services many invariants. This is what makes GoS economically attractive.
3. **The remaining 6 semantic cards are the LLM-judge budget**. Per-turn cost ≈ 1 graph-update LLM call + 0-6 judge calls (only when symptoms suggest something the graph didn't catch). Compared to the naïve "judge each card each step", this is roughly an order of magnitude cheaper.
4. **GoS's deficiency taxonomy validates the AFC set**: GoS independently identified 4 deficiencies (Evidence Fabrication, Context Drift, Failed Backtracking, Early Stopping); each maps to a specific AFC. This is convergent design evidence — two papers from different angles arrived at overlapping failure axes.
5. **Schema extensions to consider** (analogous to GoS's typed nodes):
   - `v_constraint` node type → enables AFC-0009 (constraint ignorance) as an invariant.
   - `v_action` / `v_plan` node types → enables AFC-0011 + AFC-0013 partial coverage.
   - Tool-result polarity tags (success/failure/numeric value) → upgrades AFC-0006, AFC-0029 from partial to clean.
   These extensions are non-trivial but each unlocks ~1-2 more cards from the LLM-judge bucket into the graph-invariant bucket.

## Caveats

- Building a *faithful* GoS-style graph from arbitrary multi-agent trajectories is itself a nontrivial LLM task; GoS handles it well in their two abductive-reasoning domains (medical, microservice failure diagnosis) but generality to e.g. ALFWorld or WebShop trajectories is unproven. The graph-construction step's *fidelity* is the load-bearing assumption — if it's noisy, all invariants downstream are unreliable.
- The mapping above assumes single-trajectory analysis. Cross-trajectory failure patterns (e.g., AFC-0011 inefficient plan = "wasteful vs minimal expert plan") require comparing to reference graphs, which is a separate abstraction.
- GoS targets *abductive* reasoning specifically. For purely deductive or open-ended exploration, the state-machine drill-down/backtracking semantics may not map; the causal-graph half still does.

## See also

- `DETECTION-FEASIBILITY.md` — the rule-vs-semantic 4-tier classification this doc refines.
- `../FAILURE-PATTERNS-SYNTHESIS.md` §4 — implications for AgentLens harness design.
- GoS paper — `~/.cache/nanochat/knowledge/2603.21250/sections/Methodology.tex` for the formal definitions of (G, S, h\*, drill-down, backtracking).
