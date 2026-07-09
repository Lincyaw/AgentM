# GRADE: Graph Representation of LLM Agent Dependency and Execution

- **arXiv:** 2606.22741 · **Author:** Yue Zhao (USC) · single-author preprint (ACL styling)
- **Code:** https://github.com/yzhao062/grade
- **One line:** Model *any* LLM-agent run as one typed graph with **two edge layers** — a free **execution** layer (what ran, in what order) and a costly **dependency** layer (what each step *relied on*) — and grade every dependency edge by *how it is known* (observed / declared / inferred). The dependency layer predicts failure where run size can't and transfers across agent classes; the execution layer localizes the faulting step. Off-the-shelf GNNs can't read the graph because they have no channel for the source grade.

---

## 1. The problem it names

> **LLM agents fail for what they relied on, but the record shows only what they did.**

A trace records which steps executed and in what order. It never records what each step *depended on* — the value it read, the resource it held, the earlier result it reused. That reliance is exactly where a growing share of agent failures live (stale read, overwritten resource, an input wrong from the start; cf. MAST). The canonical example (Fig. 1): a booking agent *reads* a price, *holds* a reservation, *confirms* it. The confirm is correct only while the price is current; if it goes stale between read and confirm, the run fails — yet every step still ran in order, so the failure appears **nowhere** in the execution trace.

Two failure modes, one per layer:
- **Coordination failures** live in the **execution** layer: wrong handoff, step out of order, an agent that never returns control. *Visible, recorded for free.*
- **Reliance failures** live in the **dependency** layer: a step consumed state that was wrong or had moved. *Invisible by construction* — no trace records it.

Second problem: there is **no shared way to draw agent runs**. Tool use, coding, web nav, multi-agent — each is drawn its own way (ReAct trace, edit sequence, orchestration graph), so analysis is rebuilt per class, each drawing re-centering execution and reproducing the blind spot.

## 2. The representation (§2)

A run is a directed, typed, temporal multigraph `G = (V, E_X, E_D, τ, t, σ)`:
- **4 node types:** `agent`, `decision`, `tool`, `resource` (resource = state a decision reads/writes, e.g. a DB row or source file; resource nodes version the state).
- **Execution edges `E_X`** — `emit` (agent emits its decision/tool nodes) and `handoff` (control passes step→step). Read deterministically off the trace: `E_X = f_exec(T)`. **Free.**
- **Dependency edges `E_D`** — `depends-on` / `reads` / `writes`, always time-respecting (`t(u) < t(v)`). **Not fixed by the trace.**
- **Source map `σ: E_D → {observed, declared, inferred}`** — the **attachment model**, the paper's key move. Per *edge*, not per run:
  - **observed** — the access already appears in the trace (a coding step edits a named file; a DB step reads a named row).
  - **declared** — added instrumentation logs read/write events the raw trace omits (or a contract naming which resources a tool may touch).
  - **inferred** — no access info exists; the edge is posited under a named assumption. Weakest is `A₀` = **full history**: every step depended on *every* earlier step.

**Central claim of the framing:** report not the dependency layer's accuracy in isolation but its **marginal lift** over the free execution layer:
```
Δ_dep = AUC(FLAT + dependency) − AUC(FLAT)
```
where **FLAT** = run-size counts (steps, tool calls, decisions, agents) — exactly what the execution layer hands over for free. The costly layer must "pay its own way" against run size. Δ_dep is *allowed* to be ~0 where size already separates failures.

**Prior representations are projections** (Table 1): ReAct/tool traces, MetaGPT, GPTSwarm, GoT reasoning graphs, computation graphs, data-lineage/PROV, agent-anomaly graphs — each keeps *one* layer and drops the other, and **none grades dependency edges by source**. GRADE = both layers + the grade → surfaces **both** failure modes.

## 3. The degenerate regime (§6 — the technical heart)

Under `A₀` (full history), the dependency layer **collapses to a function of run size**:
```
|E_D| = C(n,2) = n(n−1)/2 ,   depth(E_D) = n−1
```
Both edge count and longest chain are deterministic functions of `n`. A measured reliance edge and an `A₀` artifact carry the **same edge type**, so *topology alone cannot tell a real dependency from an invented one*.

- **Saturation ratio** `ρ = |E_D| / C(n,2) ∈ [0,1]` — a topology-only test computable *before any model is fit*. ρ→1 = full-history artifact (shape features are run size in disguise); ρ≪1 = the layer genuinely *selects* among possible reliances (shape carries real signal).
- Even away from the extreme, across the observed corpora edge count correlates with step count at |r| up to **0.96** — which is why every feature is normalized by run length.
- **Operational gate:** admit dependency-shape features where ρ is small; withhold them where ρ→1. Observed layer median ρ≈0.01; inferred layer ρ=1.000 by construction.

**Why generic GNNs misread it (4 modes):** (1) size-/permutation-blind pooling keys on run size (the full-history degree sequence is fixed by `n`); (2) **no channel for σ** — one adjacency, can't down-weight an inferred edge vs a declared one; (3) added capacity overfits corpus idiosyncrasy → transfers worse; (4) a joint embedding returns one score and **cannot decompose into the marginal-lift estimand** at all. GIN / R-GCN / HGT all confirm this empirically.

## 4. Evidence (6 observed-dependency corpora + Who&When for localization)

Corpora: **tau-bench, tau2-bench** (DB tool use), **SWE-agent, SWE-Gym, OpenHands** (software eng.), **AgentRewardBench** (web). Label 1 = failed run. Probe = standardized logistic regression, ROC-AUC, 5-fold CV × 5 seeds, seed-block 95% CIs.

**(a) Keystone — dependency helps where run size is weak (§4.1).** Order corpora by FLAT baseline:
| Corpus | FLAT | Δ_dep | verdict |
|---|---|---|---|
| tau-bench | 0.583 | **+0.031** | size-weak → lift |
| SWE-agent | 0.628 | **+0.085** | size-weak → lift |
| SWE-Gym | 0.663 | **+0.142** | size-weak → largest lift |
| OpenHands | 0.698 | −0.004 | size-strong → null |
| tau2-bench | 0.722 | −0.013 | size-strong → null |
| web | 0.765 | −0.007 | size-strong → null |

Two honesty checks: shape adds *beyond bare revisit density* (+0.038 / +0.036 / +0.064 on the 3 helpers); and where dependency looks redundant with size it still predicts failure alone (dep-only 0.694 on OpenHands, 0.755 on web) → overlap, not empty layer.

**(b) Transfer — pooled leave-one-corpus-out (§4.2) — "the asymmetry is the finding".** Train on 5 corpora, test on held-out 6th. `dep-only` (size-normalized) **clears chance on all 6** held-out classes and never inverts. **`flat` (run size) inverts on two**: 0.468 (tau-bench), 0.350 (SWE-Gym) — because "what counts as a long run" is class-specific, so a pooled size coefficient can flip. Every dep / flat+dep transfer AUC clears 0.5 at p<0.05 (one-sided Mann-Whitney U ≡ AUC). **Run size is a decent within-class predictor but is disqualified as the shared invariant; size-normalized dependency shape holds direction across classes.**

**(c) Observed vs inferred (§6.2, Table `gate`).** Rebuild each corpus's dependency layer under `A₀`, same features. Inferred features **follow run size into its inversions and add a third** — collapsing to 0.288 on web (where size transfers best at 0.766). Source-blind GIN is fooled identically (0.244 web, 0.332 SWE-Gym). The attachment grade is the *only* thing that differs between the columns.

**(d) GNNs lose in transfer (§6.3).** Within-corpus: features 0.716 vs GIN 0.696 / R-GCN 0.713 / HGT 0.696 (competitive, relation-aware ones sometimes win). **In transfer the ranking inverts:** features 0.642 mean, worst case 0.569; GIN 0.591; R-GCN 0.521 (0.409 SWE-Gym); HGT 0.523 (0.235 web). GIN swings ±0.19 across seeds. **Scale isn't the excuse** — GINs are *strong* on QM9/MUTAG (same small-typed scale); the difference is structural: molecular bonds carry a single observed grade and size doesn't encode the label.

**(e) Richer features cost portability (§6.4).** 4 lean features → 13 named features lifts within-corpus AUC on 4/6 but transfers *worse* on 4/6. **Abstraction, not capacity, carries the unification.**

**(f) Localization — the execution layer earns its keep (§7).** Who&When (126 failed multi-agent runs, step-level fault labels). Faults cluster (57% in first third, 54% at most-active agent), so the bar is an **early-fault position prior**, not chance. Ranking steps by agent centrality + handoff position:
| Model | top-1 | top-3 | MRR |
|---|---|---|---|
| Random floor | 0.119 | 0.346 | 0.324 |
| Position prior | 0.159 | 0.516 | 0.407 |
| **Execution structure** | **0.211** | **0.614** | **0.454** |

An auditor shown the top-3 steps finds the fault **3 in 5** vs ~1 in 2 for position. Who&When has *no* per-step read/write → its dependency layer is degenerate (position-equivalent by construction) → localization *must* come from the execution layer, and it does. (No corpus today has both observed dependencies *and* step-level fault labels, so localizing *reliance* faults awaits data.)

**(g) The declared grade on real data (Appx, FLORA-Bench).** ~6×10⁵ workflow-task pairs with explicit dependency DAGs (AFLOW programs / G-Designer planner). Declared layer is sparse (ρ median 0.071, far from ceiling). Dependency shape lifts AUC up to **+0.231** on AFLOW (workflow templates couple to failure) but ~0 on G-Designer (task-adaptive DAG whose shape doesn't track failure). Declared structure carries signal **when coupled to outcome**, not whenever an agent emits a graph.

## 5. Research directions the paper opens

The attachment model orders the agenda around **one lever: lightweight instrumentation that promotes inferred edges → observed/declared.**
1. **Observe the dependency layer** — instrument tools to log reads/writes (cheapest, highest payoff).
2. **Localize reliance faults** — mirror of §7, needs a corpus with observed deps + labeled faulting edge.
3. **Prune reliance no later step uses** — off-path work an efficiency pass targets (caution: static reachability tracks *density*, not waste — needs the observed grade).
4. **Repair / re-plan around fragile structure** (hub concentration, contended resources) — a control problem, left open.
5. **Act on a reliance before it fails** — an observed reliance can be re-checked before use.
6. **Fleet-scale structural anomaly detection** — flagged open (their unsupervised attempts recover run length in disguise → degenerate regime again).

---

## 6. Why this matters for **AgentM** (this repo)

AgentM is a pluggable agent framework whose whole observability spine — `agentm trace` (messages / turns / tools / spans / index across a parent + N-child session tree), plus the **auditor / verifier / RCA** line of work — is *exactly* the "execution layer" this paper says everyone records for free. GRADE is a sharp lens on what AgentM is already building and where the ceiling is.

Concrete connections and things worth trying:

- **AgentM traces = the execution layer, and only that.** `agentm trace tools --session <id>` joins tool calls + args + results in run order — the free `E_X`. What AgentM does **not** yet materialize is `E_D`: *which resource (file, DB row, prior tool result) a step relied on.* This paper is a direct argument for adding a **dependency layer** to the trace model. The `resource` node type (file / DB row / source file, versioned) maps cleanly onto AgentM's `FileOperations` / `BashOperations` / tool-result surface — reads and writes are *already flowing through atoms*, so many edges could be **observed** rather than inferred with modest work.

- **The "reasoning-auditor, not data-analyst" ceiling is this paper's thesis.** Memory `[[project_auditor_reasoning_not_data]]` records that the auditor's reminders are "well-grounded but structurally ceilinged." GRADE explains the ceiling structurally: an auditor reading only the execution trace is reading one of two layers, and **reliance failures are invisible in it by construction.** The lift comes from surfacing the dependency layer with an *observability grade*, not from better reasoning over the same execution log. This is a design argument for the auditor/verifier, not just a metric.

- **The attachment grade ≈ AgentM's "no preset enums / free-text + LLM-decided" instinct, but sharper.** `observed / declared / inferred` is a *3-value epistemic grade on how an edge is known* — not a subjective label vocabulary, so it doesn't conflict with `[[feedback_no_preset_subjective_labels]]`. It's the kind of principled, mechanism-grounded distinction the repo favors. Worth stamping on any dependency edge AgentM's verifier reconstructs.

- **Fault localization is directly reusable for RCA / Who&When-style work.** The `contrib/scenarios/rca/` line and the verifier (`[[project_verifier_finds_gt_label_bug]]`) do exactly step-level fault attribution. GRADE's result — **rank steps by agent centrality + handoff position, beat the early-fault *position prior* (not chance)** — is a cheap, execution-layer-only baseline the RCA harness should measure against. The "position prior, not random floor" methodology point is important: RCA metrics that only beat chance may be beating nothing.

- **Don't reach for a GNN.** If anyone proposes learning over AgentM's session graph with an off-the-shelf GNN, this paper is the counter-argument: without a channel for the source grade, message passing keys on run size and *inverts under transfer across scenarios*. AgentM spans many scenarios/agent classes (chatbot, coding, RCA, web) — precisely the transfer setting where size-blind models fail. **Lean, source-aware, size-normalized features win.**

- **`ρ` (saturation ratio) is a free diagnostic AgentM could compute per session.** Before trusting any dependency-structure signal, compute ρ = |E_D| / C(n,2). If ρ→1, the "structure" is run length in disguise. This is a topology-only guardrail that fits AgentM's "measure before optimizing / surface problems early" axioms.

- **Efficiency angle for workflows.** GRADE's "prune reliance no later step uses" maps onto AgentM's `workflow.py` orchestration (`agent()` / `parallel()` / `pipeline()`): an observed dependency graph over child sessions could mark off-path steps. Caveat from the paper: needs the *observed* grade — static reachability over inferred edges just tracks density.

**Highest-leverage experiment for AgentM:** add a minimal **observed dependency layer** to the trace store — for each tool step, log the resource ids it read and wrote (files via `FileOperations`, rows/keys via query tools) — then reproduce GRADE's keystone: does `FLAT + dependency` beat `FLAT` at predicting session failure on AgentM's own corpora (RCA, verifier, ARL runs)? The instrumentation is the lever; everything else is analysis over data the framework already routes through atoms.
