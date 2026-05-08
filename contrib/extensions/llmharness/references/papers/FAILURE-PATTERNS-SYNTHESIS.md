# Cross-Paper Synthesis: Failure Patterns in LLM (Multi-)Agent Systems

Compiled 2026-05-07 from per-paper extractions by 10 parallel reading agents.
Source PDFs in this same directory; per-paper detailed notes are in each
agent's transcript and quoted below in §3.

---

## 1. One-glance comparison

Columns:
- **Taxonomy?** does the paper define a named failure-mode list?
- **# Patterns** total distinct named patterns (— if none).
- **Source of taxonomy** own / inherited.
- **Mechanism framing** how patterns are believed to *form*.
- **Detection** how the paper *finds* the pattern in a trajectory.
- **Sample size** how many failed trajectories the paper actually grounds claims on.

| # | Paper | Taxonomy? | # Patterns | Source | Mechanism framing | Detection | Failed-traj sample |
|---|---|---|---|---|---|---|---|
| 1 | **Who&When** (ICML'25) | No | — | — | Counterfactual-corrected step (no types) | LLM-as-judge: All-at-Once / Step-by-Step / Binary Search | 184 (127 MASs) |
| 2 | **AEGIS** (ICLR'26) | Yes | **14** | Inherited from MAST (Cemri+'25) | Synthesized via *prompt injection* + *response corruption* | SFT / GRPO-RL / DCL contrastive | 9,533 synth |
| 3 | **AgentDiagnose** (EMNLP'25 demo) | Yes (5 *competencies*, not modes) | **5** | Own (informed by web-agent lit.) | Low score on each axis = the corresponding failure | LLM-as-judge per axis (1-4 rubric) | 30 human-annotated for calibration; 46k filtered |
| 4 | **AgenTracer** | No (anti-taxonomy choice) | — | — (cites MAST as too coarse) | Counterfactual replay + programmed code-mutation injection | RL-trained Qwen3-8B with multi-granular reward (agent + Gaussian-step) | 2,476 (TracerTraj-2.5K) |
| 5 | **ELPO** | No | — | — | Operational *recoverability*: first irrecoverable step | Binary-search rollout trees (BEL) | tree-budget per traj |
| 6 | **EAGER** | Yes (per MAS) | 4-7 (varies by system) | Own, manual coding | Two scopes: intra-agent reasoning vs inter-agent orchestration | Reasoning-Scoped Contrastive Learning + step-wise embedding match | per-MAS small set (3 MASs studied) |
| 7 | **Watson** (ASE'25) | No (only 2 case-study modes) | — | — | "Latent reasoning errors / misaligned assumptions" from fast-thinking gap | Surrogate agent FIM/RepCoT + PromptExp attribution + judge | 2 cases + MMLU/SWE-bench |
| 8 | **A2P** | No | — | — | Pearl 3-level: ε (exogenous) abducted → do(a*) → predict | Single-prompt scaffolding (Abduct-Act-Predict) on top of all-at-once | 184 (Who&When) |
| 9 | **AgentDebug** | Yes ⭐ | **17** (5 modules) | Own, derived from 500+ failed traj | Single root-cause cascades through later modules | GPT-4.1 detector × stage1 fine-grained × stage2 counterfactual critical-step | 200 hand-annotated (AgentErrorBench) |
| 10 | **GraphTracer** | No (3 perturbation generators) | 3 (perturbations) | Own | Failures *propagate* over Information Dependency Graph (IDG) | IDG construction + impact ranking + RL-trained Qwen3-8B with graph rewards | 2,147 (GraphTraj-2.5K) |

⭐ = the most taxonomy-rich paper.

### 1a. Sub-pattern counts, in one place

For surveyability, here are all of the *named* patterns across the corpus.

**AEGIS / MAST (14)** — three umbrellas:
- **Specification (5)**: Task-spec deviation, Role-spec deviation, Add redundant steps, Remove conversation history, Remove termination conditions
- **Inter-Agent Misalignment (6)**: Repeat handled tasks, Make request ambiguous, Deviate from main goal, Withhold information, Ignore other agents, Inconsistent reasoning
- **Task Verification (3)**: Premature termination, Remove verification steps, Incorrect verification

**AgentDebug / AgentErrorTaxonomy (17)** — five modules:
- **Memory (3)**: Over-simplification, Hallucination (false memory), Retrieval failure
- **Reflection (4)**: Progress misassessment, Outcome misinterpretation, Causal misattribution, Hallucination
- **Planning (3)**: Constraint ignorance, Impossible action, Inefficient planning
- **Action (3)**: Planning–action disconnect, Format error, Parameter error
- **System (4)**: Step-limit exhaustion, Tool execution error, LLM limit, Environment error

**EAGER per MAS** (manual coding, % of failures):
- AutoGen-Code: Incorrect Code 48%, Decomposition Error 34%, Round Limitation 17%
- RCLAgent: Critical Trace Miss 53%, Metrics Query Error 42%, Round Limitation 5%
- SWE-Agent: Incorrect Code 46%, Localization Error 28%, Editing Error 26%

**AgentDiagnose competencies (5)**: backtracking & exploration, task decomposition, observation reading, self-verification, objective quality.

**GraphTracer perturbation generators (3)**: source pollution, conflict injection, edge removal.

---

## 2. Comparative analysis along the user's four axes

### 2.1 How many patterns?

The corpus splits into two camps:

- **Taxonomy camp** (AEGIS, AgentDebug, EAGER, AgentDiagnose) — invests in named buckets. Counts 5–17. AEGIS imports MAST so its 14 transitively shows up wherever MAST is cited. AgentDebug's 17 is the most domain-orthogonal because it organizes by *cognitive module* (memory/reflection/planning/action/system) rather than by symptom.
- **Position camp** (Who&When, AgenTracer, ELPO, A2P, GraphTracer; partly Watson) — refuses to categorize and asks only "which step / agent is the decisive cause?". Treats type as a free-text rationale. Their reasoning: typed labels are too coarse to drive credit assignment.

EAGER occupies a middle ground: per-system manual coding, no universal label set; argues the *distribution* over patterns is highly system-specific (see §1a percentages — almost zero overlap across the three MASs studied).

**Takeaway for AgentLens**: if a sibling-agent diagnosis system needs LLM-judgable categories, AgentDebug's 17-mode taxonomy is the most reusable starting point because it is module-aligned and not tied to any specific MAS framework. AEGIS/MAST's 14 modes are the most widely-used in the literature (Who&When, AgenTracer benchmarks all reference back to MAST). For finer-grained credit assignment, supplement with positional notions from AgenTracer/A2P/GraphTracer.

### 2.2 How do patterns *form*?

Three explanatory frames recur:

| Frame | Papers | Core claim |
|---|---|---|
| **Cascade / propagation** | AgentDebug, GraphTracer, EAGER | A single early error in module-X (memory/plan/source-node) corrupts everything that depends on it; final failure is downstream of root cause. AgentDebug names early-mid steps 5–15 as the propagation hot zone; GraphTracer formalizes propagation over an IDG; EAGER classifies whether cascade lives intra-agent (model-centric) or inter-agent (orchestration-centric). |
| **Recoverability gap** | ELPO, AgenTracer, A2P, Who&When | A step is a "real" failure iff a counterfactual correction would have flipped the outcome. ELPO searches recoverability via suffix sampling; AgenTracer/A2P apply oracle-guided rectification; Who&When formalizes (i*, t*) = earliest decisive pair. No semantic story about *why* the agent erred — only *where*. |
| **Misalignment between observable and latent** | Watson, AgentDiagnose | The agent's stated reasoning / surface trace doesn't match what actually drove the decision. Watson roots this in fast-thinking generation: the prompt-attribution-implied path diverges from the agent's self-explanation. AgentDiagnose's "self-verification" axis measures the same gap. |

Synthesis-derivable bridge: a cascading propagation framing **plus** a recoverability gate **gives** a clean operational definition: *the root cause is the earliest step whose correction breaks the propagation chain*. This is exactly AgentDebug's stage-2 critical-error definition and matches GraphTracer's IDG root-set definition.

### 2.3 What triggers each pattern?

Trigger conditions are reported at three granularities:

- **Domain** — failure-type distributions are sharply domain-conditioned. EAGER's three MASs share almost no patterns (code → Incorrect Code; RCA → Trace Miss; SWE-bench → Localization). AgentDebug also shows GAIA strongly skews to **Inefficient Planning** (18/50 traj), WebShop to **Memory Hallucination**, ALFWorld to **Impossible Action / Step Limit**.
- **Trajectory length** — long trajectories surface different failures than short. AgentDebug clusters errors at steps 5–15. ELPO empirically shows the first irrecoverable step gives much larger recovery-rate gaps than later ones (AIME 32.8% → 18.0% from 1st to 3rd). EAGER's "Round Limitation" pattern only appears when the round budget is the binding constraint.
- **Topology / architecture** — AgentDebug's Modular > Memory+ReAct > Reflection > ReAct > Act-Only ranking on ALFWorld (0.38 → 0.10) shows architecture choice changes failure distribution. AEGIS's per-framework injection success rate ranges 34% (DyLAN) → 76.5% (LLM-Debate), implying frameworks differ wildly in robustness.
- **Information topology** — GraphTracer's high-out-degree source-node weighting (P_perturb ∝ deg+(v)·I(deg-(v)=0)) says: errors at high-fanout source nodes cause maximal damage. The *graph* property, not the timestep, determines blast radius.

**Practical implication**: a sibling-agent diagnoser needs domain-conditional priors. Hard-coding "tool error" thresholds will miss the 50%+ of failures that are decomposition / planning failures in some systems but trace-retrieval / metrics-query in others.

### 2.4 How is each pattern detected?

Five detection paradigms appear, often combined:

| Paradigm | Used by | Strengths | Weaknesses |
|---|---|---|---|
| **LLM-as-judge over full trajectory** | Who&When (All-at-Once), A2P, AgentDiagnose, AgentDebug stage 1 | No training; works on any trajectory | Step-level acc 12-25% on Who&When without scaffolding. Long context breaks it. |
| **LLM-as-judge with counterfactual / causal scaffolding** | A2P (Abduct-Act-Predict), AgentDebug stage 2, GraphTracer counterfactual validate | Step-level lifts to 47%-70% on Who&When | Per-trajectory cost; relies on simulated counterfactual quality |
| **Recoverability search** | ELPO BEL (binary search), Who&When binary search | O(log T) anchor probes; principled | Needs ability to sample suffixes from arbitrary prefix |
| **Fine-tuned tracer model** (RL or SFT or contrastive) | AEGIS (SFT/GRPO/DCL), AgenTracer-8B, GraphTracer-8B | Fast at inference; can localize step + agent | Needs synthetic labeled data → injection pipelines (AEGIS prompt+response corruption; GraphTracer graph-aware perturbations) |
| **Embedding-based retrieval against past failures** | EAGER (Reasoning-Scoped Contrastive Learning), AgentDiagnose viz | Online; cheap once knowledge base seeded | Off-the-shelf embeddings perform poorly (EAGER reports BGE-M3 Recall@10 = 22% on paraphrased traces) — needs custom representation |
| **Surrogate-agent reasoning reconstruction** | Watson | Doesn't alter primary; works post-hoc on closed traces | Trends only (p > 0.05 on debugging downstream); cost = N parallel completions |

Reported step-level attribution accuracy on **Who&When** (the de-facto benchmark) lines up neatly with the paradigm:
- LLM-as-judge naive: ~14% (Who&When baselines)
- LLM-as-judge + causal scaffolding: 29-47% (A2P)
- RL-trained tracer: 21-50% (AgenTracer-8B, GraphTracer-8B)

So roughly: **scaffolded prompting ≈ trained tracer ≈ 3× naive prompting**. No method exceeds ~50% step-level accuracy on hand-crafted trajectories — a strong signal that the problem remains hard.

---

## 3. Per-paper one-paragraph summaries

### 3.1 Who&When (ICML'25)
*"Decisive error" is the earliest step whose counterfactual correction flips Z=1→0; failure-responsible pair (i*, t*) = (agent at decisive step, decisive step).* No taxonomy. 184 tasks across 127 MASs (CaptainAgent + Magnetic-One). Three LLM-as-judge baselines (All-at-Once / Step-by-Step / Binary Search) reach ≤53.5% agent-level / ≤14.2% step-level on GPT-4o; reasoning models (o1, R1) don't help meaningfully. **Use as benchmark, not as taxonomy source.**

### 3.2 AEGIS (ICLR'26)
*Imports MAST's 14 error modes (3 umbrellas: Specification, Inter-Agent Misalignment, Task Verification).* Synthesizes 9,533 trajectories via LLM-based Adaptive Manipulator that does either prompt injection or response corruption per Injection Plan. Three learning paradigms: SFT (best — Qwen2.5-14B → 26.51 avg), GRPO RL (18.41), small-encoder Disentangled Contrastive Learning (12.61, 23M params). Agent-level F1 > Error-level F1 universally. **Reusable as: ready-to-use 14-mode taxonomy + injection-recipe template.**

### 3.3 AgentDiagnose (EMNLP'25 demo)
*Five "competency" axes — backtracking, decomposition, observation reading, self-verification, objective quality.* LLM-as-judge with 1-4 rubric per axis; Pearson r with humans 0.39-0.78 (decomposition easiest, backtracking hardest). Four visualization views (t-SNE / word cloud / nav graph / trajectory). Filtering NNetNav-Live 46k → top-6k lifted WebArena +0.98%. **Reusable as: low-friction diagnostic axes for web-agent trajectories.**

### 3.4 AgenTracer
*Anti-taxonomy: only positional pair (i*, t*).* Builds TracerTraj-2.5K (2,476 traj) via (a) DeepSeek-R1 counterfactual replay on failed traj + (b) programmed code-mutation injection on successful traj. AgenTracer-8B (Qwen3-8B + multi-granular GRPO with Gaussian step kernel) beats Gemini-2.5-Pro by 18.18% on Who&When; gives 4.8-14.2% downstream improvement on MetaGPT/MaAS. **Reusable as: training pipeline for a small tracer model.**

### 3.5 ELPO (Error-Localized Policy Optimization)
*"First irrecoverable step" defined operationally via suffix sampling — a step where no continuation under fixed budget reaches the correct answer.* BEL = binary-search rollout tree → O(log K) probes; hierarchical (branch + trajectory) advantage attribution; error-localized adaptive clipping that strengthens corrective updates on the critical-step suffix only. ELPO-7B beats GRPO/DAPO/ToRL on TIR by +2.2% avg. **Reusable as: principled localization without LLM-judge.**

### 3.6 EAGER
*Per-MAS manual taxonomies — failures concentrate sharply per system: AutoGen-Code = Incorrect Code 48%; RCLAgent = Critical Trace Miss 53%; SWE-Agent = Incorrect Code 46% + Localization 28%.* Reasoning-Scoped Contrastive Learning encodes intra-agent + inter-agent semantics. Step-wise detection during execution + reflexive mitigation; expert+RCA loop after final-output failure updates fine and coarse failure-knowledge tiers. F1 73.6/86.2/79.9 anomaly detection across the three MASs. **Reusable as: empirical evidence that off-the-shelf embeddings fail (Recall@10 = 22% paraphrase retrieval) → need custom representation.**

### 3.7 Watson (ASE'25)
*"Latent reasoning errors / misaligned assumptions" — only two case-study patterns surfaced: wrong-file fixation, salient-keyword distraction.* Surrogate agent (same model + temp/top_p) replicates primary's output via FIM (preferred) or RepCoT (fallback), then PromptExp + LLM judge verify alignment between the surrogate's reasoning and the prompt components actually attributed. Self-consistency-style aggregation summarizes traces. WA-SR debugging gives MMLU/SWE-bench trends but p>0.05. **Reusable as: framing of "cognitive observability" + concrete decoupled surrogate recipe.**

### 3.8 A2P (Abduct, Act, Predict)
*No own taxonomy; consumes Who&When labels.* Maps Pearl's three-level causal hierarchy to a single LLM prompt: Abduction infers ε exogenous (knowledge gap, flawed assumption, query misinterpretation) → Action defines minimal do(a*_t) → Prediction simulates 3-5 turns. Step-level acc 47.46% Alg-Gen (2.85× over all-at-once 16.67%), 29.31% Hand-Crafted (2.43× over 12.07%). Ablations: removing Prediction is the largest hit (-12.07pp Hand-Crafted); contextual step-numbering alone explains ~30pp. **Reusable as: prompt scaffold to 3× naive LLM-judge.**

### 3.9 AgentDebug ⭐
*17 fine-grained sub-error-types in 5 modules (Memory/Reflection/Planning/Action/System), grounded in 500+ failed trajectories from ALFWorld+WebShop+GAIA.* AgentErrorBench: 200 hand-annotated trajectories, Cohen's κ=0.55. Per-domain dominance: GAIA → Inefficient Planning, WebShop → Memory Hallucination, ALFWorld → Impossible Action + Step Limit. AgentDebug = stage-1 per-step per-module detector (GPT-4.1) + stage-2 counterfactual critical-step + stage-3 up-to-N=5 re-rollouts. +24% all-correct over baselines on Who&When-style step attribution; up to +26% relative success via re-rollout. **The richest, most reusable taxonomy in the corpus.**

### 3.10 GraphTracer
*"Patterns" = three graph-aware perturbation generators, not symptom categories: source pollution (high-out-degree node), conflict injection (sibling-pair contradiction), edge removal (max edge-betweenness).* Information Dependency Graph G_τ = (V, E) with node = (timestep, agent, observation), edge = explicit citation. Pipeline: incremental IDG construction → backward traversal from failure node → impact ranking by α·deg+ + (1-α)·betweenness → counterfactual validate → forward greedy path reconstruction. GraphTracer-8B (Qwen3-8B + GRPO with format/source-node/path rewards) reaches step-level 49.97% / 28.63% on Who&When automated/handcraft — current best. **Reusable as: structural framing replacing temporal sequence; IDG schema is implementable on top of any trajectory log.**

---

## 4. Implications for AgentLens

Mapping survey findings back to the AgentLens design points (sibling agent + diagnosis harness):

1. **Pick a taxonomy backbone.** AgentDebug's 17-mode 5-module taxonomy is the most domain-orthogonal and cascade-aware. Use it as the default error label set; cross-reference MAST/AEGIS labels for traceability when consumed by external tooling. Don't try to invent yet another taxonomy.
2. **Combine positional + categorical attribution.** Pure typed classification has weak step-level accuracy; pure positional (Who&When-style) lacks actionable feedback. Output (i*, t*, error_type, evidence, suggested_fix) like AgentDebug's detector prompt — it enables both sibling-agent feedback and downstream re-rollout.
3. **Adopt the cascade hypothesis as the diagnostic prior.** When a failure surfaces at step T, walk back through dependencies (à la GraphTracer IDG; conceptually identical to AgentDebug's earliest-critical-step) before flagging the local symptom. Mid-trajectory steps 5-15 are the empirical hot zone.
4. **Don't trust off-the-shelf embeddings for trace retrieval** (EAGER §2.2). Either fine-tune a Reasoning-Scoped representation or fall back to LLM-as-judge with structured per-step prompts.
5. **Use counterfactual scaffolding, not only pattern matching.** A2P shows Abduct-Act-Predict gives 3× over all-at-once for free. Bake it into the sibling agent's diagnostic prompt template.
6. **For long traces, decouple coarse-vs-fine analysis.** EAGER and the AgentLens 2.4.2 sketch both want this: cheap step-wise embedding match → if matches → expensive counterfactual validate. The two-stage cost model is what makes online diagnosis affordable.
7. **Per-domain priors matter more than generic priors.** EAGER's per-MAS distributions and AgentDebug's per-domain heatmaps both confirm: failure distributions shift dramatically. The diagnosis harness needs to learn (or be configured with) a per-system prior over which modules fail.

---

## 5. References (PDF filenames in this directory)

```
01_who-and-when_2505.00212.pdf
02_aegis_2509.14295.pdf
03_agentdiagnose_emnlp2025-demos-15.pdf
04_agentracer_2509.03312.pdf
05_elpo_2602.09598.pdf
06_eager_2603.21522.pdf
07_watson_2411.03455.pdf
08_a2p_2509.10401.pdf
09_agentdebug_2509.25370.pdf
10_graphtracer_2510.10581v1.pdf
```

See `INDEX.md` for arXiv/venue links and one-line descriptions.
