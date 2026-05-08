# Failure Attribution & Trajectory Analysis — Paper Index

Background reading for the AgentLens / sibling-agent diagnosis line of work
(Section 2.4.2 in the project notes). Downloaded 2026-05-07.

## Core papers

| # | Short name | Title | Venue | arXiv | File |
|---|------------|-------|-------|-------|------|
| 1 | Who&When | Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems | ICML 2025 Spotlight | [2505.00212](https://arxiv.org/abs/2505.00212) | `01_who-and-when_2505.00212.pdf` |
| 2 | AEGIS | Aegis: Automated Error Generation and Attribution for Multi-Agent Systems | ICLR 2026 (under review) | [2509.14295](https://arxiv.org/abs/2509.14295) | `02_aegis_2509.14295.pdf` |

## Related work

| # | Short name | Title | Venue | Source | File |
|---|------------|-------|-------|--------|------|
| 3 | AgentDiagnose | AgentDiagnose: An Open Toolkit for Diagnosing LLM Agent Trajectories | EMNLP 2025 Demo | [ACL Anthology](https://aclanthology.org/2025.emnlp-demos.15/) | `03_agentdiagnose_emnlp2025-demos-15.pdf` |
| 4 | AgenTracer | AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems? | preprint | [2509.03312](https://arxiv.org/abs/2509.03312) | `04_agentracer_2509.03312.pdf` |
| 5 | ELPO | Learning from the Irrecoverable: Error-Localized Policy Optimization for Tool-Integrated LLM Reasoning | preprint | [2602.09598](https://arxiv.org/abs/2602.09598) | `05_elpo_2602.09598.pdf` |
| 6 | EAGER | Efficient Failure Management for Multi-Agent Systems with Reasoning Trace Representation | preprint | [2603.21522](https://arxiv.org/abs/2603.21522) | `06_eager_2603.21522.pdf` |
| 7 | Watson | Watson: A Cognitive Observability Framework for the Reasoning of LLM-Powered Agents | ASE 2025 | [2411.03455](https://arxiv.org/abs/2411.03455) | `07_watson_2411.03455.pdf` |
| 8 | A2P | Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems | preprint | [2509.10401](https://arxiv.org/abs/2509.10401) | `08_a2p_2509.10401.pdf` |
| 9 | AgentDebug | Where LLM Agents Fail and How They can Learn From Failures | preprint | [2509.25370](https://arxiv.org/abs/2509.25370) | `09_agentdebug_2509.25370.pdf` |
| 10 | GraphTracer | GraphTracer: Graph-Guided Failure Tracing in LLM Agents for Robust Multi-Turn Deep Search | preprint (v1) | [2510.10581v1](https://arxiv.org/abs/2510.10581v1) | `10_graphtracer_2510.10581v1.pdf` |

## One-line takeaway per paper

1. **Who&When** — defines automated failure attribution; releases 184-trajectory dataset over 127 MAS (CaptainAgent + Magnetic-One); best agent-level acc 53.5%, step-level 14.2%.
2. **AEGIS** — auto-generates 9,533 error trajectories via LLM-based context-aware fault injection; supports SFT / contrastive (DCL) / RL (GRPO) paradigms.
3. **AgentDiagnose** — five-axis competency scoring (backtracking, decomposition, observation reading, self-verification, objective quality) + visualization toolkit.
4. **AgenTracer** — counterfactual replay + programmed fault injection → TracerTraj-2.5K → AgenTracer-8B beats Gemini-2.5-Pro by 18.18% on Who&When.
5. **ELPO** — locates first-irrecoverable step via binary-search rollout trees, hierarchical advantage attribution, error-localized adaptive clipping (over GRPO).
6. **EAGER** — reasoning-trace representation captures intra-/inter-agent semantics; step-wise detection + reflexive mitigation + RCA-driven knowledge update.
7. **Watson** — surrogate-agent post-hoc reasoning reconstruction without altering the primary agent (cognitive observability).
8. **A2P** — Pearl's three-step causal scaffolding (Abduction → Action → Prediction) lifts step-level acc to 47.46% on Who&When (2.85× baseline).
9. **AgentDebug** — AgentErrorTaxonomy (memory/reflection/planning/action/system) + AgentErrorBench + iterative re-rollout from root cause.
10. **GraphTracer** — Information Dependency Graph replaces temporal sequence; graph-aware fault injection + RL with structural rewards.

## How this maps to AgentLens (Section 2.4.2)

- **Sibling-agent diagnosis** ←→ Watson (surrogate decoupled from primary).
- **Structured preprocessing / harness for trajectory** ←→ AgentDiagnose (visualization + competency scoring), GraphTracer (IDG over flat logs).
- **Coarse-to-fine multi-stage analysis** ←→ A2P (abduce → minimal intervention → counterfactual predict), ELPO (binary-search localization).
- **Knowledge accumulation from production failures** ←→ EAGER (RCA + per-agent / per-system failure knowledge), AgentDebug (taxonomy-driven re-rollout).
- **Datasets / benchmarks for evaluation** ←→ Who&When, AEGIS-9533, TracerTraj-2.5K, AgentErrorBench.
