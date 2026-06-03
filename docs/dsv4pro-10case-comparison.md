# Cognitive-Audit Harness: Three-Model Comparison (10 cases)

Date: 2026-05-30

## Experiment Config

| Run | Auditor model | Agent model | Extractor | Dataset |
|---|---|---|---|---|
| doubao-noext | Doubao-Seed-2.0-pro | Doubao-Seed-2.0-pro | skip | ops-lite-fixed-100 (10 subset) |
| glm51 | GLM-5.1 | Doubao-Seed-2.0-pro | skip | ops-lite-fixed-100 (10 subset) |
| dsv4pro-noext | DeepSeek-V4-pro | DeepSeek-V4-pro | skip | ops-lite-fixed-100 (10 subset) |

Note: glm51 uses Doubao as the continuation agent (only the auditor is GLM-5.1).
dsv4pro-noext uses DSv4pro for both auditor AND continuation agent.

## Summary

| Model | fired | ctrl_exact | iv_exact | helped | harmed | delta | any_svc | all_svc | avg_prec | avg_recall |
|---|---|---|---|---|---|---|---|---|---|---|
| doubao-noext | 10/10 | 2 | 1 | 0 | 1 | **-1** | 9/10 | 2/10 | 0.733 | 0.550 |
| glm51 | 9/10 | 2 | 2 | 1 | 1 | **+0** | 10/10 | 2/10 | 0.950 | 0.600 |
| dsv4pro-noext | 7/10 | 2 | 3 | 1 | 0 | **+1** | 10/10 | 3/10 | 1.000 | 0.650 |

## Per-Case Detail

| Case | GT (2 services) | Model | fired | n_iv | depth | ctrl | iv | prec | recall | answer |
|---|---|---|---|---|---|---|---|---|---|---|
| KQHDBB5 | geo, profile | doubao | T | 3 | 3 | T | T | 1.00 | 1.00 | geo, profile |
| | | glm51 | T | 1 | 1 | T | T | 1.00 | 1.00 | geo, profile |
| | | dsv4pro | **F** | 0 | 0 | T | T | 1.00 | 1.00 | geo, profile |
| KQHDBBF | profile, recommendation | doubao | T | 1 | 1 | T | **F HARMED** | 1.00 | 0.50 | recommendation |
| | | glm51 | T | 1 | 1 | T | **F HARMED** | 1.00 | 0.50 | recommendation |
| | | dsv4pro | T | 1 | 1 | T | **T** | 1.00 | 1.00 | profile, recommendation |
| KQHREG | profile, reservation | doubao | T | 1 | 1 | F | F | 1.00 | 0.50 | reservation |
| | | glm51 | T | 2 | 2 | F | F | 1.00 | 0.50 | reservation |
| | | dsv4pro | **F** | 0 | 0 | F | F | 1.00 | 0.50 | reservation |
| KQJ23R | ts-admin-order, ts-order-other | doubao | T | 1 | 1 | F | F | 1.00 | 0.50 | ts-order-other |
| | | glm51 | F | 0 | 0 | F | F | 1.00 | 0.50 | ts-order-other |
| | | dsv4pro | T | 1 | 1 | F | F | 1.00 | 0.50 | ts-order-other |
| KQJ24C | ts-route-plan, ts-route | doubao | T | 3 | 3 | F | F | 0.50 | 0.50 | ts-route, ts-admin-route |
| | | glm51 | T | 3 | 3 | F | F | 1.00 | 0.50 | ts-route |
| | | dsv4pro | T | 2 | 2 | F | F | 1.00 | 0.50 | ts-route |
| KQJ5SN | geo, search | doubao | T | 2 | 2 | F | F | 1.00 | 0.50 | search |
| | | glm51 | T | 3 | 3 | F | F | 1.00 | 0.50 | search |
| | | dsv4pro | T | 2 | 2 | F | F | 1.00 | 0.50 | search |
| KQJDR3 | ts-consign-price, ts-consign | doubao | T | 1 | 1 | F | F | 0.17 | 0.50 | 6 services (5 FP) |
| | | glm51 | T | 1 | 1 | F | F | 1.00 | 0.50 | ts-consign-price |
| | | dsv4pro | T | 2 | 2 | F | F | 1.00 | 0.50 | ts-consign-price |
| KQJFKX | ts-food, ts-train-food | doubao | T | 3 | 3 | F | F | 0.00 | 0.00 | ts-route (wrong) |
| | | glm51 | T | 3 | 3 | F | **T HELPED** | 1.00 | 1.00 | ts-food, ts-train-food |
| | | dsv4pro | T | 1 | 1 | F | **T HELPED** | 1.00 | 1.00 | ts-food, ts-train-food |
| KQJKBS | ts-food, ts-train-food | doubao | T | 3 | 3 | F | F | 0.67 | 1.00 | ts-food, ts-route, ts-train-food |
| | | glm51 | T | 3 | 3 | F | F | 0.50 | 0.50 | ts-food, ts-route |
| | | dsv4pro | T | 1 | 1 | F | F | 1.00 | 0.50 | ts-train-food |
| KQJNRD | rate, search | doubao | T | 2 | 2 | F | F | 1.00 | 0.50 | search |
| | | glm51 | T | 3 | 3 | F | F | 1.00 | 0.50 | search |
| | | dsv4pro | **F** | 0 | 0 | F | F | 1.00 | 0.50 | search |

## Fork Tree Detail (DSv4pro depth > 1 cases)

| Case | Node | Depth | Fork@turn | Continuation turns | Input tokens | Duration |
|---|---|---|---|---|---|---|
| KQJ24C | n0 | 0 | - | 43 (control) | - | - |
| | n1 | 1 | 30 | 59 | 1.5M | 18 min |
| | n2 (leaf) | 2 | 151 | 2 | 82K | 2 min |
| KQJ5SN | n0 | 0 | - | 45 (control) | - | - |
| | n1 | 1 | 30 | 59 | 1.9M | 20 min |
| | n2 (leaf) | 2 | 51 | 25 | 696K | 12 min |
| KQJDR3 | n0 | 0 | - | 45 (control) | - | - |
| | n1 | 1 | 20 | 60 | 2.2M | 22 min |
| | n2 (leaf) | 2 | 101 | 27 | 1.6M | 13 min |

## Key Findings

1. **DSv4pro is the only model with positive delta (+1)**: helped=1, harmed=0.
   Doubao delta=-1 (harmed=1), GLM delta=0 (helped=1 but also harmed=1).

2. **DSv4pro precision = 1.000**: zero false positives across all 10 cases.
   Doubao avg_prec=0.733 (diluted by FPs in KQJDR3 and KQJKBS).

3. **DSv4pro fires selectively (7/10)**: correctly does not intervene when the
   agent is already correct (KQHDBB5) or when the trajectory lacks actionable
   gaps (KQHREG, KQJNRD). Doubao fires 10/10 indiscriminately.

4. **Contradiction_flag is the effective reminder type**: DSv4pro's auditor
   says "your data contradicts your conclusion" rather than generic "you haven't
   checked X". This is the only reminder pattern correlated with helped outcomes
   (both KQHDBBF and KQJFKX).

5. **3 cases hit depth=2 forks**: the n1 continuation runs 59-60 turns before
   the second auditor fire. KQJ24C's n2 leaf is only 2 turns because the agent
   had already exhausted its investigation in n1.

6. **DSv4pro as agent is also a factor**: the dsv4pro-noext run uses DSv4pro for
   both auditor AND agent, while glm51 uses GLM-5.1 auditor + Doubao agent.
   DSv4pro agent never adds false positives (precision=1.000) which could be
   agent capability, not just auditor quality.

## Open Questions

- Separate auditor vs agent contribution: run auditor=DSv4pro + agent=Doubao to
  isolate whether the improvement comes from better auditor direction or better
  agent capability.
- Scale: these are 10 selected cases; a full 100-case run would give statistical
  confidence.
- Extractor value: dsv4pro-ext (with extractor) is running; compare whether the
  logic-flow graph improves auditor quality vs raw trajectory mode.
