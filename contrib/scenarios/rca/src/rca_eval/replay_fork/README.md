# replay-fork

Counterfactual harness experiment for the RCA scenario: re-run the
cognitive-audit harness (extractor + auditor) over a **recorded** baseline
trajectory with a chosen harness model, fork the main agent wherever the
auditor surfaces a reminder, and judge whether the intervention raises the
success rate — without ever re-running the main agent for the control arm.

## Why

The live A/B (baseline vs `harness.sync`) re-runs the whole agent for both
arms. That is expensive and conflates "did the reminder help?" with main-
agent nondeterminism. This tool fixes the control arm to the *recorded*
baseline and only pays for the intervention continuations, so the delta is
attributable to the reminders. It also lets the harness (extractor +
auditor) run on a different, cheaper model than the agent.

## Design

```
CaseSource (Protocol)  --->  ReplayCase {case_id, system_prompt,
  EvalDbCaseSource              backbone_messages, data_dir,
  (eval.db schema lives          control_response, control_correct}
   here, nowhere else)
        |
        v
ReplayForkDriver  --- harness_provider (glm-5.1) --> fork-tree engine
        |            --- agent (Doubao) ----------> fork continuations
        v
LeafJudge (Protocol) ---> JudgeOutcome {correct == exact_match}
  RcabenchJudge            (GT read from data_dir/injection.json)
        |
        v
ResultSink (Protocol) ---> JsonlResultSink
```

The driver depends only on `ReplayCase` + the protocols. Swapping where
baselines are stored is a new `CaseSource`; a different success metric is a
new `LeafJudge`; a different output is a new `ResultSink`. The control arm
is the recorded backbone (served by the factory's root call without an
agent run); only forks invoke the agent.

`max_surfaces_per_node=1` makes the fork-tree a greedy spine — inject on the
first surface, continue, re-audit, repeat up to `--max-depth` — the offline
analogue of the live "inject every N turns" behaviour. The deepest
continuation's submission is judged; when the auditor never fires there is
no intervention and the case keeps its control outcome.

## Environment setup

Use the **published rcabench-platform wheel** (what `uv sync` installs).
Both the agent (`AgentRCAOutput`) and the judge (`evaluate_v2`,
`EvaluationResultV2.exact_match`, matching the baseline `correct`) come from
the wheel's `evaluation.v2` package, so agent and judge stay on one
rcabench. Do **not** install the rcabench source repo editable: its `main`
has flattened `evaluation.v2` into `evaluation` (renamed `evaluate`, dropped
`v2`), which breaks `agent.py`'s `from ...evaluation.v2 import AgentRCAOutput`.

```bash
set -a; . ./.env; set +a   # agent (Doubao) endpoint from OPENAI_* / .env
uv run python -m rca_eval.replay_fork.cli run ...
```

The agent (continuation) model resolves through the ambient `OPENAI_*` env
(same as the rca eval driver), so `.env` must point at the endpoint the
recorded baseline used. The harness model is a `~/.agentm/config.toml`
profile whose endpoint/key travel in the profile — independent of `.env`.

## Usage

```bash
uv run --no-sync python -m rca_eval.replay_fork.cli run \
    --source-exp agentm-ab100-baseline-0525-0847 \
    --db /home/ddq/AoyangSpace/AgentM/eval.db \
    --harness-model ark-glm51 \
    --agent-model Doubao-Seed-2.0-pro --agent-provider openai \
    --scenario rca:baseline \
    --max-depth 3 --max-concurrency 12 \
    --out runs/glm51-replay/results.jsonl \
    --sidecar-dir runs/glm51-replay/sidecars
```

## Output

`results.jsonl` — one `ReplayCaseResult` per line:

| field | meaning |
|---|---|
| `case_id` | dataset batch id |
| `fired` | did the harness surface any reminder? |
| `n_interventions` | depth of the intervention spine (0 if silent) |
| `control_correct` | recorded baseline correctness |
| `intervene_correct` | judged leaf correctness (= control when silent) |
| `intervention_path` | reminder texts applied, root → leaf |
| `judge_detail` | precision / recall / f1 / service_exact_match / any_*_hit |
| `error` | per-case failure, if any (batch continues) |

The printed summary reports control vs intervene success counts and the
W→R (helped) / R→W (harmed) flip counts. Per-case fork-tree replay sidecars
(`<case_id>.chained.jsonl`) land under `--sidecar-dir` for trace inspection.
