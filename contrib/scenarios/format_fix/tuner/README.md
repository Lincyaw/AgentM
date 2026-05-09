# Adding a Tuner to Your Scenario

This directory is the worked example. To make any scenario self-evolving,
copy this layout and fill in the three pieces that are intrinsically
yours: **eval tasks**, **grader**, **tuner prompt**. Everything else is
generic mechanism.

> Read [`../../../.claude/designs/per-task-evolution-loop.md`](../../../../.claude/designs/per-task-evolution-loop.md)
> first if you haven't — this README is the practitioner's recipe; the
> design doc is the why.

## Mental model

A tuner is a normal scenario. It is **not** a supervisor of the
production scenario — it runs offline, in its own session, and reads
historical traces. What makes it a "tuner" is that it stacks three atoms
that let it (a) see prior runs, (b) re-run the production scenario in a
sandbox under proposed atom versions, (c) commit a winning version.

The relationship between tuner and target is one config field:
`target_scenario: <production_scenario>`. That's it.

## What the mechanism gives you for free

Once you declare `task_class` and stack the three evolution atoms, you
get without writing any code:

- `task_class` populated on `session.fingerprint.task_meta` (production + eval)
- Trace filtering by `(task_class, fingerprint)` via `tool_query_traces`
- Sub-session source overrides for hypothesis testing (no working-tree
  mutation, no git noise)
- Multi-sample variance + `2σ` noise-floor gate
- Holdout-aware eval reporting
- Tier-2 atom protection (auto-deferred to `pending_human_approval`)
- Constitution-protected `.agentm/decisions/<scenario>/activations.jsonl`
  audit log (with sibling `candidates/` pool for forward-compat)
- Cross-session activation: tuner writes the new atom file directly,
  git commits, next production session loads the new version on startup

## What you must define for a new scenario

These are the three load-bearing inputs. They cannot be auto-generated
without polluting the eval signal — they are the scenario's identity.

### 1. `eval/tasks/*.yaml` — your representative input distribution

Each YAML is one task:

```yaml
id: <unique-task-id>
task_class: <your_scenario>     # must match production manifest
description: <human-readable>
holdout: false                  # true → reported separately, not in primary score
input:
  user_message: <input passed to the production agent>
  fixtures: []                  # optional file paths the agent can read
expected:
  value: <whatever your grader compares against>
  # ...or whatever shape your grader needs
budget:
  max_turns: 25
  max_cost_usd: 0.50
```

Aim for 10–50 tasks. Mark 10–25% as `holdout: true` so the gate doesn't
overfit them.

### 2. `eval/grader.py` (preferred) or `eval/grader.md` — what counts as success

Two modes, pick one per scenario:

**Programmatic grader (`grader.py`)** — best when "correct" is decidable:

```python
def grade(task: dict, agent_output: str, trace: list[dict]) -> dict:
    """Return {score: float in [0,1], dimensions: {...}, rationale: str}."""
    expected = task["expected"]["value"]
    try:
        actual = json.loads(agent_output)
    except json.JSONDecodeError:
        return {"score": 0.0, "rationale": "not valid JSON"}
    return {
        "score": 1.0 if actual == expected else 0.0,
        "rationale": "deep-equal" if actual == expected else f"mismatch",
    }
```

Free of LLM noise. Use this whenever the answer admits a check.

**Rubric grader (`grader.md`)** — best when "correct" needs judgment (RCA,
plan quality, code review):

A prompt template the eval runner fills with `(task, agent_output, trace)`
and feeds to an LLM. Returns `{score, dimensions, rationale}` JSON. Note
this introduces grader noise — make sure your `samples_per_task` and
noise-floor gate compensate.

### 3. `tuner/prompt.md` — the quality signal for this task class

This is where "what counts as 'better' for my scenario" lives. PR #67's
deliberate choice was *not* to put this in a stats library — it lives in
the prompt because it scales with model capability and is per-scenario.

Template:

```markdown
You are the <SCENARIO> scenario tuner. Evolve the <ATOM> atom so the
production scenario passes more eval tasks. One atom per iteration;
never multi-atom mutations.

Quality signal:
  primary: <metric name from the grader>     # e.g. grade_mean
  guards : <metric>, <metric>                # e.g. tool_error_rate, cost_per_task

Loop:
  1. tool_query_traces(task_class="<SCENARIO>", n=20)
  2. Read the loaded <ATOM> source
  3. tool_eval_run(eval_dir=<...>) → baseline_id
  4. Design a focused improvement; write the full new source
  5. tool_eval_run(eval_dir=<...>, atom_source_overrides={"<ATOM>": <src>})
     → proposed_id
  6. tool_propose_change(target_atom="<ATOM>", new_source=<src>,
     rationale=..., eval_run_baseline=<...>, eval_run_proposed=<...>,
     decision="activate")

Stop condition: if 3 consecutive iterations cross no-improvement
threshold, conclude and exit.

Constraints:
  - MANIFEST.name must remain "<ATOM>"
  - No imports from agentm.harness.session or agentm.core._internal.*
  - <any other scenario-specific guard rails>
```

## The recipe

```bash
# 1. Copy this directory's parent as a starting point
cp -r contrib/scenarios/format_fix contrib/scenarios/<your_scenario>
cd contrib/scenarios/<your_scenario>

# 2. Edit manifest.yaml
#    - rename task_class
#    - keep observability atom in the extension list
#    - tighten the prompt: agent MUST use the atom under evolution,
#      otherwise eval signal will be polluted by LLM bypass
sed -i 's/format_fix/<your_scenario>/g' manifest.yaml

# 3. Replace the atom under evolution
#    - Either delete tool_normalize_json.py and add your own atom file,
#    - Or evolve a builtin atom (set target_atom to its name in tuner/manifest.yaml)
rm tool_normalize_json.py
# ... write your atom or point at an existing one

# 4. Replace eval/tasks/*.yaml with your tasks
#    - 10–50 tasks, 10–25% holdout
#    - input.user_message must trigger the atom under evolution
rm eval/tasks/*.yaml
# ... write yours

# 5. Replace eval/grader.py with your grader (or grader.md for rubric)
# ... edit

# 6. Edit tuner/manifest.yaml
#    - target_scenario: <your_scenario>
#    - eval_dir: contrib/scenarios/<your_scenario>/eval
#    - tune promotion thresholds if needed (defaults are usually fine)
sed -i 's/format_fix/<your_scenario>/g' tuner/manifest.yaml

# 7. Edit tuner/prompt.md
#    - target atom name(s)
#    - quality signal: primary metric, guards
#    - any scenario-specific constraints

# 8. Smoke test the production scenario first
agentm --scenario <your_scenario> --cwd /tmp/sandbox "<sample input>"
# → check .agentm/observability/*.jsonl shows task_class=<your_scenario>

# 9. Run one tuning iteration
agentm --scenario <your_scenario>/tuner --cwd /tmp/sandbox \
  "Run one tuning iteration on <your_scenario>."
# → check .agentm/decisions/<your_scenario>/activations.jsonl
```

## Key knobs in `tuner/manifest.yaml`

```yaml
extensions:
  - module: agentm.extensions.builtin.tool_eval_run
    config:
      target_scenario: <your_scenario>            # ← required
      eval_dir: contrib/scenarios/<your_scenario>/eval
  - module: agentm.extensions.builtin.tool_propose_change
    config:
      target_scenario: <your_scenario>            # ← required, same value
      promotion:
        threshold_relative: 0.05      # +5% on primary required to activate
        guard_tolerance: 0.10         # ±10% on guards
        stop_after_no_improvement: 3  # tuner-side honor system today
```

`threshold_relative` is the *minimum* improvement; the **noise-floor
check is independent and applied on top** (proposed - baseline must
exceed `2·sqrt(stderr_b² + stderr_p²)`). The noise check is the safety
net — even if you set `threshold_relative: 0.0`, an underpowered eval
won't activate.

## Gotchas (real, hit during format_fix bring-up)

**Production prompt must force the agent to depend on the atom.** A
capable LLM will happily produce correct output even if the tool returns
garbage, defeating the eval. Phrase the prompt so the tool's output IS
the agent's output ("reply with the tool's text verbatim"). If your atom
is one of many tools the agent uses, write tasks that *can only* be
solved by that atom.

**Observability atom must be in the production manifest.** Without it,
no `.jsonl` is written and `tool_query_traces` returns empty. Easy to
forget; check the trace directory after a smoke run.

**Tuner does not load the production atom.** That's intentional, and
`tool_propose_change` handles cross-session resolution via filesystem
walk. But it means `api.list_atoms()` in the tuner won't show the target
atom — don't rely on this in custom tuner logic.

**Sample correlation under deterministic graders.** With a deterministic
grader and a stable LLM, `samples_per_task > 1` doesn't divide stderr by
sqrt(N) because the samples aren't IID. Don't crank samples expecting
free statistical power — run more *tasks* instead.

**Multiple tuners on a shared atom.** If two tuners (e.g. for two
production scenarios) target the same atom file, ResourceWriter
serializes the writes but the second tuner's "baseline" already includes
the first's change. Today this race is observable in `activations.jsonl`;
no automatic resolution. Avoid it for now by giving each scenario its
own atom files.

## Promotion decisions: three channels

| `decision=` | Gate applies? | Use for |
|---|---|---|
| `activate` | yes (threshold + noise + guards) | normal "I have evidence this is better" |
| `exploratory` | no | eval is underpowered but you want production signal; flagged in record so production regression watching can be stricter |
| `rollback` | no (rollback is always safe) | re-activate the previous version after a regression |

All three write a decision record. Tier-2 atoms always defer to
`pending_human_approval` regardless of channel — security policy.
