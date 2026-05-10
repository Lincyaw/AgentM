# format_fix

**What this is**: a toy task class used as the worked example for the
per-task evolution loop. The production agent receives a malformed-but-
fixable JSON-ish string and must reply with the canonical JSON. The atom
under evolution is `tool_normalize_json` — its v1 implementation is
deliberately weak so the tuner has something real to evolve.

**Why it exists**: validates the [per-task evolution loop](../../../.claude/designs/per-task-evolution-loop.md)
end-to-end without coupling the validation to a heavyweight scenario.
Three properties make it the right lab:

1. **Deterministic grader.** `eval/grader.py` does `json.loads(actual) == expected`.
   No LLM in the eval signal — when a result is 0/1, that's the atom's
   doing, not grader noise.
2. **Cheap.** A single eval run is N tasks × N samples × ~2-turn
   production sessions. Burns small.
3. **Atom quality is the bottleneck.** The production prompt mandates
   verbatim tool output, so the eval score *cannot* be saved by a
   capable LLM bypassing the broken atom. If the atom is wrong, the
   answer is wrong.

If you just want to see the loop work, run the tuner here. If you want
to wire the loop onto your own scenario, copy this directory and read
[`tuner/README.md`](tuner/README.md).

## Layout

```
format_fix/
├── manifest.yaml          # production scenario; declares task_class: format_fix
├── tool_normalize_json.py # the atom under evolution (post-tuning: v2)
├── eval/
│   ├── tasks/*.yaml       # 8 representative tasks, 2 marked holdout
│   └── grader.py          # deterministic deep-equal grader
└── tuner/
    ├── manifest.yaml      # meta-scenario: stacks the 3 evolution atoms
    ├── prompt.md          # tuner system prompt (quality signal lives here)
    └── README.md          # how to clone this onto your own scenario
```

## Run the production scenario

```bash
agentm --scenario format_fix \
       --cwd /tmp/ff_sandbox \
       "{'count': 42, 'ratio': 3.14, 'active': True, 'empty': None}"
# → {"count": 42, "ratio": 3.14, "active": true, "empty": null}
```

The agent calls `normalize_json`, copies its text result verbatim, and
that becomes the final answer. Observability traces land under
`<cwd>/.agentm/observability/`.

## Run the tuner (one iteration)

```bash
agentm --scenario format_fix/tuner \
       --cwd /tmp/ff_sandbox \
       "Run one tuning iteration. Eval baseline, propose an improved
        tool_normalize_json (stdlib only), eval proposed, propose_change."
```

Decisions land in `<cwd>/.agentm/decisions/format_fix/activations.jsonl`
(append-only audit log). Activations write through to
`tool_normalize_json.py` and commit via git.

## What the eval set covers

| Task | Stresses | Holdout |
|---|---|---|
| 01_simple_keys | single-quoted scalars | no |
| 02_nested | nested objects | no |
| 03_unicode | non-ASCII characters | no |
| 04_numeric_types | Python `True`/`None` → JSON `true`/`null` | no |
| 05_array_of_objects | arrays | no |
| 06_trailing_comma | trailing comma before `}` | **yes** |
| 07_comments | JSONC `//` line comments | **yes** |
| 08_deeply_nested | depth | no |

Holdout tasks are **not** included in the primary score the promotion
gate looks at; their score is reported separately so you can spot
overfitting after the fact.

## What this scenario is NOT

- Not a benchmark — 8 tasks is too few to be one.
- Not a realistic JSON normalizer — `ast.literal_eval` is fine for the
  lab, not for the wild (no comment stripping, no error recovery).
- Not a stand-in for RCA / plan_mode / etc. — those need real eval
  suites and (likely) LLM-rubric graders. format_fix proves the
  *mechanism*; real scenarios prove that the mechanism is *enough*.
