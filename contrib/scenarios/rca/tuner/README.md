# rca:baseline tuner

This is a tuner. Run with:

```bash
agentm --scenario rca/tuner --cwd <sandbox> "Run one tuning iteration."
```

What gets tuned: `contrib/scenarios/rca/prompts/investigator.md`
(kind=`system_prompt`; shared prompts dir after the rca + rca:baseline merge
— only loaded by rca:baseline's manifest). The eval suite that drives the
signal lives at `contrib/scenarios/rca/eval/baseline/` (3 tasks,
programmatic grader).

For the recipe behind this layout see
`../../format_fix/tuner/README.md`. For the production scenario this
tuner targets see `../README.md` (or, lacking one, `../manifest.yaml`).
