# rca_single tuner

This is a tuner. Run with:

```bash
agentm --scenario rca_single/tuner --cwd <sandbox> "Run one tuning iteration."
```

What gets tuned: `contrib/scenarios/rca_single/prompts/investigator.md`
(kind=`system_prompt`). The eval suite that drives the signal lives at
`contrib/scenarios/rca_single/eval/` (3 tasks, programmatic grader).

For the recipe behind this layout see
`../../format_fix/tuner/README.md`. For the production scenario this
tuner targets see `../README.md` (or, lacking one, `../manifest.yaml`).
