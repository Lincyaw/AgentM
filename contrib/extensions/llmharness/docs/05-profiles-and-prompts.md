# Profiles and prompts

Two independent knobs let you A/B the extractor and auditor children
without touching code:

| Knob | Picks | Where it lives |
|---|---|---|
| **profile** | tool set (which `register_tool`s the child mounts) | `audit/<phase>/profiles.py` |
| **prompt** | framing text | `audit/<phase>/prompts/*.md` |

The two are intentionally orthogonal — you can pair the `minimal`
tool profile with the `full` framing (e.g. to A/B whether the longer
framing helps even without drill-down), or vice versa.

---

## 1. Adapter config

All four knobs are read at adapter install time:

```bash
agentm --extension llmharness.adapters.agentm \
  --extension-config llmharness.adapters.agentm='{
    "extractor_profile": "minimal",
    "extractor_prompt":  "default",
    "auditor_profile":   "minimal",
    "auditor_prompt":    "minimal"
  }' ...
```

| Key | Default | Effect |
|---|---|---|
| `extractor_profile` | `"minimal"` | Resolves to a tool tuple from `audit/extractor/profiles.py::PROFILES`. |
| `extractor_tools` | `null` | Explicit list overriding the profile. `submit_events` is force-included. |
| `extractor_prompt` | `"default"` | Named variant (file under `audit/extractor/prompts/`) or absolute path. |
| `auditor_profile` | `"minimal"` | Resolves to a tool tuple from `audit/auditor/profiles.py::PROFILES`. |
| `auditor_tools` | `null` | Explicit list overriding the profile. `submit_verdict` is force-included. |
| `auditor_prompt` | `"minimal"` | Named variant (file under `audit/auditor/prompts/`) or absolute path. |

The legacy `prompt_override_extractor` / `prompt_override_auditor`
keys still work and trump the named lookup — pass raw text when you
don't want the file dance.

---

## 2. Built-in profiles

### Extractor

| Profile | Tools |
|---|---|
| `minimal` (default) | `submit_events` |

There is only one extractor profile today. The knob exists for
symmetry and for future A/B (e.g. an incremental
`register_event` + `add_edge` profile for very small models).

### Auditor

| Profile | Tools |
|---|---|
| `minimal` (default) | `submit_verdict` |
| `with_drill_down` | `submit_verdict`, `get_event_detail`, `get_turn` |

`minimal` is the recommended target for small-model SFT — the
student sees one tool, calls it once. `with_drill_down` is the upper
bound for larger teacher models.

---

## 3. Built-in prompts

### Extractor (`audit/extractor/prompts/`)

| Name | When to use |
|---|---|
| `default` | Production prompt; covers EventKind classification, witness rules, procedure. |

### Auditor (`audit/auditor/prompts/`)

| Name | When to use |
|---|---|
| `minimal` (default) | Pairs with the `minimal` profile. No drill-down references. Slimmer framing for small models. |
| `full` | Pairs with `with_drill_down`. Mentions `get_event_detail` / `get_turn`. Original framing with all four lenses. |

Recommended pairings:

| Use case | profile | prompt |
|---|---|---|
| Small-model SFT (~4B) | `minimal` | `minimal` |
| Large teacher / A/B upper bound | `with_drill_down` | `full` |
| Ablation: longer framing without drill-down | `minimal` | `full` |

The third row is exactly the kind of thing the orthogonal knobs are
for.

---

## 4. Adding a new variant

### New tool profile

Edit `audit/auditor/profiles.py` (or the extractor sibling) and add
an entry to `PROFILES`:

```python
PROFILES["just_verdict_and_turn"] = (
    TOOL_SUBMIT_VERDICT,
    TOOL_GET_TURN,
)
```

That's it. The resolver picks it up by name. No other code changes.

### New prompt variant

Drop a markdown file under `audit/auditor/prompts/` (or
`audit/extractor/prompts/`) named `<phase>_<variant>.md` — e.g.
`auditor_terse.md`. Reference it as:

```bash
--extension-config llmharness.adapters.agentm='{"auditor_prompt": "terse"}'
```

Or use an absolute path:

```bash
--extension-config llmharness.adapters.agentm='{"auditor_prompt": "/abs/path/to/my_prompt.md"}'
```

The loader resolves `<name>.md` first, then `<phase>_<name>.md`.

---

## 5. Replay records carry the resolution

The replay sidecar's `compose_kwargs.base_prompt` holds the resolved
framing text, and `compose_kwargs.tools` holds the resolved tool
tuple. `llmharness-replay` rebuilds the exact child surface from the
record — so an A/B run today and a replay tomorrow stay byte-aligned.

---

## 6. SFT export

`llmharness-distill export` packages the auditor side's framing into
the SFT JSONL's `input.system` field. Today the exporter uses the
package's eager-loaded `AUDITOR_SYSTEM_PROMPT` (the `minimal`
variant) so training framing matches the default deployment surface.
If you train against a different framing, override the exporter or
patch the constant — keeping training/inference framing aligned is
load-bearing for D3.
