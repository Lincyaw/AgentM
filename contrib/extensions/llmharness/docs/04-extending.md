# Extending

Two extension points:

1. **Audit checks** — register pure graph checks that emit advisory
   `Finding`s into the auditor's prompt.
2. **Distill datasets** — adapt the distill pipeline to a dataset
   that is not the rca shape.

---

## 1. Registering an audit check

A check is a §11 single-file atom: one `MANIFEST` plus an
`install(api, config)`. No atom-to-atom imports, no
`core.runtime.*` import, no `core._internal` import. Mount it
**after** the adapter so the registry service is published when
the check resolves it.

Template:

```python
# my_pkg/check_foo.py
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from llmharness.audit.registry import (
    SERVICE_KEY,
    AuditCheckRegistry,
    CheckContext,
)
from llmharness.schema import EventKind, Finding


MANIFEST = ExtensionManifest(
    name="check_foo",
    description="Flag <whatever your invariant is>.",
    registers=(),
    config_schema={"type": "object", "additionalProperties": False},
    api_version=1,
    tier=1,
)


class _FooCheck:
    name = "foo"

    def __call__(self, ctx: CheckContext) -> list[Finding]:
        findings: list[Finding] = []
        # ctx exposes: events, edges, phases, recent_verdicts,
        # last_continuation_notes, turn_index. All frozen snapshots.
        for ev in ctx.events:
            if ev.kind is EventKind.HYP and not any(
                e.src == ev.id for e in ctx.edges
            ):
                findings.append(Finding(
                    category="open_hypothesis",
                    description=f"hypothesis event {ev.id} has no outgoing edge",
                    related_event_ids=(ev.id,),
                ))
        return findings


def install(api: ExtensionAPI, config: dict) -> None:
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, AuditCheckRegistry):
        raise RuntimeError(
            "audit registry service not published; "
            "mount llmharness.atom first"
        )
    registry.register_check(_FooCheck())
```

Mount:

```bash
agentm \
  --extension llmharness.atom \
  --extension my_pkg.check_foo
```

Rules:

* Checks are **pure functions** of `CheckContext`. No I/O, no
  network, no clock. Same context ⇒ same findings.
* Findings are **advisory**. The auditor LLM may ignore them.
  Phrase the `description` as a hypothesis, not a directive.
* `category` is free-text (no preset enum) — pick a short tag the
  downstream evaluator can group on.

Reference atoms ship under `llmharness.extensions.`:

| Atom | Flags |
|---|---|
| `check_repeated_actions` | ≥2 `act` events sharing an identical summary |
| `check_open_branches` | `dec` / `hyp` events with no outgoing data edge |
| `check_premature_conclusion` | `concl` events with <2 incoming edges |

---

## 2. Adapting the distill pipeline to a new dataset

The only dataset-specific module is `distill/gt.py`. Its
`load_dataset` indexes a JSONL keyed by `source` / `datapack_name`
and pulls a fixed set of fields into `GroundTruth`. To support a
new dataset shape, either:

### Option A — coerce upstream

Pre-process your dataset into the rca shape (`source`,
`ground_truth`, `fault_type`, `fault_category`) and the existing
loader works unchanged. Recommended when feasible — keeps the
distill code path uniform.

### Option B — add a loader

Add a sibling loader and a thin dispatch in `cli.py`:

```python
# distill/gt_my_dataset.py
from .gt import GroundTruth

def load_my_dataset(jsonl_path: Path) -> dict[str, GroundTruth]:
    out: dict[str, GroundTruth] = {}
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            sid = row["task_id"]               # whatever the id field is
            out[sid] = GroundTruth(
                sample_id=sid,
                root_causes=tuple(row.get("answers") or []),
                fault_type=row.get("category", ""),
                fault_category=row.get("super_category", ""),
                extra={"raw": row},
            )
    return out
```

Then wire it into the CLI with a `--dataset-format` flag.

What `GroundTruth` carries — and how the oracle prompt consumes
it — is intentionally flat (a few strings). If your dataset has
richer GT (e.g. an answer graph, expected reasoning chain), put
it in `extra` and extend `to_prompt_block()` to render it.

### Sample-id contract

Whatever loader you use, the keys returned by it MUST match what
the binding atom writes to `<sid>.meta.json`. The contract is:

```
LLMHARNESS_DISTILL_SAMPLE_ID  (env, set by your driver)
  ──▶ binding.install() writes meta sidecar with sample_id=<that>
  ──▶ load_dataset()[sample_id] resolves to a GroundTruth
```

If a session's `sample_id` is not in the dataset index, the
labeler logs a warning and skips that session — never silently
labels with empty GT.

---

## 3. Where NOT to extend

* **Don't add fields to `ReplayRecord`** for downstream use cases.
  The replay schema is intentionally agnostic to distillation,
  evaluation, etc. Use a sibling sidecar (the binding atom's
  `<sid>.meta.json` is the pattern).

* **Don't import `distill/*` from the adapter or audit children.**
  Distill is a pure offline consumer of the replay sidecar. The
  static dependency arrow only goes one way — see
  [01-architecture.md §4](01-architecture.md#4-static-dependency-graph).

* **Don't bake scenario logic into the auditor prompt.** Scenario-
  specific rules belong in audit checks; the auditor's job is
  general-purpose reasoning over the graph + findings.
