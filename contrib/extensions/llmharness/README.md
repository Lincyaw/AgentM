# llmharness

AgentM extension that supervises a running main agent: extract a typed
event-and-edge graph from each turn window, then run an auditor
periodically against that graph to surface advisory reminders.

The package is mounted onto a session via:

```bash
agentm --extension llmharness.adapters.agentm
```

`--extension` is repeatable; reference check atoms (see below) stack on
top.

## Layout

```
src/llmharness/
  schema.py                  Event, Edge, Finding, Verdict, Reminder
  adapters/agentm.py         orchestrator (TurnEnd → extractor → auditor)
  audit/
    extractor/               Phase 1: register_event / add_edge / submit_extraction
    auditor/                 Phase 2: submit_verdict + drill-down tools
    registry.py              AuditCheckRegistry (scenario-registered checks)
  extensions/                reference one-file check atoms (PR5)
```

## v3 flow (issue #134)

```
TurnEndEvent  ──▶  Phase 1 extractor child
                     register_event(...)   → Event
                     add_edge(...)         → Edge (witness-validated)
                     submit_extraction()   → terminate
                  └─▶  llmharness.audit_event / audit_edge entries

every k turns ──▶  Phase 2 auditor child
                     graph + Findings (from registered checks) + drill-down tools
                     submit_verdict(surface_reminder, reminder_text, ...)
                  └─▶  optional one-line reminder injected before next user prompt
```

The extractor is a graph builder: events are nodes, edges are first-
class records carrying witness fields (cited entities + verbatim
quote) checked at construction time. The auditor consumes the frozen
graph plus advisory `Finding` records produced by registered checks;
findings are signals, not directives — the auditor may ignore them.

## Registering a scenario check

Scenarios extend the auditor with their own pure graph checks. A check
is one file: a `MANIFEST` plus an `install(api, config)` that resolves
the registry service and registers a callable. Mount it after the
adapter:

```bash
agentm \
  --extension llmharness.adapters.agentm \
  --extension llmharness.extensions.check_repeated_actions
```

The reference atom (abbreviated):

```python
# src/llmharness/extensions/check_repeated_actions.py
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from ..audit.registry import SERVICE_KEY, AuditCheckRegistry, CheckContext
from ..schema import EventKind, Finding

MANIFEST = ExtensionManifest(name="check_repeated_actions", ..., tier=1)


class _RepeatedActionsCheck:
    name = "repeated_actions"

    def __call__(self, ctx: CheckContext) -> list[Finding]:
        ...  # pure: same context ⇒ same findings, no I/O


def install(api: ExtensionAPI, config: dict) -> None:
    registry = api.get_service(SERVICE_KEY)
    if not isinstance(registry, AuditCheckRegistry):
        raise RuntimeError(
            "audit registry service not published; "
            "mount llmharness.adapters.agentm first"
        )
    registry.register_check(_RepeatedActionsCheck())
```

Three reference atoms ship under `llmharness.extensions.`:

| Atom                            | Flags                                              |
|---------------------------------|----------------------------------------------------|
| `check_repeated_actions`        | ≥2 `act` events sharing an identical summary       |
| `check_open_branches`           | `dec` / `hyp` events with no outgoing data edge    |
| `check_premature_conclusion`    | `concl` events with <2 incoming edges              |

Downstream packages (e.g. rca-autorl) typically write their own one-
file atoms following the same shape and mount them via repeated
`--extension` flags.

## Schema stability

`schema.py` is the public contract for downstream consumers. Breaking
changes bump the package version in `pyproject.toml`. The v3 schema
break (issue #134) is documented in `schema.py`'s module docstring.
