# Handler Priority — Per-Channel Dispatch Order Declaration

Today every handler subscribed to a bus channel runs in registration order, and that order is determined by atom install order (alphabetical for auto-discovery, scenario-YAML order otherwise). For mutating channels the earlier handler runs first; for replacement channels (the `_collect_*` helpers in [agent-loop](agent-loop.md)) the **last non-None return wins**. This couples dispatch order to filename — a brittle invariant that broke once already (reload silently bumped atoms to the tail of every channel; fixed by [position preservation](../../src/agentm/harness/atom_reloader.py) but the underlying coupling stayed).

This document specifies a priority knob on `api.on(...)` so atoms declare their intended dispatch tier explicitly.

Related: [pluggable-architecture](pluggable-architecture.md) §3.5 (event bus), [agent-loop](agent-loop.md) (decide_turn_action resolution lattice), [self-modifiable-architecture](self-modifiable-architecture.md) §reload (handler ordering during reload).

## Goals

1. **Explicit ordering**. An atom that *needs* to run before another (e.g. a permission veto before a tool dispatcher; a compaction step before observability records the final messages) can say so without depending on filename luck.
2. **One rule covers mutation and replacement**. The same priority knob controls both "earlier sees state first" and "later overrides earlier's return".
3. **Backward compatible**. Existing atoms — every one in `extensions/builtin/` plus the contrib tree — keep working unchanged. They land in the default tier.
4. **No dependency graph**. No `depends_on` / `before` / `after` declarations. We pick numeric tiers, accept that two atoms in the same tier are still ordered by registration, and revisit constraints only if real demand surfaces.

## Non-Goals

- **No cross-channel priority**. Each `api.on(channel, handler, priority=...)` call is independent. An atom can be `PRE` on `tool_call` and `POST` on `tool_result`.
- **No system-wide priority numbers**. The mapping from symbolic names to numbers is internal. Scenarios should not hard-code raw integers (we expose the numeric door for unusual cases, but the convention is symbolic).
- **No handler-level disable**. Atom-level enable/disable already exists via install/unload. A handler that should sometimes silently no-op is a state-machine concern inside the handler, not the bus's problem.
- **No re-entrant emit ordering**. If a handler emits a new event during dispatch, that nested emit fires its own handlers in priority order, but the outer iteration is unaffected — the snapshot list captured at `emit()` entry is still what runs.

## Priority Tiers

Three symbolic tiers, mapped to integers. Lower number = runs earlier.

| Symbol | Numeric | Intended use |
|---|---|---|
| `BusPriority.PRE` | `100` | Security gates, validation, decisions that must happen *before* normal handlers see the event. Examples: a permission atom that vetoes a `tool_call` based on policy; a sanitizer that strips secrets out of a `before_send_to_llm` payload before any other handler can read it. |
| `BusPriority.NORMAL` | `500` | Default. Where business-logic atoms live. Examples: `system_prompt`, `prompt_templates`, `skill_loader`, every tool atom's input pre-processing. |
| `BusPriority.POST` | `900` | Audit, tracing, observability — handlers that should see the *final* state after PRE and NORMAL have run. Examples: `observability`, downstream loggers, side-effect notifiers. |

Atoms get `NORMAL` when they don't pass `priority`. Numeric values between the tiers are legal (e.g. `priority=300` to land between PRE and NORMAL) but discouraged in scenarios — the symbolic constants exist precisely so that a future re-numbering is invisible.

## API Surface

`ExtensionAPI.on` gains an optional keyword-only parameter:

```python
def on(
    self,
    channel: str,
    handler: Handler,
    *,
    priority: int = BusPriority.NORMAL,
) -> Unsubscribe: ...
```

Same Protocol overloads continue to apply (channel-typed Literal narrows the handler signature). Both kwarg and positional calls are accepted; existing call sites that pass only `(channel, handler)` get `NORMAL`.

The `BusPriority` constants live in `agentm.core.abi.events` next to `EventBus` — they are part of the kernel ABI, importable by atoms and the LLM provider alike.

```python
from agentm.core.abi import BusPriority

api.on(ToolCallEvent.CHANNEL, _veto_handler, priority=BusPriority.PRE)
api.on(ToolResultEvent.CHANNEL, _logger, priority=BusPriority.POST)
```

## Internal Storage

`EventBus._handlers` upgrades from `dict[str, list[Handler]]` to `dict[str, list[_Subscription]]`, where:

```python
@dataclass(frozen=True, slots=True)
class _Subscription:
    priority: int
    seq: int
    handler: Handler
```

`seq` is a monotonic counter incremented on every `on()` call across the whole bus (not per channel), used purely as a tie-breaker. Two handlers at the same priority run in registration order — the existing FIFO behavior, preserved for the un-prioritized common case.

Insertion uses `bisect.insort` keyed on `(priority, seq)`: O(log n) per subscribe. Emission iterates the list once: same O(n) cost as today.

`emit` and `emit_sync` change exactly one line — the inner loop unpacks `_Subscription.handler` instead of using the bare callable. Observer hooks (`on_handler_done`) keep getting the bare handler so existing observability code is untouched.

## Replacement-vs-Mutation Semantics

Priority controls dispatch *order*. The existing resolution rules in `loop._collect_replacement` — "last non-None wins" — stay verbatim.

This is enough because dispatch order *is* the authority order under last-wins:

- A `PRE` handler that returns a replacement gets immediately overridden by any later `NORMAL` or `POST` handler that returns one.
- A `POST` handler is the de-facto authoritative voice on a replacement channel: by tier, it runs last.
- Within a tier, registration order still breaks the tie (as today).

The block contract on `tool_call` (any handler returning `{"block": True}` aborts the call) is unchanged — block is OR-aggregated, not last-wins, so priority doesn't change its outcome but does control whether a downstream handler ever runs to *see* the blocked decision.

The veto contract on `before_install_atom` / `before_unload_atom` follows the same rule as block — first refusal wins, but priority controls who gets to refuse first. In practice a `PRE`-tier permission atom decides first; lower-tier observability still runs.

## Reload Interaction

When `reload_atom` re-installs an atom, its handlers re-register at the same `priority` they had originally — because the new install code passes `priority=...` exactly the same way the old code did. `bisect.insort` lands them in the right slot regardless of position-preservation gymnastics.

The existing position-preservation fix in [atom_reloader](../../src/agentm/harness/atom_reloader.py) `_capture_handler_positions` / `_restore_handler_positions` becomes redundant for prioritized handlers — but stays in place as a safety net for atoms that don't declare priority and rely on default-tier FIFO ordering. Both mechanisms compose cleanly: priority wins the cross-tier ordering, position preservation handles within-tier ordering.

## Migration

Zero-touch for the existing atom set. Default tier is `NORMAL`, registration order within a tier is unchanged, every existing call site gets the same dispatch order it has today. Files that benefit from explicit priority *will* be touched, but optionally:

| Atom | Suggested tier | Reason |
|---|---|---|
| `permission` | `PRE` on `tool_call` | A deny decision should run before any other handler sees the call |
| `tool_filter` | `PRE` on `before_send_to_llm` | Strip disallowed tools from the list before the model sees it |
| `system_prompt` | `NORMAL` (default) | Business-logic prompt assembly |
| `skill_loader` | `NORMAL` (default) | Same |
| `prompt_templates` | `NORMAL` (default) | Same |
| `observability` | `POST` everywhere | See final state after every other handler ran |
| `cost_budget` | `POST` on `before_send_to_llm` | Compute cost on the exact bytes about to leave |

These re-tagging changes ship in a follow-up PR, not the priority infrastructure PR. Decoupling lets the infrastructure land green without coordinating across every builtin.

## Edge Cases

1. **Subscribing during emit**. The bus snapshots its handler list at `emit()` entry (`handlers = list(registered or ())`). Handlers that subscribe new handlers during dispatch don't see them this turn — same as today; priority insertion happens on the live list and is visible to the *next* emit, deterministically placed by tier.

2. **Same-priority same-seq?**. Impossible. `seq` is monotonic across the whole bus; two `on()` calls always get distinct `seq` values.

3. **Negative or above-tier priorities**. Allowed. `priority=-1` runs before `PRE`; `priority=10000` runs after `POST`. Useful for emergency overrides; reviewers should push back on raw numbers in atom code reviews.

4. **Priority on `emit_sync` vs `emit`**. Identical semantics — sync emit iterates the same priority-sorted list, just skipping coroutines.

5. **Reload changing priority**. An atom rewriting itself can call `api.on(channel, handler, priority=X)` with a different `X` than its prior install. New tier wins; old tier is forgotten when the prior subscription is unsubscribed by `_remove_handlers`. This matches what every other "atom rewrites itself" surface does.

## Test Plan

Per the first-principles testing rule in CLAUDE.md, this touches one fail-stop position: handler dispatch order under reload. One e2e test guards that:

- Three atoms register on the same channel at `PRE` / `NORMAL` / `POST`.
- Each handler appends a marker to a list inside the event payload.
- Verify dispatch order is `[PRE, NORMAL, POST]` regardless of atom install order.
- Reload the `PRE` atom (changing its marker text); verify dispatch order is still `[PRE-v2, NORMAL, POST]`. (Without priority, reload would have moved PRE to the tail and produced `[NORMAL, POST, PRE-v2]`.)
- Subscribe a fourth handler at `priority=BusPriority.POST` AFTER the first POST handler, verify it runs after — within-tier FIFO survives.

No fine-grained tests for the bisect insertion, `_Subscription` shape, or numeric vs symbolic priority — those are framework guarantees, not AgentM-defining invariants.

## Acceptance Criteria

| | |
|---|---|
| ✓ | `BusPriority.PRE/NORMAL/POST` constants live in `core/abi/events.py` and re-export from `agentm.core.abi`. |
| ✓ | `ExtensionAPI.on` accepts `priority=...` keyword; default `NORMAL`. |
| ✓ | `EventBus.on` accepts `priority`; insertion uses `bisect.insort` on `(priority, seq)`. |
| ✓ | `EventBus.emit` and `emit_sync` dispatch in `(priority, seq)` order. |
| ✓ | The single e2e test in the test plan passes. |
| ✓ | Existing 89 tests still pass with no changes (zero-touch migration verified). |
| ✓ | `uv run mypy src/` and `uv run ruff check src/` clean. |

## Out of Scope

- **Constraint declarations** (`before=other_atom`, `after=...`). Track in a separate design if/when we hit a case priority can't express.
- **Per-handler observability attribution upgrade**. Already pursued in the broader observability backlog — priority changes the dispatch order, not the trace shape.
- **Re-tagging built-in atoms with explicit priority**. Follow-up PR; specifically called out in Migration above.
