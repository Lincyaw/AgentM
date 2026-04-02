# Task: Configuration and Scenario Wiring

**Plan**: [investigation-sanitizer](../plans/2026-04-02-investigation-sanitizer.md)
**Phase**: 5 — Config & Wiring
**Design**: [investigation-sanitizer](../designs/investigation-sanitizer.md) §10, §6.4
**Depends on**: [sanitizer-middleware](2026-04-02-sanitizer-middleware.md)

## Scope

Add sanitizer configuration to the config schema, wire sanitizer components into the RCA scenario setup, and update scenario YAML defaults.

## Deliverables

### Config schema — `src/agentm/config/schema.py`

```python
class SanitizerConfig(BaseModel):
    enabled: bool = False
    critic_model: str = ""           # empty = use compression_model
    periodic_interval: int = 5
    drift_window: int = 3
    drift_threshold: int = 3
    max_block_retries: int = 3       # degrade after N consecutive blocks
    block_on: list[str] = ["C1", "C2", "J3"]
    warn_on: list[str] = ["E1", "E2", "E3", "E4", "C4", "J2", "P1"]
    disable: list[str] = []
```

Add to `OrchestratorConfig`:
```python
sanitizer: SanitizerConfig | None = None
```

Validation: `block_on`/`warn_on`/`disable` codes must be from valid set; warn on unknown codes.

### RCA scenario wiring — `src/agentm/scenarios/rca/scenario.py`

Update `RCAScenario.setup()`:
1. Read sanitizer config from `ctx` (passed through SetupContext or config)
2. If sanitizer enabled:
   - Create `InvestigationTracker()`
   - Create `CodeSanitizer(severity_map, disabled_codes, drift_window, drift_threshold)`
   - Resolve critic model: use `sanitizer.critic_model` if set, else fall back to compression model
   - Create `CriticSanitizer(model, severity_map, disabled_codes)`
   - Create `SanitizerMiddleware(sanitizers, tracker, hypothesis_store, profile_store, trajectory, periodic_interval, max_block_retries)`
   - Add to `ScenarioWiring.orchestrator_middleware`
3. If sanitizer disabled or not configured: no middleware added (current behavior)

### Scenario YAML — `config/scenarios/rca_hypothesis/scenario.yaml`

Add default sanitizer config under orchestrator:
```yaml
orchestrator:
  sanitizer:
    enabled: true
    periodic_interval: 5
    drift_window: 3
    block_on: [C1, C2, J3]
    warn_on: [E1, E2, E3, E4, C4, J2, P1]
```

### SetupContext extension

If sanitizer config is not directly accessible from SetupContext, evaluate minimal change needed. Options:
- Pass full `ScenarioConfig` through SetupContext (add `config` field)
- Or pass sanitizer config specifically
- Prefer the minimal option that doesn't break existing scenarios

### Tests

- Config validation: valid codes accepted, unknown codes warned
- RCA scenario setup with sanitizer enabled: verify middleware in wiring
- RCA scenario setup with sanitizer disabled: verify no sanitizer middleware
- Config defaults: verify block_on/warn_on defaults match design
