# Task: Vault Integration with Builder and Scenarios

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: implementer
**Depends**: [vault-tools](2026-03-16-vault-tools.md), [vault-migration](2026-03-16-vault-migration.md)

## Objective

Wire the Memory Vault into the existing system: builder.py dependency injection, scenarios/memory_extraction/ registration, pyproject.toml dependency, and public API surface.

## Inputs

- [Design doc § Integration Points, § Configuration](../designs/memory-vault.md)
- `src/agentm/builder.py` — KnowledgeStore wiring
- `src/agentm/scenarios/memory_extraction/__init__.py` — registration pattern
- `pyproject.toml` — dependency list

## Outputs

- `src/agentm/tools/vault/__init__.py` (new) — public API
- `src/agentm/builder.py` (modify) — add vault creation and tool injection
- `src/agentm/scenarios/memory_extraction/__init__.py` (modify) — register vault tools
- `pyproject.toml` (modify) — add sqlite-vec dependency
- `tests/unit/test_vault_integration.py` (new)

## Implementation Details

### 1. Public API (`__init__.py`)

```python
from agentm.tools.vault.store import MarkdownVault
from agentm.tools.vault.tools import create_vault_tools
from agentm.tools.vault.migration import migrate_knowledge_store

__all__ = ["MarkdownVault", "create_vault_tools", "migrate_knowledge_store"]
```

### 2. Builder wiring (`builder.py`)

In `AgentSystemBuilder.build()`, when system_type is `memory_extraction`:

```python
from agentm.tools.vault import MarkdownVault, create_vault_tools
vault = MarkdownVault(vault_dir)
VAULT_TOOLS = create_vault_tools(vault)
```

Wire vault tools into the tool resolution loop (same pattern as KNOWLEDGE_TOOLS).

### 3. Scenario registration

Update `scenarios/memory_extraction/__init__.py` to register vault tool names.

### 4. Dependency

Add `sqlite-vec>=0.1.6` to `pyproject.toml` dependencies.

## Acceptance Conditions

- [ ] `from agentm.tools.vault import MarkdownVault, create_vault_tools` works
- [ ] Builder creates MarkdownVault when system_type is memory_extraction
- [ ] Vault tools available in scenario tool resolution
- [ ] sqlite-vec in pyproject.toml dependencies
- [ ] Existing KnowledgeStore path still works for hypothesis_driven
- [ ] `uv sync` succeeds with new dependency
- [ ] Full test suite (existing + new) passes

## Notes

- Keep backward compatibility: KnowledgeStore remains for hypothesis_driven scenario
- Vault replaces KnowledgeStore only for memory_extraction
