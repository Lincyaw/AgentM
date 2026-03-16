---
name: vault
description: "Memory Vault MCP server — Markdown + SQLite knowledge store with bidirectional [[wikilinks]]. Use when the user wants to create, read, edit, search, or manage interconnected knowledge notes. Provides 11 tools: vault_write, vault_write_batch, vault_read, vault_edit, vault_delete, vault_rename, vault_list, vault_search, vault_backlinks, vault_traverse, vault_lint."
disable-model-invocation: true
argument-hint: "[setup|usage|tools]"
---

# vault-mcp

A Markdown + YAML frontmatter knowledge store with bidirectional `[[wikilinks]]`, backed by SQLite (FTS5 full-text search + sqlite-vec vector search). Designed for AI agents to maintain an interconnected knowledge base.

Package: [vault-mcp](https://github.com/lincyaw/vault-mcp)

## Installation

```bash
uvx vault-mcp
pip install vault-mcp                   # core (keyword search only)
pip install "vault-mcp[semantic]"       # + embedding-based semantic search
```

## Running the MCP Server

```bash
# Start with default settings (vault in ./vault)
vault-mcp

# Custom vault directory
VAULT_DIR=/path/to/notes vault-mcp

# With semantic search enabled
VAULT_DIR=./vault VAULT_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-nano vault-mcp
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VAULT_DIR` | `./vault` | Root directory for markdown files |
| `VAULT_EMBEDDING_MODEL` | *(none)* | Sentence-transformers model name. Enables semantic search when set |

## Client Configuration

### Claude Code

```bash
claude mcp add vault -- vault-mcp
```

Or `.mcp.json`:

```json
{
  "mcpServers": {
    "vault": {
      "type": "stdio",
      "command": "vault-mcp",
      "env": { "VAULT_DIR": "./vault" }
    }
  }
}
```

### Cursor / Windsurf / Other MCP Clients

Add to your MCP settings:

```json
{
  "vault": {
    "command": "vault-mcp",
    "env": { "VAULT_DIR": "./vault" }
  }
}
```

### Python Library (no MCP)

```python
from vault_mcp import MarkdownVault, create_vault_tools

vault = MarkdownVault("./my-vault")
vault.write("skill/debugging", {"type": "skill", "confidence": "pattern"}, "# Debugging\n\n...")

# Or get all 11 tools as a dict of callables
tools = create_vault_tools(vault)
result = tools["vault_search"](query="debugging", mode="keyword")
```

## Note Format

Each note is a `.md` file with YAML frontmatter, organized in type-based folders:

```
vault/
  skill/
    timeout-diagnosis.md
  concept/
    connection-pooling.md
  episodic/
    2024-01-15-incident.md
```

```markdown
---
type: skill
tags: [database, timeout]
confidence: pattern
status: active
---

# Timeout Diagnosis

Description of the skill...

## Evidence
- [[episodic/2024-01-15-incident]]: First observed during outage

## Related
- [[concept/connection-pooling]]
```

### Frontmatter Fields

| Field | Required | Values |
|-------|----------|--------|
| `type` | Yes | `skill`, `episodic`, `concept`, `failure-pattern`, `system-knowledge` |
| `confidence` | Yes | `fact`, `pattern`, `heuristic` |
| `tags` | No | Free-form string list |
| `status` | Auto | `active`, `superseded`, `archived` |

## Tools Reference

### Write & Edit

**`vault_write(path, frontmatter, body)`** — Create or overwrite a note.

**`vault_write_batch(entries)`** — Write multiple notes atomically. `entries` is a list of `{path, frontmatter, body}` dicts.

**`vault_edit(path, operation, params)`** — Incremental edit without rewriting the full file:
- `replace_string`: `{"old": "...", "new": "..."}`
- `set_frontmatter`: `{"confidence": "fact", "tags": [...]}`
- `replace_section`: `{"heading": "## Evidence", "body": "new content"}`
- `append_section`: `{"heading": "## Evidence", "content": "- new line"}`

**`vault_delete(path)`** — Delete a note and clean indexes.

**`vault_rename(old_path, new_path)`** — Move a note. All `[[backlinks]]` in other notes are rewritten automatically.

### Read & Browse

**`vault_read(path)`** — Read a note's frontmatter and body.

**`vault_list(path, depth, type_filter)`** — Browse directory structure.

### Search & Discovery

**`vault_search(query, filters, mode, limit)`** — Search the vault.
- `mode`: `"keyword"` (FTS5), `"semantic"` (vector), `"hybrid"` (both, default)
- `filters`: `{"type": "skill", "confidence": "fact", "tags": ["db"]}`
- Returns rich results: path, score, title, type, confidence, tags, snippet

**`vault_backlinks(path)`** — Find all notes that link to this path.

**`vault_traverse(start, depth, direction)`** — BFS graph traversal.
- `direction`: `"forward"`, `"backward"`, `"both"`
- Returns nodes with depth + directed edges (source → target)

**`vault_lint()`** — Check vault health: dead `[[wikilinks]]` and orphan notes.

## Validation

Every write/edit operation automatically validates and returns warnings:
- Required frontmatter fields present (`type`, `confidence`)
- Field values are valid enums
- Body has an H1 heading
- All `[[wikilinks]]` point to existing notes

The write succeeds regardless — warnings are informational for the agent to decide whether to fix.

## Architecture

```
.md files (source of truth, git-trackable)
    │
    ▼
.vault.db (SQLite index, auto-rebuilt if deleted)
    ├── notes       — YAML frontmatter metadata
    ├── notes_fts   — FTS5 full-text search
    ├── links       — [[wikilink]] directed graph
    ├── tags        — tag index
    └── notes_vec   — embedding vectors (optional)
```
