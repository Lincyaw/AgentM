# Provider Layer, CLI, and Logging Reference

## LLM provider layer

The SDK wraps LLM providers (`llm_openai`, `llm_anthropic`). Never import
`openai` or `anthropic` directly in atoms — the provider layer handles:

- Message encoding (tool calls, images, system prompts)
- Streaming with backpressure
- Retry with exponential backoff (via the `retry_policy` atom)
- `reasoning_effort` mapping across providers
- `config.toml` model profiles

### Model profiles (`$AGENTM_HOME/config.toml`)

`AGENTM_HOME` defaults to `~/.agentm`.

```toml
default_model = "doubao"

[models.doubao]
provider = "openai"
model = "doubao-seed-2-0-pro-260215"
base_url = "https://ark.cn-beijing.volces.com/api/v3"
api_key = "..."
context_window = 131072
reasoning_effort = "high"

[models.doubao.extra_body]
thinking = { type = "enabled" }
```

Precedence: CLI flag > env / `.env` > `config.toml` profile.

### Structured output

Don't use `openai.beta.chat.completions.parse()` or Anthropic's tool-use
for structured extraction. The SDK's provider layer already handles
`response_format` and JSON schema conversion. If an atom needs structured
LLM output, use the provider's built-in support through `ProviderConfig`.

---

## CLI conventions (typer)

The CLI uses `typer` with a single root `app` in `agentm/cli/main.py`; command
groups live under `agentm/cli/`, with trace internals grouped under
`agentm/cli/trace/`.

```python
@app.command()
def my_subcommand(
    name: Annotated[str, typer.Argument(help="...")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    ...
```

Rules:
- Use `typer.BadParameter` for input validation errors (exit code 2)
- Use `raise SystemExit(1)` for runtime failures
- stdout for machine-readable output, stderr for human messages
- `--format ndjson|table|text` when output has multiple consumers
- Non-interactive by default (no prompts, no `input()`)

---

## Logging and diagnostics

Atoms use **two channels** depending on audience:

### For observability / debugging (developers)

```python
import logging
logger = logging.getLogger(__name__)
logger.warning("something unexpected: %s", detail)
```

### For user-visible diagnostics (surfaced in TUI/trace)

```python
from agentm.core.abi.events import DiagnosticEvent

await api.events.emit(
    DiagnosticEvent.CHANNEL,
    DiagnosticEvent(level="warning", source="my_atom", message="...")
)
```

Don't use `print()`. Don't write to stdout from atoms — the CLI owns stdout.
