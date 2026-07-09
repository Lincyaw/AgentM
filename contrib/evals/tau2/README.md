# tau2-bench Evaluation for AgentM

Evaluates AgentM model profiles on [tau2-bench](https://github.com/sierra-research/tau2-bench) —
a benchmark for tool-using conversational agents in customer-service domains.

## Setup

```bash
# Clone tau2-bench (sibling directory)
cd ~/AoyangSpace
git clone https://github.com/sierra-research/tau2-bench.git

# Install dependencies
cd tau2-bench
uv sync
uv run tau2 check-data

# Set env so the adapter finds tau2-bench
export TAU2_BENCH_DIR=~/AoyangSpace/tau2-bench
```

## Usage

All commands run from the AgentM repo root via the unified eval CLI:

```bash
# List available AgentM model profiles
agentm eval tau2 list-profiles

# List tau2 domains
agentm eval tau2 list-domains

# Quick sanity check (mock domain, 1 task)
agentm eval tau2 run --model litellm-dsv4flash --domain mock --num-tasks 1

# Airline domain (3 tasks)
agentm eval tau2 run --model litellm-dsv4flash --domain airline --num-tasks 3

# Full retail eval
agentm eval tau2 run --model litellm-dsv4flash --domain retail

# Use different model for user simulator
agentm eval tau2 run --model litellm-dsv4flash --user-model azure-gpt --domain airline

# Raw litellm model (no AgentM profile)
agentm eval tau2 run --raw-llm openai/gpt-4.1 --domain airline --num-tasks 5

# Parallel execution
agentm eval tau2 run --model litellm-dsv4flash --domain airline -j 4

# Custom experiment ID
agentm eval tau2 run --model litellm-dsv4flash --domain airline --exp-id my-airline-run
```

## Domains

| Domain | Tasks | Description |
|--------|-------|-------------|
| `mock` | 4 | Lightweight test domain |
| `airline` | 100+ | Flight booking, cancellation, customer support |
| `retail` | 100+ | Order management, returns, product inquiries |
| `telecom` | 100+ | Account management, troubleshooting |

## Scoring

- **reward**: product of DB match × communicate check (0 or 1)
- **pass_rate**: fraction of tasks with reward >= 1.0
- **DB match**: predicted database end state matches the target

Results are recorded in the unified experiment output at
`~/.agentm/eval_runs/{exp_id}/results.jsonl`.

## Architecture

```
AgentM config.toml          tau2-bench
────────────────           ──────────
model profiles  ──→  tau2_agent.py   ──→  HalfDuplexAgent
(base_url,             (adapter)          (generate_next_message)
 api_key,                    │
 model)                      ↓
                        litellm.completion()
                             │
                             ↓
                      tau2 Orchestrator
                    (environment + user sim)
```

The adapter reads AgentM's `~/.agentm/config.toml` model profiles,
resolves them into litellm-compatible parameters, and plugs into
tau2's standard evaluation pipeline.
