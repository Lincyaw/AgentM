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
```

## Usage

All commands run from the AgentM repo root, using tau2-bench's venv:

```bash
# List available AgentM model profiles
cd ~/AoyangSpace/tau2-bench
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py list-profiles

# List tau2 domains
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py list-domains

# Quick sanity check (mock domain, 1 task)
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py run \
    --model litellm-dsv4flash --domain mock --num-tasks 1

# Airline domain (3 tasks)
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py run \
    --model litellm-dsv4flash --domain airline --num-tasks 3

# Retail domain (full, all tasks)
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py run \
    --model litellm-dsv4flash --domain retail

# Use different model for user simulator
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py run \
    --model litellm-dsv4flash --user-model azure-gpt --domain airline

# Raw litellm model (no AgentM profile needed)
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py run \
    --raw-llm openai/gpt-4.1 --domain airline --num-tasks 5

# Parallel execution
uv run python ~/AoyangSpace/AgentM/contrib/evals/tau2/run_eval.py run \
    --model litellm-dsv4flash --domain airline -j 4
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
- **pass^k**: whether the agent solves the same task on every one of k tries
- **DB match**: predicted database end state matches the target

Results are auto-saved to `tau2-bench/data/simulations/`.

## Architecture

```
AgentM config.toml          tau2-bench
────────────────           ──────────
model profiles  ──→  agentm_agent.py  ──→  HalfDuplexAgent
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
