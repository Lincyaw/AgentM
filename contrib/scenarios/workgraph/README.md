# WorkGraph

WorkGraph is a lightweight orchestration scenario for long-running repository
automation. A main agent talks to the user and maintains a filesystem task bus;
short-lived workflows claim one task at a time, dispatch isolated workers, and
write results back to the bus.

The design intentionally keeps the control plane small:

- files are messages;
- directories are queues;
- lock files are leases;
- development workers run in ARL `agent_env` sandboxes;
- coder/verifier/merger agents own git and GitHub operations;
- verifier agents independently check worker branches.

## State Layout

By default the state directory is `.agentm/workgraph/` under the workflow cwd:

```text
.agentm/workgraph/
  goals.md
  decisions.md
  ready/
  running/
  verified/
  merging/
  merge_pending/
  done/
  failed/
  conflicts/
  locks/
  results/
```

Task files are Markdown. Only a few header fields are interpreted by the
workflow; the remaining body is passed verbatim to the worker.

`done/` means the task has landed in the base branch. A task that only has a
verified PR lives in `verified/`, and a PR with GitHub auto-merge enabled but
not yet landed lives in `merge_pending/`. `Depends` is evaluated against
`done/` so downstream work only starts after upstream changes are actually in
the base branch.

Prefer domain-sized tasks: one bounded context, domain capability, or coherent
cross-service capability with one owner and one validation gate. Do not split a
feature into many helper-sized tasks unless the dependencies, ownership, or
conflict risk justify it. Use service/domain locks for normal parallel work;
avoid broad locks such as `project-index` unless the orchestrator intentionally
wants to serialize a high-risk shared edit.

```md
# REQ-005 Service Plan domain foundation

Depends: REQ-004
Locks: services/service-plan
Repo: git@github.com:OperationsPAI/train-ticket.git
Base: refactor/greenfield-ddd

## Goal
Implement the first coherent Service Plan delivery: aggregates, value objects,
events, tests, docs, and requirement metadata needed by downstream Capacity and
Trip Planning work.

## Context
Read:
- project-index.yaml
- docs/02-domains/service-plan.md

## Validation
- cd services/service-plan && go test ./...
- make check-strict
```

## Main Agent

Run the main orchestrator as a normal AgentM scenario:

```bash
agentm --scenario workgraph/main --cwd /path/to/control-repo
```

The main agent should create or edit task files, monitor state directories, and
invoke the develop and merge workflows when work is ready.

## Develop Workflow

Run one scheduling pass manually:

```bash
agentm workflow run contrib/scenarios/workgraph/workflow/develop.py \
  --cwd /path/to/control-repo \
  --args '{
    "state_dir": ".agentm/workgraph",
    "repo": "git@github.com:OperationsPAI/train-ticket.git",
    "base": "refactor/greenfield-ddd",
    "max_parallel": 1,
    "agent_env": {
      "backend": "agent_env",
      "image": "train-ticket-dev:local",
      "gateway_url": "http://localhost:8080",
      "work_dir": "/workspace",
      "create_timeout": 1200,
      "delete_on_shutdown": false
    }
  }'
```

`agent_env` is required for development workers. The worker manifests install
the `operations` atom with `backend: agent_env`, and the workflow fails before
claiming tasks unless `args.agent_env.image` or `AGENTM_AGENT_ENV_IMAGE` is
set. When the controller process has `GH_TOKEN` or `GITHUB_TOKEN`, the workflow
forwards both common names into the sandbox through the operations atom's
`agent_env.config_env.vars` interface. Explicit `args.agent_env.config_env`
values win when they provide the same variable name.
Worker prompts require agents to avoid printing credentials. For GitHub
git-over-HTTPS, use a token URL or a Basic extraheader generated from
`x-access-token:${GH_TOKEN}`; Bearer headers are not reliable for git clone or
push.
For automatic task claiming, pass an image so each claimed task gets its own
sandbox. Do not pass a shared `attach_session` to the develop workflow; the
only supported attach path is the workflow's own per-task session reuse.

Within one workflow task pipeline, WorkGraph reuses the same ARL sandbox across
follow-up worker calls. The coder reports `AgentEnvSession`, the workflow stores
it in `results/<task-id>/agent_env_session.txt`, and the verifier in that same
pipeline attaches to that session instead of creating a fresh sandbox.

## Worker Contract

The workflow does not perform git operations for the worker. The coder runs in
its own ARL sandbox and is expected to clone, branch, commit, push, and open a
PR. A sandbox-local commit is not a delivery: `Status: success` requires a
remote branch or PR visible to the verifier. Its final response must begin with
one of:

```text
Status: success
Status: failed
Status: conflict
AgentEnvSession: <agent_env.session_id or none>
Branch: <branch or none>
Commit: <commit or none>
Remote: <remote branch name/url or none>
PR: <url or none>
```

The workflow coerces a coder `Status: success` with empty `Remote:` and `PR:`
to failed before verification. The verifier receives the task and the coder's
result, performs a fresh clone or checkout of the remote branch/PR, runs the
validation commands, and reports:

```text
Status: passed
Status: failed
AgentEnvSession: <agent_env.session_id or none>
```

The workflow moves verifier-passed task files to `verified/`; failed or
conflicted tasks move to `failed/` or `conflicts/`. It writes `result.md`,
`validation.md`, `task.md`, and `agent_env_session.txt` under
`results/<task-id>/`.

## Merge Workflow

Run one merge scheduling pass manually:

```bash
agentm workflow run contrib/scenarios/workgraph/workflow/merge.py \
  --cwd /path/to/control-repo \
  --args '{
    "state_dir": ".agentm/workgraph",
    "repo": "git@github.com:OperationsPAI/train-ticket.git",
    "base": "refactor/greenfield-ddd",
    "max_parallel": 1,
    "agent_env": {
      "backend": "agent_env",
      "image": "train-ticket-dev:local",
      "gateway_url": "http://localhost:8080",
      "work_dir": "/workspace",
      "create_timeout": 1200,
      "delete_on_shutdown": false
    }
  }'
```

The merge workflow claims `merge_pending/` before `verified/`, then moves the
task to `merging/` while the merge agent runs. It uses a filesystem lock derived
from the task's repo/base pair so multiple PRs targeting the same base are
rebased and merged serially. Different repos or base branches may still run in
parallel when `max_parallel` allows it.

The merge agent runs in ARL `agent_env` and owns all `git` and `gh` operations:
inspect the PR, fetch the latest base, rebase the PR branch, push the rebased
branch with `--force-with-lease`, run validation, and merge through GitHub with
`gh pr merge --rebase --delete-branch`. If branch protection requires waiting,
it may enable auto-merge with `gh pr merge --auto --rebase --delete-branch` and
report `Status: auto_merge`; the workflow then parks the task in
`merge_pending/` for a later pass. A successful merge moves the task to
`done/`; non-mechanical conflicts move to `conflicts/`; validation or tooling
failures move to `failed/`.
