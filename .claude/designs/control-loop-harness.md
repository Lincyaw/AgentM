# Control-Loop Harness

Status: **core loop implemented** in workbuddy `internal/control/` (Objective,
Observer/GHObserver, Feedback, Controller.Decide with Resume/Block/Complete +
stuck detection, Run loop, agent_bridge); migration steps 4-6 not yet done
(remove legacy auto-submit/RESULT parsing, extend sandbox lifecycle)
Owner: workbuddy orchestration layer + AgentM session resume

Reaches into: `.github/workbuddy/` (agent prompts, workflow config),
workbuddy Go coordinator (dispatch + post-condition observer), AgentM
`--resume <session_id>` (session continuity).

Related: [agent-team](agent-team.md), [agent-loop](agent-loop.md),
[single-process-gateway](single-process-gateway.md),
[session-inbox](session-inbox.md).

## 1. Problem

The current workbuddy dispatch model is procedural: agent prompts
contain step-by-step bash scripts, the agent executes them in order,
any step failure breaks the chain, and retries start fresh sessions
losing all accumulated context. Three concrete costs:

1. **Fragile sequencing.** A single failed `git push` aborts the
   entire session. The agent has no room to diagnose, retry, or choose
   an alternative path because the prompt prescribes the exact command
   sequence.
2. **Context loss on retry.** Each retry spawns a new session. The
   agent re-reads the repo, re-discovers the issue, re-plans the
   approach. Prior investigation is discarded. With a 3-retry cap this
   means up to 4x the token cost for the same task, with no
   incremental progress guarantee.
3. **Harness-as-worker.** `autoSubmitAgentChanges` and `_on_shutdown`
   auto-submit perform work on behalf of the agent (push, create PR).
   This splits responsibility: failures in harness-managed steps are
   invisible to the agent, and the agent cannot correct what it did
   not do.

## 2. Control theory framing

The fix is to apply the standard closed-loop control model. The
mapping:

```
r (setpoint)   = acceptance criteria + desired end state
                 (branch pushed, PR open, label flipped)
G (plant)      = LLM agent in sandbox
H (sensor)     = post-condition observer
                 (checks git state, GitHub API)
C (controller) = workbuddy worker control loop
e (error)      = r - y = what the agent achieved vs what's needed
u (input)      = resume message injected into the same session
y (output)     = observable world state after agent turn
```

The controller never acts on the plant's self-report. It measures
ground truth (git remote state, GitHub API) and feeds the delta back
as a factual error signal. The plant (agent) keeps its full context
across iterations because the session is resumed, not restarted.

## 3. Design principles

### 3.1 Goal-oriented prompts, not procedures

Tell the agent WHAT to achieve (push branch, create PR, flip label),
never HOW (no bash script templates). The agent decides its own
approach. This is the same "runtime guards the boundary, agent owns
the decision" principle from `sub_agent_lifecycle` and `agent_team` --
generalised to the orchestration layer.

### 3.2 Observe ground truth, don't trust self-report

The harness checks actual world state (git remote, GitHub API) after
each agent turn. No `RESULT:` line, no structured output contract.
Truth is in the observable state, not in the agent's claim about what
it did.

### 3.3 Resume, don't restart

When post-conditions are not met, resume the SAME session with an
error signal describing what is missing. The agent keeps its full
context. AgentM's `--resume <session_id>` supports this directly.
Context is the most expensive thing to rebuild; discarding it on
retry is the single largest waste in the current model.

### 3.4 No fallbacks

The harness never does work on behalf of the agent (no auto-submit,
no coordinator-managed git push). Everything goes through the agent.
The harness only observes and re-invokes. If the harness performs
a step, the agent cannot learn from or correct that step; the
feedback loop is broken at exactly the point where it matters most.

### 3.5 Sandbox lifecycle = control loop lifecycle

The sandbox pod stays alive until all post-conditions pass (or max
rounds hit), not just for a single agent turn. Tearing down the
sandbox between rounds would destroy the agent's local state (working
tree, build artifacts, installed dependencies) -- equivalent to the
context loss problem at the infrastructure level.

## 3.6 Formal concepts

Four functions define the control loop. Each is a pluggable interface --
the first version uses rule-based implementations, but the interface is
designed so a meta-agent (LLM) can replace any function later.

### Observation function

```
O(t) = observe(world_state) -> ObservationVector
```

Observation is the foundation -- you can control only what you can
observe. Three layers:

| Layer | What | How | When |
|---|---|---|---|
| Code | files modified, committed, pushed, lint/test pass | git commands in sandbox / on host | each agent turn end |
| Collaboration | PR exists, review comment posted, label state, issue state | GitHub API (`gh` CLI) | each agent session end |
| Behavior | token usage, turn count, tool call patterns, stuck detection (consecutive identical errors) | agentm session metadata | real-time stream |

The behavior layer is critical -- it lets the harness intervene when the
agent is "alive but unproductive" (e.g., 3 consecutive turns calling the
same command with the same error).

### Objective function

```
r = objective(state) -> ConditionVector
```

Each workflow state has a vector of boolean conditions, not a single
pass/fail. Split into pre-conditions (entry gate) and post-conditions
(exit gate):

```python
# Example: developing state
precondition_developing = {
    "has_acceptance_criteria": True,
    "repo_cloneable": True,
    "gh_token_valid": True,
}

postcondition_developing = {
    "files_modified": True,
    "committed": True,
    "pushed": True,
    "pr_open": True,
    "issue_commented": True,
    "label_is": "status:reviewing",
}

# Example: reviewing state
precondition_reviewing = {
    "branch_exists_remote": True,
    "pr_open": True,
}

postcondition_reviewing = {
    "diff_inspected": True,
    "each_ac_evaluated": True,
    "review_comment_posted": True,
    "label_is": "status:merging | status:developing",
}
```

Pre-condition failure does not start a new agent -- it resumes the
previous-stage agent to complete the missing work.

### Feedback function

```
e(t) = feedback(objective, O(t)) -> Delta
```

Computes the structured gap between objective and observation. Not
boolean "failed" but a delta vector:

```python
def feedback(objective, observation):
    delta = {}
    for key, target in objective.items():
        actual = observation.get(key)
        if actual != target:
            delta[key] = {"expected": target, "actual": actual}
    return delta
```

The delta is rendered as natural language for the agent:

```
Post-condition check: files_modified OK, committed OK, pushed FAIL,
pr_open FAIL.
Please push branch workbuddy/issue-42 and create a PR.
```

### Control function

```
u(t) = control(e(t), history) -> Action
```

A policy that maps error signal + history to an action. First version is
rule-based; interface supports LLM meta-agent replacement.

| Error pattern | History | Action |
|---|---|---|
| Missing post-conditions | First occurrence | **Resume** same session with delta message |
| Same error repeating | 2nd-3rd round | **Resume** with hint ("last time you tried X, consider Y") |
| Same error N rounds | 4th+ round | **Block** -- mark `status:blocked`, wait for human |
| Pre-condition unmet | -- | **Don't dispatch** -- resume previous-stage agent |
| Behavior anomaly (stuck) | -- | **Interrupt** + resume with "you seem stuck, try a different approach" |

## 3.7 Closed-loop diagram

```
        +---------------------------------------------+
        |         Control Function                    |
        |  u = control(e, history)                    |
        |  -> resume / block / interrupt              |
        +----------+----------------------------------+
                   | u (resume message)
                   v
              +---------+
     r ------>|  Plant   |------> y (agent actions)
  (objective) |  (LLM)  |
              +---------+
                              |
                              v
                   +------------------+
                   |    Observer      |
                   | O = observe(y)   |
                   | code + collab +  |
                   | behavior layers  |
                   +--------+---------+
                            | O(t)
                            v
                   +------------------+
                   | Feedback Function|
                   | e = r - O(t)     |
                   +--------+---------+
                            | e (delta)
                            +----------> Control Function
```

## 3.8 Pluggability -- meta-agent interface

Each function (observe, objective, feedback, control) is an interface.
The first version is rule-based. Future versions can replace any
function with an LLM call:

- **Observation**: rule-based (git/gh commands) -- unlikely to need LLM
- **Objective**: rule-based (condition vectors per state) -- could use
  LLM to generate objectives from free-text issue descriptions
- **Feedback**: rule-based (delta rendering) -- could use LLM to
  generate more helpful hints
- **Control**: rule-based (policy table) -- prime candidate for
  meta-agent: "given this history of attempts and errors, what should
  we try next?"

The interface boundary is: each function takes structured input and
returns structured output. Whether a rule engine or an LLM produces the
output is transparent to the loop.

## 4. Post-condition observer

After each agent turn, the controller checks:

| Check | Method | Pass condition |
|---|---|---|
| Files modified | `git diff` in sandbox | Non-empty diff vs baseline |
| Committed | `git log --oneline HEAD` | New commit(s) since baseline |
| Pushed | `git ls-remote origin <branch>` | Branch exists on remote |
| PR created | `gh pr list --head <branch>` | At least one open PR |
| Label flipped | `gh issue view N --json labels` | Target label present |

Checks are ordered by dependency: a missing commit implies push will
also fail, so the error signal reports the earliest unmet condition
first. All checks use ground-truth queries against git and GitHub API
-- never parsed from agent output.

## 5. Error signal format

When post-conditions fail, the resume message is factual and minimal:

```
Post-condition check: files modified OK, committed OK, pushed FAIL,
PR FAIL, label FAIL.
Please push your changes to branch workbuddy/issue-N, create a PR,
and flip the issue label to status:reviewing.
```

The signal states what is true and what is missing. It does not
prescribe commands, suggest strategies, or explain why the agent's
previous attempt failed. The agent has its full session context and
can reason about the gap itself.

## 6. Termination conditions

- **All post-conditions pass** -- success. Delete sandbox, advance
  state machine.
- **Observer results unchanged for N consecutive rounds** -- the agent
  is stuck (same checks fail, no observable progress). Mark
  `status:blocked`, preserve session for human inspection.
- **Max rounds exhausted** (default 5) -- mark `status:blocked`.
  The round cap prevents unbounded token spend on a single task.

The existing `max_retries` and `max_review_cycles` in the workflow
schema compose with the per-dispatch round cap: `max_retries` governs
state-machine-level retries (developing -> blocked -> developing),
while the round cap governs within-dispatch resume iterations.

## 7. What gets removed

| Current mechanism | Why it exists | Replacement |
|---|---|---|
| `autoSubmitAgentChanges` in workbuddy bridge | Harness pushes/PRs on agent's behalf | Agent does it; observer verifies |
| `_on_shutdown` auto-submit in `operations_agent_env` | Same, at SDK level | Same |
| `RESULT:` line structured output contract | Harness parses agent self-report | Observer checks ground truth |
| Step-by-step bash templates in agent prompts | Procedural execution script | Goal-oriented prompt (section 8) |

## 8. Goal-oriented prompt template

The dev-agent prompt becomes:

```
You are working on issue #{{.Issue.Number}} in repo {{.Repo}}.

Title: {{.Issue.Title}}
Body:
{{.Issue.Body}}

Previous comments (including review feedback):
{{.Issue.CommentsText}}

Related PRs:
{{.RelatedPRsText}}

The repo is cloned at /workspace. You have full git and gh access.

Read CLAUDE.md first. Implement the acceptance criteria. When your
changes are ready:
- Push to branch `workbuddy/issue-{{.Issue.Number}}`
- Open a PR referencing this issue
- Comment on the issue summarizing what you did
- Flip the label to `status:reviewing`
```

No scripts, no step numbering, no `RESULT:` line instruction.

The review-agent prompt follows the same principle: state what
constitutes a passing review, let the agent decide how to verify.

## 9. Workflow as higher-order control loop

The dev-review-merge state machine is itself a control loop:

```
           setpoint: "acceptance criteria met"
                      │
                      ▼
              ┌──────────────┐
              │  dev-agent   │◄──── error: review feedback
              │  (plant)     │
              └──────┬───────┘
                     │ artifact (PR)
                     ▼
              ┌──────────────┐
              │ review-agent │──── sensor + secondary controller
              │  (observer)  │
              └──────┬───────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
     criteria pass      criteria fail
            │                 │
            ▼                 └──► error signal (review comment)
     ┌──────────────┐              fed back to dev-agent
     │ merge gate   │
     │ (final gate) │
     └──────────────┘
```

- The dev-agent is the plant for the "implement" setpoint.
- The review-agent is a sensor + secondary controller: review
  rejection is the error signal, fed back to the dev-agent with the
  review feedback as context.
- The merge gate is the final quality check before the system output
  (merged PR).

The inner loop (section 4-6, post-condition observer within a single
dispatch) and the outer loop (dev-review cycle) are independent
feedback loops operating at different timescales. The inner loop
ensures the agent produces a complete artifact; the outer loop ensures
the artifact meets acceptance criteria.

## 10. Relationship to AgentM concepts

This design is deliberately external to `agentm.core`. The control
loop lives in the workbuddy coordinator (Go), not in the AgentM SDK.
AgentM provides the mechanisms the harness relies on:

- **`--resume <session_id>`** -- session continuity for resume-not-
  restart. No SDK change needed; the existing resume path works.
- **`session_inbox`** (when landed) -- the resume message naturally
  enters through the inbox as a user-source item.
- **Observability JSONL** -- the controller could optionally inspect
  the agent's trace for richer diagnostics, but the primary feedback
  loop uses only external ground truth (git, GitHub API).

No new atoms, no new ExtensionAPI surface, no core changes required for the
external workbuddy control loop. (AgentM's existing mechanisms --
`DecideTurnActionEvent` + `Inject` + `TurnEndEvent.messages` + `SessionInbox`
-- are sufficient for atoms to implement turn-level control loops internally
without any new core surface.)

## 11. Migration path

1. **Rewrite dev-agent prompt** to goal-oriented form (section 8).
   Keep review-agent prompt unchanged initially -- it is already
   closer to goal-oriented.
2. **Implement post-condition observer** in the workbuddy coordinator
   as a Go function called after each agent dispatch completes.
3. **Wire resume loop**: on post-condition failure, call
   `agentm --resume <session_id> -p "<error signal>"` instead of
   spawning a new session.
4. **Remove `autoSubmitAgentChanges`** from the bridge and
   `_on_shutdown` from `operations_agent_env`.
5. **Remove `RESULT:` parsing** from the coordinator.
6. **Extend sandbox lifecycle** to span multiple resume rounds within
   one dispatch.

Steps 1-3 can land independently. Steps 4-5 are cleanup after the
new loop is validated. Step 6 requires infrastructure changes to the
K8s pod lifecycle.
