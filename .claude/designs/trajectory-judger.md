# Trajectory Judger Scenario Design

## Overview

This scenario is designed to analyze RCA agent trajectories, classify them according to a decision-tree methodology, and identify the reasons behind an agent's success or failure.

It adopts a **single-pass analysis mode**: the agent reads the entire trajectory in one go, outputs structured labels, and does not enforce a step-by-step execution.

**Scenario Name**: `trajectory_judger`
**Note**: This is a separate scenario from the existing `trajectory_analysis` scenario. The latter is skill-driven for cross-case pattern extraction, while this one is decision-tree guided for single-case classification.

## Decision-Tree Taxonomy

```
Is the conclusion correct?
├── Yes
│   └── Is the evidence reliable?
│       ├── Yes → success (full success)
│       └── No → lucky_hit (got it by chance, needs attention)
│
└── No
    └── At which stage did the failure occur?
        ├── Exploration stage → exploration_fail (didn't find it)
        ├── Confirmation stage → confirmation_fail (didn't verify it)
        └── Judgment stage → judgment_fail (misunderstood or hallucinated)
```

## Category Definitions

| Category | Criterion | Typical Scenario |
|----------|-----------|------------------|
| **success** | Correct conclusion + sufficient and reliable evidence | Normal successful case |
| **lucky_hit** | Correct conclusion + insufficient or problematic evidence | Got it right by chance; needs optimization |
| **exploration_fail** | Failed to find the root-cause service | Exploration blind spot, stopped too early, confirmation bias |
| **confirmation_fail** | Found the service but failed to verify it | Query failure, context drift, lack of confirmation awareness |
| **judgment_fail** | Verified the service but reached the wrong conclusion | Misread data, hallucination, stuck in wrong branch |

## Core Distinction: confirmation_fail vs judgment_fail

| | confirmation_fail | judgment_fail |
|--|-------------------|---------------|
| **Essence** | **Action omission** — didn't perform validation | **Cognitive error** — validated but misinterpreted |
| **One-liner** | "Queried but didn't confirm" | "Confirmed but misread" |
| **Criterion** | There is a query targeting the service, but the agent offered no explicit interpretation of the result | There is a query + interpretation, but the interpretation contradicts the data |

## Sub-Type Definitions

| Main Category | Sub-Type | Definition | Typical Manifestation |
|---------------|----------|------------|-----------------------|
| **exploration_fail** | **missed_target** | Never queried the ground-truth service | No metrics/logs/trace query for this service anywhere in the trajectory |
| | **insufficient_exploration** | Queried the service but stopped too early | Fewer than 2 queries or less than 20% of total steps |
| | **confirmation_bias** | Over-focused on the initial hypothesis | Queries for a single service account for > 60% of all queries |
| **confirmation_fail** | **dropped_lead** | Mentioned the service but dropped it with no follow-up query | Text mentions the service, but no subsequent metrics/logs/trace query |
| | **verification_fail** | Query execution failed or returned no valid result | Query error or empty/abnormal response |
| **judgment_fail** | **misread_data** | Query result exists but interpretation is wrong | Data clearly contradicts the conclusion |
| | **overconfident_wrong** | Reached a firm conclusion without substantive evidence | No valid query, yet a definitive conclusion is given |
| **lucky_hit** | **hallucination_evidence** | Fabricated evidence | validation fails |
| | **incomplete_chain** | Evidence chain is incomplete | Missing a complete symptom → hypothesis → verification → conclusion chain |

## Single-Pass Analysis Flow

The agent reads the trajectory and ground truth in one pass, analyzes them according to the methodology below, and directly outputs a `TrajectoryLabel`.

### Analysis Methodology (Prompt Guidance)

```
## Analysis Steps

1. **Extract the conclusion**
   - Extract the agent's final conclusion from the last assistant message
   - Record the step number and the original quote

2. **Compare with Ground Truth**
   - Compare the agent's conclusion against the true root cause
   - Determine: correct / partially correct / incorrect

3. **Classify** (if the conclusion is incorrect)

   a. **Did the agent ever query the true service?**
      - Search the trajectory for metrics/logs/trace queries targeting the ground-truth service
      - None → exploration_fail
      - Yes → continue

   b. **Did the agent attempt to confirm?**
      - After the query, did the agent provide an explicit interpretation of the result?
      - No → confirmation_fail
      - Yes → continue

   c. **After confirmation, was the conclusion correct?**
      - Is the interpretation consistent with the data?
      - Inconsistent → judgment_fail

4. **Evidence reliability check** (if the conclusion is correct)
   - Check for fabricated evidence (re-run key queries if possible)
   - Check whether the evidence chain is complete
   - Issues found → lucky_hit
   - No issues → success

5. **Output the label**
   - Fill in all fields of TrajectoryLabel
   - The reasoning field must contain the full analysis process
```

### Output Structure

```python
class TrajectoryLabel(BaseModel):
    """Trajectory analysis label"""

    # Basic info
    trajectory_id: str
    case_id: str

    # Contrast
    agent_conclusion: list[str]           # Services identified by the agent as root cause
    ground_truth: list[str]               # Actual root-cause services
    is_correct: bool                      # Fully correct?
    is_partial: bool                      # Partially correct?

    # Classification result
    category: Literal[
        "success",
        "lucky_hit",
        "exploration_fail",
        "confirmation_fail",
        "judgment_fail"
    ]
    sub_type: str | None                  # Sub-type (see table above)

    # Analysis details
    reasoning: str                        # Detailed justification (200+ characters)
    evidence: list[dict]                  # Evidence supporting the label

    # Key step locations
    key_steps: {
        "found_step": int | None,         # First step mentioning the true service
        "confirm_step": int | None,       # Step attempting confirmation
        "conclusion_step": int            # Step outputting the conclusion
    }

    # Statistics
    stats: {
        "total_queries": int,             # Total number of queries
        "queried_services": list[str],    # List of services queried
        "query_distribution": dict[str, int]  # Query count per service
    }

    # Metadata
    analyzed_at: datetime
    analyzer_version: str
```

### Output Example

```json
{
  "trajectory_id": "tid-123",
  "case_id": "case-456",
  "agent_conclusion": ["service-a"],
  "ground_truth": ["service-b"],
  "is_correct": false,
  "is_partial": false,
  "category": "exploration_fail",
  "sub_type": "confirmation_bias",
  "reasoning": "The agent's conclusion is incorrect. The ground-truth root cause is service-b, yet the agent never explored this service. Analyzing the trajectory reveals 8 total queries, 5 of which were concentrated on the initial hypothesis service-a, demonstrating clear confirmation bias. At step 3 the agent briefly remarked 'maybe we should look at service-b', but was immediately distracted, and the service was never mentioned again in the remaining 12 steps.",
  "evidence": [
    {"type": "conclusion_mismatch", "detail": "agent identified service-a, actual root cause is service-b"},
    {"type": "query_distribution", "detail": "5 out of 8 queries focused on service-a (62.5%)"},
    {"type": "missed_target", "detail": "never queried metrics/logs/trace for service-b"},
    {"type": "dropped_mention", "detail": "step 3 mentioned service-b but no follow-up occurred"}
  ],
  "key_steps": {
    "found_step": 3,
    "confirm_step": null,
    "conclusion_step": 15
  },
  "stats": {
    "total_queries": 8,
    "queried_services": ["service-a", "service-c", "gateway"],
    "query_distribution": {"service-a": 5, "service-c": 2, "gateway": 1}
  },
  "analyzed_at": "2024-01-15T10:30:00Z",
  "analyzer_version": "1.0.0"
}
```

## Agent Workflow

### Single-Trajectory Analysis

**Input**: `AnalyzeTask`
```python
class AnalyzeTask(BaseModel):
    trajectory_id: str
    trajectory_data: dict  # trajectory JSON
    case_id: str
    ground_truth: list[str]
```

**Output**: `TrajectoryLabel`

**Implementation**: Uses the harness agent interface configured with `output_schema=TrajectoryLabel`

```python
async def analyze(task: AnalyzeTask) -> TrajectoryLabel:
    from agentm.tools.trajectory_reader import jq_query

    # Get scenario wiring
    scenario = get_scenario("trajectory_judger")
    wiring = scenario.setup(ctx)

    # Use harness agent (implementation depends on current harness interface)
    agent = ...  # harness agent with output_schema=wiring.output_schema

    input_text = f"""
    Trajectory ID: {task.trajectory_id}
    Case ID: {task.case_id}
    Ground Truth: {task.ground_truth}

    Trajectory Data:
    {json.dumps(task.trajectory_data, indent=2)}

    Please analyze the above trajectory and output a TrajectoryLabel.
    """

    result = await agent.run(input_text)
    return TrajectoryLabel(**result.output)
```

**Note**: The `jq_query` tool is available at `src/agentm/tools/trajectory_reader.jq_query` for trajectory JSON queries.

### Batch Analysis

**Workflow**:
1. Load the list of trajectories to be analyzed
2. Execute single analyses sequentially or concurrently
3. Collect all labels
4. Generate a classification statistics report

**Output**:
```python
class BatchReport(BaseModel):
    total: int
    by_category: dict[str, int]
    by_sub_type: dict[str, int]
    confidence_distribution: dict[str, list[float]]
    detailed_results: list[TrajectoryLabel]
```

## Prompt Design

### System Prompt

```
You are an RCA Trajectory analysis expert. Your task is to analyze a single agent trajectory and classify its success/failure category according to the decision-tree methodology.

## Decision Tree
{decision_tree}

## Core Distinction
- confirmation_fail = "Queried but didn't confirm" (action omission)
- judgment_fail = "Confirmed but misread" (cognitive error)

## Analysis Requirements
1. Read the entire trajectory to understand the agent's reasoning process
2. Compare the agent's conclusion against the ground truth
3. Determine the category and sub-type according to the decision tree
4. Explain the analysis process in detail in the reasoning field (200+ characters)
5. List concrete evidence in the evidence field (step number, query content, agent statement)

## Available Tools
- jq_query: for complex JSON queries (optional; you may also read directly). Already implemented in `src/agentm/tools/trajectory_reader.py`.

## Output Format
You must output structured data conforming to the TrajectoryLabel schema, including the complete analysis process.
```

## Integration Architecture

### Directory Structure
```
src/agentm/scenarios/trajectory_judger/
├── __init__.py          # register() → register_scenario(TrajectoryJudgerScenario())
├── scenario.py          # TrajectoryJudgerScenario.setup() → ScenarioWiring
└── data.py              # TrajectoryLabel, AnalyzeTask, BatchReport
```

### Scenario Registration
Add to `src/agentm/scenarios/__init__.py`:
```python
from agentm.scenarios.trajectory_judger import register as register_tj
register_tj()
```

### Tool Dependencies
- `jq_query`: Already implemented in `src/agentm/tools/trajectory_reader.py`

### Key Design Decisions
1. **Separate scenario**: Named `trajectory_judger` to avoid conflict with existing `trajectory_analysis`
2. **Minimal wiring**: Only `output_schema` required; no orchestrator/worker tools needed for single-pass analysis
3. **Reusable tool**: `jq_query` from trajectory_reader module can be injected at runtime if needed

## Differences from Existing trajectory_analysis

| | Existing trajectory_analysis | This scenario (trajectory_judger) |
|--|------------------------------|-----------------------------------|
| **Scenario name** | `trajectory_analysis` | `trajectory_judger` |
| **Purpose** | Extract cross-case patterns and generate knowledge | Single-case diagnostic classification |
| **Output** | Vault knowledge entries | Structured labels (TrajectoryLabel) |
| **Method** | Free-form analysis | Decision-tree guided |
| **Execution mode** | Multi-turn iterative | Single-pass |
| **Use case** | Post-hoc summarization and optimization | Batch labeling, failure attribution |

## Implementation Plan

1. **Phase 1**: Foundation
   - Create `src/agentm/scenarios/trajectory_judger/` directory structure
   - Define data models (`TrajectoryLabel`, `AnalyzeTask`, `BatchReport` in `data.py`)
   - Implement `TrajectoryJudgerScenario` class in `scenario.py`
   - Add `register()` in `__init__.py`
   - Register in `src/agentm/scenarios/__init__.py`

2. **Phase 2**: Integration
   - Add scenario to `index.yaml` concepts
   - Create system prompt template in `config/scenarios/trajectory_judger/prompts/`
   - Add unit tests for data models

3. **Phase 3**: Validation
   - Validate classification accuracy on known cases
   - Refine criteria

## Edge-Case Handling

### Multiple root-cause services
- **All found**: category is determined by the normal flow
- **Partially found**: `is_partial: true`, category treated as incorrect
- **None found**: analyze based on the first ground-truth service

### No conclusion / abnormal termination
- Agent produced no conclusion → category: `exploration_fail`, sub_type: `insufficient_exploration`
- Agent error / timeout → separately marked as `error`, excluded from classification statistics

### Partially correct conclusion
- Found one true root cause but also included incorrect services → treated as incorrect (output is inaccurate)
- Found only some root causes, missing others → treated as incorrect
