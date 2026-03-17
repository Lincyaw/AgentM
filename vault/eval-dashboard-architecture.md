---
confidence: pattern
tags:
- eval
- dashboard
- frontend
- backend
- websocket
- streaming
type: concept
---
# Eval Dashboard Architecture

## Overview

The Eval Dashboard extends AgentM's existing monitoring infrastructure to provide real-time visibility into batch LLM evaluation pipelines. It leverages the established WebSocket/REST architecture and adds eval-specific event types, pages, and state tracking.

## Current State (Baseline)

### Existing Infrastructure

1. **FastAPI Server** (`src/agentm/server/app.py`)
   - WebSocket endpoint at `/ws` with `Broadcaster` for multi-client event distribution
   - REST APIs for topology, checkpoint history, state snapshots, trajectory export
   - Trajectory event replay on client connect (historical event synchronization)
   - Message serialization pipeline for LangChain objects

2. **Event System** (`src/agentm/core/trajectory.py`)
   - `TrajectoryCollector` writes events to JSONL at `trajectories/{run_id}.jsonl`
   - `TrajectoryEvent` model with fields: `run_id`, `seq`, `timestamp`, `agent_path`, `node_name`, `event_type`, `data`, `task_id`
   - Async listener pattern via `add_listener()` and `_notify_listeners()` (used for WebSocket broadcasting)
   - Sync variant `record_sync()` for pre-model hooks

3. **Frontend Dashboard** (`src/agentm/server/static/js/`)
   - React 18 app with 4 pages: Topology, Execution, Conversation, Debug
   - WebSocket hook `useWebSocket()` establishes connection with exponential backoff reconnection
   - Event handler maps `event_type` to agent state updates
   - D3.js topology visualization with live status updates
   - Plugin system via `SCENARIO_PLUGINS` for scenario-specific state parsing

4. **Eval Pipeline** (`src/agentm/cli/eval.py`)
   - Four phases: preprocess → rollout → judge → stat
   - Rollout uses `run_investigation_headless()` returning `(structured_response_json, trajectory_json)`
   - Concurrent sample execution via `concurrency` parameter
   - Per-sample timeout enforcement via `asyncio.wait_for()`
   - Judge model with rate limiting and retries
   - rcabench-platform v3 `BaseBenchmark` orchestrates lifecycle

## New Eval Events to Emit

```python
# Eval lifecycle
eval_start: {exp_id, dataset, total_samples, concurrency, timestamp_start}
eval_phase_transition: {phase: preprocess|rollout|judge|stat, timestamp_start, timestamp_end}

# Sample-level (during rollout)
sample_start: {sample_id, dataset_index, incident, data_dir}
sample_complete: {sample_id, status: ok|skip|error, reason?, error?, duration_seconds, response_length, trajectory_length}

# Judge phase
judge_start: {total_samples_to_judge}
judge_result: {sample_id, judge_model, score, feedback}

# Stat phase
eval_stat: {total_samples, ok_count, skip_count, error_count, avg_duration_seconds, avg_judge_score, score_distribution}
```

## Backend Extensions

### New File: `src/agentm/server/eval_state.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime

@dataclass
class SampleMetrics:
    sample_id: str
    dataset_index: int
    incident: str
    status: Literal["pending", "in_progress", "completed", "skipped", "error"]
    duration_seconds: Optional[float]
    response_length: int
    trajectory_length: int
    judge_score: Optional[float]
    judge_feedback: Optional[str]
    error: Optional[str]
    timestamp_start: datetime
    timestamp_complete: Optional[datetime]

@dataclass
class EvalMetrics:
    total_samples: int = 0
    completed_samples: int = 0
    skipped_samples: int = 0
    error_samples: int = 0
    total_duration_seconds: float = 0.0
    judge_scores: list[float] = field(default_factory=list)
    
    @property
    def avg_duration_seconds(self) -> float:
        if self.completed_samples == 0:
            return 0.0
        return self.total_duration_seconds / self.completed_samples
    
    @property
    def avg_judge_score(self) -> float:
        if not self.judge_scores:
            return 0.0
        return sum(self.judge_scores) / len(self.judge_scores)

@dataclass
class EvalSession:
    exp_id: str
    dataset: str
    total_samples: int
    concurrency: int
    agent_type: str
    model_name: Optional[str]
    timestamp_start: datetime
    phase: Literal["preprocess", "rollout", "judge", "stat"] = "preprocess"
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    samples: dict[str, SampleMetrics] = field(default_factory=dict)
```

### Modify `src/agentm/server/app.py`

Add to `create_dashboard_app()`:

```python
# New state
app.state.eval_session = None  # EvalSession instance during eval
app.state.eval_lock = asyncio.Lock()

# New endpoints
@app.get("/api/eval/current")
async def get_eval_session() -> dict:
    session = app.state.eval_session
    if not session:
        return {"error": "No active eval session"}
    return {
        "exp_id": session.exp_id,
        "dataset": session.dataset,
        "total_samples": session.total_samples,
        "phase": session.phase,
        "samples_completed": session.metrics.completed_samples,
        "samples_error": session.metrics.error_samples,
        "samples_skipped": session.metrics.skipped_samples,
        "avg_duration": session.metrics.avg_duration_seconds,
        "avg_judge_score": session.metrics.avg_judge_score,
    }

@app.get("/api/eval/samples")
async def list_eval_samples(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
) -> dict:
    session = app.state.eval_session
    if not session:
        return {"total": 0, "samples": []}
    
    samples = list(session.samples.values())
    if status:
        samples = [s for s in samples if s.status == status]
    
    total = len(samples)
    paginated = samples[offset:offset+limit]
    return {
        "total": total,
        "samples": [
            {
                "id": s.sample_id,
                "status": s.status,
                "duration_seconds": s.duration_seconds,
                "judge_score": s.judge_score,
                "incident_preview": s.incident[:100],
            }
            for s in paginated
        ]
    }

@app.get("/api/eval/samples/{sample_id}")
async def get_eval_sample(sample_id: str) -> dict:
    session = app.state.eval_session
    if not session or sample_id not in session.samples:
        return {"error": "Sample not found"}
    
    sample = session.samples[sample_id]
    return {
        "sample_id": sample.sample_id,
        "incident": sample.incident,
        "status": sample.status,
        "duration_seconds": sample.duration_seconds,
        "judge_score": sample.judge_score,
        "judge_feedback": sample.judge_feedback,
        "error": sample.error,
    }

@app.get("/api/eval/stats")
async def get_eval_stats() -> dict:
    session = app.state.eval_session
    if not session:
        return {}
    
    m = session.metrics
    return {
        "phase": session.phase,
        "samples": {
            "total": session.total_samples,
            "completed": m.completed_samples,
            "error": m.error_samples,
            "skipped": m.skipped_samples,
        },
        "timing": {
            "avg_duration_seconds": m.avg_duration_seconds,
        },
        "judge_scores": {
            "mean": m.avg_judge_score,
            "min": min(m.judge_scores) if m.judge_scores else 0,
            "max": max(m.judge_scores) if m.judge_scores else 0,
        }
    }
```

### Modify `src/agentm/cli/eval.py`

```python
from agentm.server.eval_state import EvalSession, EvalMetrics, SampleMetrics
from agentm.server.app import broadcast_event

async def run_eval(...):
    eval_config = ConfigLoader.load_eval_config(config_path)
    
    # Create and register eval session
    eval_session = EvalSession(
        exp_id=eval_config.exp_id,
        dataset=eval_config.data.dataset,
        total_samples=..., # from preprocessed samples
        concurrency=eval_config.concurrency,
        agent_type=eval_config.agent_type,
        model_name=eval_config.model_name,
        timestamp_start=datetime.now(),
    )
    # Assume app context provides broadcast_event and eval_session
    
    await broadcast_event({
        "event_type": "eval_start",
        "data": {
            "exp_id": eval_session.exp_id,
            "dataset": eval_session.dataset,
            "total_samples": eval_session.total_samples,
            "concurrency": eval_session.concurrency,
        }
    })
    
    # Preprocess
    benchmark.preprocess()
    await broadcast_event({
        "event_type": "eval_phase_transition",
        "data": {"phase": "preprocess", "status": "completed"}
    })
    
    # Rollout
    async def runner(sample: EvaluationSample):
        await broadcast_event({
            "event_type": "sample_start",
            "data": {
                "sample_id": sample.id,
                "dataset_index": sample.dataset_index,
                "incident": sample.augmented_question or sample.raw_question,
            }
        })
        
        start_time = time.time()
        try:
            response_json, trajectory_json = await run_investigation_headless(...)
            status = "ok" if response_json else "skip"
            error = None
        except asyncio.TimeoutError:
            status = "error"
            error = "timeout"
            response_json = None
        except Exception as e:
            status = "error"
            error = str(e)
            response_json = None
        
        duration = time.time() - start_time
        eval_session.metrics.completed_samples += 1
        eval_session.metrics.total_duration_seconds += duration
        
        await broadcast_event({
            "event_type": "sample_complete",
            "data": {
                "sample_id": sample.id,
                "status": status,
                "duration_seconds": duration,
                "error": error,
            }
        })
        
        return RolloutResult(response=response_json or "", trajectory_json=trajectory_json)
    
    ok_count, fail_count = await benchmark.rollout(runner, max_samples=...)
    
    await broadcast_event({
        "event_type": "eval_phase_transition",
        "data": {"phase": "rollout", "status": "completed"}
    })
    
    # Judge
    await broadcast_event({
        "event_type": "eval_phase_transition",
        "data": {"phase": "judge", "status": "in_progress"}
    })
    # Parse judge results and emit judge_result events
    await benchmark.judge()
    
    await broadcast_event({
        "event_type": "eval_phase_transition",
        "data": {"phase": "judge", "status": "completed"}
    })
    
    # Stat
    await benchmark.stat()
    
    await broadcast_event({
        "event_type": "eval_stat",
        "data": {
            "total_samples": eval_session.total_samples,
            "ok_count": ok_count,
            "error_count": fail_count,
            "avg_duration_seconds": eval_session.metrics.avg_duration_seconds,
            "avg_judge_score": eval_session.metrics.avg_judge_score,
        }
    })
```

## Frontend Extensions

### New Pages & Components

1. **Eval Page** (`src/agentm/server/static/js/eval.js`) — Main eval dashboard

2. **Phase Timeline** (`src/agentm/server/static/js/phase-timeline.js`) — Visual progress through preprocess → rollout → judge → stat

3. **Sample List** (`src/agentm/server/static/js/sample-list.js`) — Filterable, sortable table of samples with status, duration, judge score

4. **Sample Detail** (`src/agentm/server/static/js/sample-detail.js`) — Tabbed view: incident info, trajectory replay, judge feedback

5. **Stats Summary** (`src/agentm/server/static/js/stats-summary.js`) — Key metrics cards, judge score distribution histogram

### Modify `app.js`

Add to NAV_ITEMS:
```javascript
{ id: 'eval', icon: '📊', label: 'Eval' }
```

Add to event handlers:
```javascript
case 'eval_start':
  setScenarioState(prev => ({...prev, evalSession: event.data}));
  break;

case 'sample_complete':
  setScenarioState(prev => ({
    ...prev,
    evalSamples: [...(prev.evalSamples || []), event.data]
  }));
  break;

case 'eval_stat':
  setScenarioState(prev => ({...prev, evalStats: event.data}));
  break;
```

## Implementation Order

1. **Define events** and add docstrings to `eval.py` and middleware
2. **Create `eval_state.py`** with dataclasses
3. **Add REST endpoints** to `app.py`
4. **Emit events** from `eval.py` during pipeline execution
5. **Create eval.js** and supporting components
6. **Test WebSocket streaming** during actual eval run
7. **Polish UI** — pagination, filtering, error boundaries

## Key Architectural Points

- **Reuse existing WebSocket + REST infrastructure** — No new architectural patterns needed
- **Event-driven, not polling** — Eval events flow through same trajectory pipeline as investigation events
- **Backward compatible** — Existing pages unaffected, eval dashboard is isolated
- **Session-scoped state** — EvalSession lives for duration of eval run, then discarded
- **Sample-level granularity** — Dashboard tracks individual sample progress during rollout
- **Judge result integration** — Judge scores merged back into eval session metrics
