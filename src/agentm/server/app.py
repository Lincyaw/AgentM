"""FastAPI application factory for the AgentM Dashboard.

Serves the single-file HTML dashboard and provides:
- WebSocket endpoint for real-time event streaming
- REST API for topology, checkpoint history, replay, and fork
- Eval dashboard endpoints for batch evaluation monitoring
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Static HTML path
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ResumeRequest(BaseModel):
    checkpoint_id: str | None = None


class ForkRequest(BaseModel):
    checkpoint_id: str
    state_updates: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# WebSocket client management
# ---------------------------------------------------------------------------


class Broadcaster:
    """Manages WebSocket clients and broadcasts events."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    def add(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def remove(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast a WebSocket event envelope to all connected clients."""
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()
        if "mode" not in event:
            event["mode"] = "updates"

        data = event.get("data")
        if isinstance(data, dict):
            raw_messages = data.get("messages")
            if isinstance(raw_messages, list):
                event = {
                    **event,
                    "data": {
                        **data,
                        "messages": [_serialize_message(m) for m in raw_messages],
                    },
                }

        payload = json.dumps(event, default=str)
        dead: list[WebSocket] = []
        for ws in list(
            self._clients
        ):  # iterate a copy to avoid mutation during iteration
            try:
                await ws.send_text(payload)
            except (WebSocketDisconnect, ConnectionError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


# Default broadcaster instance for backward compatibility
_default_broadcaster = Broadcaster()


async def broadcast_event(event: dict[str, Any]) -> None:
    """Broadcast using the default broadcaster. Backward-compatible API."""
    await _default_broadcaster.broadcast(event)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Convert a LangChain BaseMessage to a JSON-safe dict."""
    result: dict[str, Any] = {
        "type": getattr(msg, "type", "unknown"),
        "content": getattr(msg, "content", ""),
    }
    if name := getattr(msg, "name", None):
        result["name"] = name
    if tool_calls := getattr(msg, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
            for tc in tool_calls
        ]
    if tool_call_id := getattr(msg, "tool_call_id", None):
        result["tool_call_id"] = tool_call_id
    return result


def _serialize_state_values(values: dict[str, Any]) -> dict[str, Any]:
    """Recursively serialize state values, handling LangChain messages."""
    result: dict[str, Any] = {}
    for key, val in values.items():
        if key == "messages" and isinstance(val, list):
            result[key] = [_serialize_message(m) for m in val]
        elif hasattr(val, "__dict__") and not isinstance(val, dict):
            # Dataclass / Pydantic model — try to convert
            try:
                result[key] = _serialize_value(val)
            except Exception:
                result[key] = str(val)
        else:
            result[key] = val
    return result


def _serialize_value(val: Any) -> Any:
    """Best-effort serialization of arbitrary values."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if hasattr(val, "__dataclass_fields__"):
        return asdict(val)
    if isinstance(val, BaseModel):
        return val.model_dump()
    return str(val)


def _extract_node_name(snapshot: Any) -> str:
    """Extract the node name that produced a checkpoint."""
    if hasattr(snapshot, "tasks") and snapshot.tasks:
        return snapshot.tasks[0].name
    writes = getattr(snapshot, "metadata", {}).get("writes", {})
    if writes:
        return next(iter(writes.keys()), "unknown")
    return "unknown"


def _load_ground_truth(data_dir: str) -> dict[str, Any] | None:
    """Load ground truth payload from a sample data directory if available."""
    if not data_dir:
        return None
    injection_path = Path(data_dir) / "injection.json"
    if not injection_path.exists():
        return None
    try:
        with injection_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    ground_truth = payload.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return None
    return ground_truth


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_dashboard_app(
    graph: Any | None = None,
    scenario_config: Any | None = None,
    task_manager: Any | None = None,
    trajectory: Any | None = None,
    thread_id: str | None = None,
    broadcaster: Broadcaster | None = None,
    eval_tracker: Any | None = None,
) -> FastAPI:
    """Create the FastAPI dashboard application.

    Args:
        graph: The compiled LangGraph graph (for checkpoint APIs).
        scenario_config: Parsed ScenarioConfig (for topology API).
        task_manager: TaskManager instance (for wiring WebSocket forwarding).
        trajectory: TrajectoryCollector instance (for replaying history on connect).
        thread_id: LangGraph thread_id for checkpoint APIs.
        broadcaster: Broadcaster instance for WebSocket client management.
            If ``None``, uses the module-level default broadcaster.
        eval_tracker: EvalTracker instance for batch evaluation monitoring.
            If ``None``, eval endpoints return inactive status.
    """
    if broadcaster is None:
        broadcaster = _default_broadcaster

    app = FastAPI(title="AgentM Dashboard")

    # Store references on app state for route handlers
    app.state.graph = graph
    app.state.scenario_config = scenario_config
    app.state.task_manager = task_manager
    app.state.trajectory = trajectory
    app.state.thread_id = thread_id
    app.state.broadcaster = broadcaster
    app.state.eval_tracker = eval_tracker

    # ── HTML dashboard ─────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── WebSocket ──────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        bc = app.state.broadcaster
        bc.add(websocket)
        # Replay trajectory history so refreshed clients get full state
        traj = app.state.trajectory
        if traj is not None:
            for evt in traj.events:
                try:
                    payload = json.dumps(
                        {
                            "event_type": evt.get("event_type", ""),
                            "agent_path": evt.get("agent_path", []),
                            "data": evt.get("data", {}),
                            "timestamp": evt.get("timestamp", ""),
                            "mode": "replay",
                        },
                        default=str,
                    )
                    await websocket.send_text(payload)
                except (WebSocketDisconnect, ConnectionError, RuntimeError):
                    bc.remove(websocket)
                    return
        # Send eval snapshot if eval tracker is active
        et = app.state.eval_tracker
        if et is not None:
            try:
                summary = et.get_summary()
                samples, total = et.get_samples(offset=0, limit=50)
                snapshot = json.dumps(
                    {
                        "channel": "eval",
                        "event_type": "eval_snapshot",
                        "data": {"summary": summary, "samples": samples, "total": total},
                        "timestamp": datetime.now().isoformat(),
                    },
                    default=str,
                )
                await websocket.send_text(snapshot)
            except (WebSocketDisconnect, ConnectionError, RuntimeError):
                bc.remove(websocket)
                return
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            bc.remove(websocket)

    # ── Topology API ───────────────────────────────────────────────────

    @app.get("/api/topology")
    async def get_topology() -> dict[str, Any]:
        sc = app.state.scenario_config
        if sc is None:
            return {"scenario_id": "unknown", "agents": []}
        agents = []
        for name, agent in sc.agents.items():
            agents.append(
                {
                    "agent_id": name,
                    "model": agent.model,
                    "tools": agent.tools,
                    "max_steps": agent.execution.max_steps,
                }
            )
        return {
            "scenario_id": sc.system.type,
            "agents": agents,
            "thread_id": app.state.thread_id,
        }

    # ── Task state API ─────────────────────────────────────────────────

    @app.get("/api/tasks/{thread_id}/state")
    async def get_current_state(thread_id: str) -> dict[str, Any]:
        g = app.state.graph
        if g is None:
            return {"error": "No graph available"}
        config = {"configurable": {"thread_id": thread_id}}
        state = g.get_state(config)
        return _serialize_state_values(state.values) if state else {}

    @app.get("/api/tasks/{thread_id}/history")
    async def get_checkpoint_history(thread_id: str) -> list[dict[str, Any]]:
        g = app.state.graph
        if g is None:
            return []
        config = {"configurable": {"thread_id": thread_id}}
        try:
            history = list(g.get_state_history(config))
        except Exception:
            return []
        return [
            {
                "step": s.metadata.get("step"),
                "source": s.metadata.get("source"),
                "checkpoint_id": s.config["configurable"].get("checkpoint_id", ""),
                "node_name": _extract_node_name(s),
                "next_nodes": list(s.next) if hasattr(s, "next") else [],
            }
            for s in reversed(history)
        ]

    @app.get("/api/tasks/{thread_id}/history/{checkpoint_id}")
    async def get_checkpoint_state(
        thread_id: str, checkpoint_id: str
    ) -> dict[str, Any]:
        g = app.state.graph
        if g is None:
            return {"error": "No graph available"}
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        state = g.get_state(config)
        if state is None:
            return {"error": "Checkpoint not found"}
        return {
            "step": state.metadata.get("step"),
            "source": state.metadata.get("source"),
            "node_name": _extract_node_name(state),
            "next_nodes": list(state.next) if hasattr(state, "next") else [],
            "values": _serialize_state_values(state.values),
        }

    @app.get("/api/tasks/{thread_id}/trajectory")
    async def export_trajectory(thread_id: str) -> StreamingResponse:
        g = app.state.graph
        if g is None:
            lines = [json.dumps({"error": "No graph available"}) + "\n"]
        else:
            config = {"configurable": {"thread_id": thread_id}}
            try:
                history = list(g.get_state_history(config))
            except Exception:
                history = []
            lines = [
                json.dumps(
                    {
                        "step": s.metadata.get("step"),
                        "source": s.metadata.get("source"),
                        "node_name": _extract_node_name(s),
                        "values": _serialize_state_values(s.values),
                    },
                    default=str,
                )
                + "\n"
                for s in reversed(history)
            ]

        async def generate():
            for line in lines:
                yield line

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f"attachment; filename=trajectory-{thread_id}.jsonl"
            },
        )

    # ── Debug actions ──────────────────────────────────────────────────

    @app.post("/api/tasks/{thread_id}/resume")
    async def resume_from_checkpoint(
        thread_id: str, body: ResumeRequest
    ) -> dict[str, Any]:
        g = app.state.graph
        if g is None:
            return {"error": "No graph available"}
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if body.checkpoint_id:
            config["configurable"]["checkpoint_id"] = body.checkpoint_id
        result = await g.ainvoke(None, config)
        return {
            "status": "resumed",
            "result": _serialize_state_values(result)
            if isinstance(result, dict)
            else str(result),
        }

    @app.post("/api/tasks/{thread_id}/fork")
    async def fork_from_checkpoint(thread_id: str, body: ForkRequest) -> dict[str, Any]:
        g = app.state.graph
        if g is None:
            return {"error": "No graph available"}
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": body.checkpoint_id,
            }
        }
        fork_config = g.update_state(config, body.state_updates)
        result = await g.ainvoke(None, fork_config)
        return {
            "status": "forked",
            "result": _serialize_state_values(result)
            if isinstance(result, dict)
            else str(result),
        }

    # ── Eval endpoints ────────────────────────────────────────────────

    @app.get("/api/eval/status")
    async def eval_status() -> dict[str, Any]:
        et = app.state.eval_tracker
        if et is None:
            return {"enabled": False}
        summary = et.get_summary()
        return {"enabled": True, **summary}

    @app.get("/api/eval/samples")
    async def eval_samples(
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        et = app.state.eval_tracker
        if et is None:
            return {"samples": [], "total": 0}
        samples, total = et.get_samples(
            offset=offset, limit=limit, status_filter=status, search=search
        )
        return {"samples": samples, "total": total, "offset": offset, "limit": limit}

    @app.get("/api/eval/samples/{sample_id}")
    async def eval_sample_detail(sample_id: str) -> dict[str, Any]:
        et = app.state.eval_tracker
        if et is None:
            return {"error": "Eval not active"}
        info = et.get_sample(sample_id)
        if info is None:
            return {"error": "Sample not found"}
        ground_truth = _load_ground_truth(info.get("data_dir", ""))
        if ground_truth is not None:
            info = {**info, "ground_truth": ground_truth}
            service = ground_truth.get("service")
            if isinstance(service, list):
                info["root_cause_services"] = [str(s) for s in service if str(s).strip()]
        return info

    @app.get("/api/eval/samples/{sample_id}/events")
    async def eval_sample_events(
        sample_id: str, after: int = 0
    ) -> dict[str, Any]:
        """Read trajectory events from JSONL file for a specific sample.

        Args:
            after: Skip the first ``after`` events (0-based). When the client
                already has N events, pass ``after=N`` to receive only new ones.
        """
        et = app.state.eval_tracker
        if et is None:
            return {"error": "Eval not active", "events": [], "total": 0}
        info = et.get_sample(sample_id)
        if info is None:
            return {"error": "Sample not found", "events": [], "total": 0}
        traj_path = info.get("trajectory_path")
        if not traj_path:
            return {
                "events": [],
                "total": 0,
                "status": info.get("status", "unknown"),
            }
        path = Path(traj_path)
        if not path.exists():
            return {
                "events": [],
                "total": 0,
                "status": info.get("status", "unknown"),
            }
        events: list[dict[str, Any]] = []
        event_index = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Skip metadata lines
                    if "_meta" in parsed:
                        continue
                    if event_index >= after:
                        events.append(parsed)
                    event_index += 1
        except OSError:
            pass
        return {
            "events": events,
            "total": event_index,
            "status": info.get("status", "unknown"),
        }

    return app
