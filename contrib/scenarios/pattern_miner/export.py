# code-health: ignore-file[AM025] -- host-side exporter normalizes untyped trial JSON and trajectory blocks
"""Export one Harbor trial into a mining case bundle (host-side).

Reads trial artifacts from a jobs directory, renders the agent trajectory
from the batch's trajectory store, and writes the bundle that the
``case_tools`` atom serves: task.md, trajectory.jsonl, eval.json,
agent.patch, oracle.patch, meta.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger

from agentm.core.abi import TextContent, ToolCallBlock
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import Turn


def read_reward(trial_dir: Path) -> tuple[float | None, str | None]:
    """Cheap reward read for selection: (parsed reward, raw reward.txt)."""

    reward_path = trial_dir / "verifier" / "reward.txt"
    if not reward_path.is_file():
        return None, None
    raw = reward_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None, raw
    try:
        return float(raw), raw
    except ValueError:
        return None, raw


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("export: cannot parse {}: {}", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _render_records(turns: list[Turn]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for turn in turns:
        if turn.response is not None:
            text_parts: list[str] = []
            calls: list[dict[str, object]] = []
            for block in turn.response.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolCallBlock):
                    calls.append(
                        {"name": block.name, "arguments": dict(block.arguments)}
                    )
            records.append(
                {
                    "turn": turn.index,
                    "role": "assistant",
                    "text": "\n".join(part for part in text_parts if part),
                    "tool_calls": calls,
                }
            )
        for tool_result in turn.tool_results:
            text = ""
            is_error = False
            if tool_result.result is not None:
                is_error = bool(tool_result.result.is_error)
                for content in tool_result.result.content:
                    if isinstance(content, TextContent):
                        text += content.text
            records.append(
                {
                    "turn": turn.index,
                    "role": "tool_result",
                    "tool": tool_result.call.name,
                    "text": text,
                    "is_error": is_error,
                }
            )
    return records


def export_case(
    trial_dir: Path,
    bundle_dir: Path,
    *,
    store: TrajectoryStore | None,
) -> dict[str, object]:
    """Write the case bundle for one trial; return its meta record."""

    config = _load_json(trial_dir / "config.json")
    result = _load_json(trial_dir / "result.json")
    verifier_dir = trial_dir / "verifier"

    task_section = config.get("task")
    task_path: Path | None = None
    if isinstance(task_section, dict) and task_section.get("path"):
        task_path = Path(str(task_section["path"]))
    task_name = str(
        result.get("task_name") or (task_path.name if task_path else trial_dir.name)
    )
    trial_name = str(result.get("trial_name") or trial_dir.name)

    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    session_id: str | None = None
    if isinstance(metadata, dict) and metadata.get("agentm_session_id"):
        session_id = str(metadata["agentm_session_id"])

    agent_section = config.get("agent")
    agent_model = (
        str(agent_section.get("model_name", ""))
        if isinstance(agent_section, dict)
        else ""
    )

    bundle_dir.mkdir(parents=True, exist_ok=True)

    instruction = ""
    if task_path is not None and (task_path / "instruction.md").is_file():
        instruction = (task_path / "instruction.md").read_text(encoding="utf-8")
    else:
        logger.warning("export: no instruction.md for {}", trial_name)
    (bundle_dir / "task.md").write_text(
        instruction or "(instruction.md not found)", encoding="utf-8"
    )

    reward, reward_txt = read_reward(trial_dir)
    g0_reasons: list[str] = []
    if reward_txt is None:
        g0_reasons.append("reward.txt missing")
    elif not reward_txt:
        g0_reasons.append("reward.txt empty")
    elif reward is None:
        g0_reasons.append(f"reward.txt unparseable: {reward_txt!r}")
    exception_info = result.get("exception_info")
    if exception_info:
        g0_reasons.append("harness exception_info present")

    eval_payload: dict[str, object] = {
        "task": task_name,
        "trial": trial_name,
        "reward": reward,
        "reward_txt": reward_txt,
        "reward_details": _load_json(verifier_dir / "reward_details.json"),
        "verifier_results": _load_json(verifier_dir / "verifier_results.json"),
        "judge_output": _load_json(verifier_dir / "judge_output.json"),
        "exception_info": exception_info,
        "g0_reasons": g0_reasons,
    }
    (bundle_dir / "eval.json").write_text(
        json.dumps(eval_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    for source_name, bundle_name in (
        ("agent.patch", "agent.patch"),
        ("oracle_filtered.patch", "oracle.patch"),
    ):
        source = verifier_dir / source_name
        if source.is_file():
            shutil.copyfile(source, bundle_dir / bundle_name)
        else:
            (bundle_dir / bundle_name).write_text(
                f"({source_name} not found)\n", encoding="utf-8"
            )

    trajectory_error: str | None = None
    record_count = 0
    if session_id is None:
        trajectory_error = "no agentm_session_id in result.json"
    elif store is None:
        trajectory_error = "no trajectory store configured"
    else:
        try:
            if not store.session_exists(session_id):
                trajectory_error = f"session {session_id} not found in store"
            else:
                _, turns = store.load(session_id)
                records = _render_records(turns)
                record_count = len(records)
                with (bundle_dir / "trajectory.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for record in records:
                        handle.write(
                            json.dumps(record, ensure_ascii=False, default=str) + "\n"
                        )
        except Exception as exc:  # noqa: BLE001
            trajectory_error = f"store read failed: {exc}"
    if trajectory_error is not None:
        (bundle_dir / "trajectory.jsonl").write_text("", encoding="utf-8")
        logger.warning("export: {}: {}", trial_name, trajectory_error)

    meta: dict[str, object] = {
        "task_name": task_name,
        "trial_name": trial_name,
        "session_id": session_id,
        "agent_model": agent_model,
        "task_path": str(task_path) if task_path is not None else None,
        "reward": reward,
        "g0_reasons": g0_reasons,
        "trajectory_records": record_count,
        "trajectory_error": trajectory_error,
    }
    (bundle_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return meta
