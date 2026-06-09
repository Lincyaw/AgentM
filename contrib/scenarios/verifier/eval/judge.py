"""Judge agent: review all hop verdicts for a completed case via SDK."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

from graph import SYNTHETIC, _duckdb_conn
from injection import get_injections
from verdict import collect_all_verdicts, extract_hop_verdict

REPO = Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Lazy import of the prompt builder from the verifier_judge scenario package.
# ---------------------------------------------------------------------------

_prompt_module: ModuleType | None = None


def _get_prompt_module() -> ModuleType:
    global _prompt_module  # noqa: PLW0603
    if _prompt_module is not None:
        return _prompt_module
    prompt_path = REPO / "contrib" / "scenarios" / "verifier_judge" / "prompt.py"
    spec = importlib.util.spec_from_file_location("verifier_judge.prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prompt module from {prompt_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _prompt_module = mod
    return mod


def _get_system_throughput(data_dir: Path) -> dict:
    """Compare load-generator root span counts between windows."""
    conn = _duckdb_conn(data_dir)
    result: dict = {}
    try:
        for w in ("normal", "abnormal"):
            row = conn.execute(
                f"SELECT COUNT(*) FROM {w}_traces "
                "WHERE (parent_span_id = '' OR parent_span_id IS NULL) "
                f"AND service_name IN {str(tuple(SYNTHETIC))}"
            ).fetchone()
            result[w] = row[0] if row else 0
    except Exception:  # noqa: BLE001
        pass
    conn.close()
    return result


# ---------------------------------------------------------------------------
# SDK helpers (shared with hop.py)
# ---------------------------------------------------------------------------


def _resolve_provider() -> tuple[str, dict[str, object]]:
    """Build a provider spec from the environment (config.toml profile)."""
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib.user_config import resolve_model_profile

    model_name = os.environ.get("AGENTM_MODEL")
    profile = resolve_model_profile(model_name)
    if profile is not None:
        build_config = profile.to_build_config()
        provider_id = os.environ.get("AGENTM_PROVIDER") or profile.provider
    else:
        registry = DEFAULT_PROVIDER_REGISTRY
        provider_id = os.environ.get("AGENTM_PROVIDER") or registry.default_provider().id
        build_config = {"model": model_name or registry.default_model(provider_id)}

    return DEFAULT_PROVIDER_REGISTRY.build(provider_id, build_config)


def _extract_verdict_from_messages(
    messages: list,  # type: ignore[type-arg]
    require_key: str = "remove",
) -> dict | None:
    """Extract the judge verdict from the session's final messages."""
    for msg in reversed(messages):
        if getattr(msg, "role", None) != "tool_result":
            continue
        for block in getattr(msg, "content", []):
            if getattr(block, "type", None) != "tool_result":
                continue
            if getattr(block, "is_error", False):
                continue
            for inner in getattr(block, "content", []):
                if getattr(inner, "type", None) != "text":
                    continue
                text = getattr(inner, "text", "")
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(obj, dict) and require_key in obj:
                    return obj
    return None


async def _run_judge_async(
    data_dir: Path,
    judge_dir: Path,
    prompt: str,
    budget: int,
) -> dict | None:
    """Run a judge session via the SDK and return the review verdict."""
    from agentm.core.abi import LoopConfig
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    os.environ["AGENTM_PROJECT_ROOT"] = str(REPO)
    os.environ["AGENTM_RCA_DATA_DIR"] = str(data_dir)

    provider_spec = _resolve_provider()
    config = AgentSessionConfig(
        cwd=str(judge_dir),
        provider=provider_spec,
        scenario="verifier_judge",
        loop_config=LoopConfig(max_tool_calls=budget),
        auto_commit=False,
    )
    session = await AgentSession.create(config)
    try:
        messages = await session.prompt(prompt)
        return _extract_verdict_from_messages(messages)
    finally:
        await session.shutdown()


def run_judge(
    case_dir: Path,
    run_dir: Path,
    *,
    budget: int = 20,
) -> dict:
    """Run a judge agent on a completed case to review all hop verdicts."""
    data_dir = case_dir.resolve()
    out = run_dir.resolve()

    # Collect inputs
    all_verdicts_path = out / "all_verdicts.json"
    if not all_verdicts_path.exists():
        all_verdicts = collect_all_verdicts(out)
        all_verdicts_path.write_text(
            json.dumps(all_verdicts, indent=2, ensure_ascii=False)
        )
    else:
        all_verdicts = json.loads(all_verdicts_path.read_text())

    trace = json.loads((out / "propagation_trace.json").read_text())
    confirmed = trace.get("confirmed_nodes", [])
    injections = get_injections(data_dir)
    throughput = _get_system_throughput(data_dir)

    seeds = {i["target"] for i in injections}

    # Index hop verdicts by target for evidence lookup.
    verdict_by_target: dict[str, dict] = {v["to"]: v for v in all_verdicts}

    # Collect rejected verdicts for prompt
    rejected_verdicts = [
        v for v in all_verdicts
        if v["verdict"] == "rejected" and v["to"] not in confirmed
    ]

    # Build the judge prompt via the scenario prompt module
    mod = _get_prompt_module()
    prompt = mod.build_judge_prompt(
        injections=injections,
        confirmed=confirmed,
        rejected_verdicts=rejected_verdicts,
        throughput=throughput,
        seeds=seeds,
        verdict_by_target=verdict_by_target,
    )

    judge_dir = out / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    # Run the judge via SDK
    try:
        verdict = asyncio.run(
            _run_judge_async(data_dir, judge_dir, prompt, budget)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    judge sdk-error: {exc}")
        verdict = None

    # Fall back to JSONL extraction if SDK extraction missed
    if verdict is None:
        obs_dir = judge_dir / ".agentm" / "observability"
        verdict = (
            extract_hop_verdict(obs_dir, "submit_judge_review", "remove")
            if obs_dir.exists() else None
        )

    if verdict:
        (judge_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False)
        )

    confirmed_set = set(confirmed)
    # Judge is promotion-only: hop confirmations are authoritative.
    suggested_remove = sorted(({
        s.strip() for s in (verdict.get("remove", []) if verdict else [])
        if isinstance(s, str) and s.strip()
    } & confirmed_set) - seeds)
    removed: set[str] = set()
    # add: rejected services to promote, not already confirmed, not seeds.
    added = {
        s.strip() for s in (verdict.get("add", []) if verdict else [])
        if isinstance(s, str) and s.strip()
    }
    added -= confirmed_set
    added -= seeds

    final_confirmed = sorted((confirmed_set - removed) | added)
    final_propagated = [s for s in final_confirmed if s not in seeds]
    judge_rationale = verdict.get("rationale", "") if verdict else ""

    # Build per-node entry with provenance + evidence
    nodes: list[dict] = []
    for svc in final_confirmed:
        if svc in seeds:
            nodes.append({"service": svc, "source": "injection_seed"})
        elif svc in added:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "judge_promoted",
                "hop_verdict": hop_v.get("verdict", ""),
                "hop_rationale": hop_v.get("rationale", ""),
                "hop_evidence": hop_v.get("symptom_evidence", []),
                "judge_rationale": judge_rationale,
            })
        else:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "hop_agent",
                "rationale": hop_v.get("rationale", ""),
                "symptom_evidence": hop_v.get("symptom_evidence", []),
            })

    # Build edges, dropping any whose target the judge removed.
    final_set = set(final_confirmed)
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    for h in trace.get("hop_log", []):
        frm, to = h.get("from", ""), h.get("to", "")
        if not frm or not to or to not in final_set:
            continue
        v = h.get("verdict", "")
        if v in ("confirmed", "edge_sql") and (frm, to) not in seen_edges:
            seen_edges.add((frm, to))
            hop_v = verdict_by_target.get(to, {})
            edge_entry: dict = {
                "from": frm, "to": to,
                "source": "hop_agent" if v == "confirmed" else "edge_sql",
            }
            if v == "confirmed":
                edge_entry["rationale"] = hop_v.get("rationale", "")
                edge_entry["symptom_evidence"] = hop_v.get(
                    "symptom_evidence", [],
                )
            edges.append(edge_entry)

    # Edges for judge-promoted nodes
    for svc in added:
        promoted_v = verdict_by_target.get(svc)
        if promoted_v and (promoted_v["from"], svc) not in seen_edges:
            seen_edges.add((promoted_v["from"], svc))
            edges.append({
                "from": promoted_v["from"], "to": svc,
                "source": "judge_promoted",
                "hop_rationale": promoted_v.get("rationale", ""),
                "hop_evidence": promoted_v.get("symptom_evidence", []),
            })

    final = {
        "seeds": sorted(seeds),
        "confirmed_nodes": final_confirmed,
        "propagated": final_propagated,
        "edges": edges,
        "nodes": nodes,
        "judge_rationale": judge_rationale,
        "removed": sorted(removed),
        "suggested_remove": suggested_remove,
        "added": sorted(added),
    }
    (out / "final_propagation.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False)
    )

    return verdict or {}
