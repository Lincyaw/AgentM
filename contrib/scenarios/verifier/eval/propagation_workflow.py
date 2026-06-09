# ruff: noqa
# mypy: ignore-errors
"""Pre-written workflow script for fault propagation verification.

Pure orchestration: BFS over the neighbor graph, parallel hop agents,
optional judge phase. All prompt/domain logic is encapsulated in the
agent units (verifier/hop and verifier/judge scenarios with their
context atoms). The workflow passes structured data via atom_config.

Runs in the workflow atom's curated namespace (agent, parallel, args,
json, log, phase, set, dict, list, sorted).

Input via ``args``:
    data_dir, graph, injections, infra_nodes, node_map,
    target_evidence, fault_docs, budget, out_dir, skip_judge

Output (return value): dict with confirmed_nodes, edges, node_evidence,
hop_log, rounds, judge (if run).
"""

skip_propagate = args.get("skip_propagate", False)
injections = args["injections"]

# When skip_propagate=True, load existing propagation results and jump to judge
if skip_propagate:
    existing = args.get("existing_propagation", {})
    confirmed = set(existing.get("confirmed_nodes", []))
    node_evidence = existing.get("node_evidence", {})
    edges = existing.get("edges", [])
    hop_log = existing.get("hop_log", [])
    round_n = existing.get("rounds", 0)
    node_fault = {}
    for inj in injections:
        target = inj["target"]
        if target:
            node_fault[target] = [inj["chaos_type"], target]
else:
    phase("propagate")
    confirmed = set()
    node_evidence = {}
    node_fault = {}
    edges = []
    hop_log = []
graph = args["graph"]
infra_nodes_list = args.get("infra_nodes", [])
infra_set = set(infra_nodes_list)
node_map = args.get("node_map", {})
data_dir = args["data_dir"]
out_dir = args["out_dir"]
hop_budget = args.get("budget", 15)
skip_judge = args.get("skip_judge", False)
fault_docs = args.get("fault_docs", {})

# Full fault set as lists (JSON-safe), handed to every hop agent
all_faults = [
    [inj["chaos_type"], inj["target"], inj.get("params", "")]
    for inj in injections
    if inj.get("target")
]

if not skip_propagate:
    # Seed from injections
    for inj in injections:
        target = inj["target"]
        if target:
            confirmed.add(target)
            ev = args.get("target_evidence", {}).get(target, {})
            node_evidence[target] = {"source": "injection_target"}
            for k, v in ev.items():
                node_evidence[target][k] = v
            node_fault[target] = [inj["chaos_type"], target]

    queue = list(confirmed)
    checked_edges = set()
    round_n = 0

while queue and not skip_propagate:
    round_n += 1
    batch = list(queue)
    queue = []

    pending_hops = []
    sql_edges = []

    for current in batch:
        for neighbor_info in graph.get(current, []):
            neighbor = neighbor_info[0]
            rel_type = neighbor_info[1]
            edge_key = current + "__" + neighbor
            if edge_key in checked_edges:
                continue
            checked_edges.add(edge_key)

            if neighbor in confirmed:
                sql_edges.append([current, neighbor])
            else:
                pending_hops.append([current, neighbor, rel_type])

    # Record SQL-verified edges between already-confirmed nodes
    for pair in sql_edges:
        edges.append(pair)
        hop_log.append({
            "round": round_n,
            "from": pair[0],
            "to": pair[1],
            "verdict": "edge_sql",
        })
        log("  edge (SQL): " + pair[0] + " -> " + pair[1])

    # co_deployed pre-check placeholder
    if node_map and pending_hops:
        pending_hops = [
            h for h in pending_hops if h[1] not in confirmed
        ]

    if not pending_hops:
        continue

    log("Round " + str(round_n) + ": " + str(len(pending_hops)) + " hops")

    # Run hops in parallel via autonomous agents
    async def _make_hop_coro(from_svc, to_svc, rel_type):
        hop_fault = node_fault.get(
            from_svc, [all_faults[0][0], all_faults[0][1]]
        )
        is_infra = to_svc in infra_set
        result = await agent(
            "Verify this propagation edge.",
            scenario="verifier/hop",
            atom_config={
                "hop_context": {
                    "from_service": from_svc,
                    "to_service": to_svc,
                    "rel_type": rel_type,
                    "fault_kind": hop_fault[0],
                    "injection_target": hop_fault[1],
                    "all_faults": all_faults,
                    "fault_docs": fault_docs,
                    "is_infra": is_infra,
                    "upstream_evidence": node_evidence.get(from_svc),
                },
                "hop_finalize": {"data_dir": data_dir},
            },
        )
        try:
            return json.loads(result)
        except Exception:
            return None

    coros = []
    for item in pending_hops:
        coros.append(_make_hop_coro(item[0], item[1], item[2]))

    results = await parallel(coros)

    for idx in range(len(pending_hops)):
        from_svc = pending_hops[idx][0]
        to_svc = pending_hops[idx][1]
        result = results[idx]
        verdict = result.get("verdict") if isinstance(result, dict) else None
        hop_log.append({
            "round": round_n,
            "from": from_svc,
            "to": to_svc,
            "verdict": verdict if verdict else "no-result",
        })
        log("  " + from_svc + " -> " + to_svc + ": " + (verdict if verdict else "no-result"))

        if verdict == "confirmed" and to_svc not in confirmed:
            confirmed.add(to_svc)
            ev = {"source": "hop_agent"}
            if isinstance(result, dict):
                for k, v in result.items():
                    ev[k] = v
            node_evidence[to_svc] = ev
            if from_svc in node_fault:
                node_fault[to_svc] = node_fault[from_svc]
            if to_svc not in infra_set:
                queue.append(to_svc)
            edges.append([from_svc, to_svc])

propagation_result = {
    "confirmed_nodes": sorted(confirmed),
    "edges": edges,
    "node_evidence": node_evidence,
    "hop_log": hop_log,
    "rounds": round_n,
}

# -- Judge phase (optional) --
judge_result = None
if not skip_judge and len(confirmed) > len(injections):
    phase("judge")

    seeds = set()
    for inj in injections:
        seeds.add(inj["target"])

    # Build verdict_by_target from existing_verdicts (judge-only) or hop_log
    existing_verdicts = args.get("existing_verdicts", [])
    verdict_by_target = {}
    if existing_verdicts:
        for v in existing_verdicts:
            verdict_by_target[v.get("to", "")] = v
    for entry in hop_log:
        to_svc = entry.get("to", "")
        if to_svc and to_svc not in verdict_by_target:
            ev = node_evidence.get(to_svc, {})
            verdict_by_target[to_svc] = {
                "from": entry.get("from", ""),
                "to": to_svc,
                "verdict": entry.get("verdict", ""),
                "rationale": ev.get("rationale", ""),
                "symptom_evidence": ev.get("symptom_evidence", []),
            }

    rejected_verdicts = [
        v for v in verdict_by_target.values()
        if v.get("verdict") == "rejected" and v["to"] not in confirmed
    ]

    judge_text = await agent(
        "Review the fault-propagation graph.",
        scenario="verifier/judge",
        atom_config={
            "judge_context": {
                "injections": injections,
                "confirmed": sorted(confirmed),
                "rejected_verdicts": rejected_verdicts,
                "throughput": {},
                "seeds": sorted(seeds),
                "verdict_by_target": verdict_by_target,
            },
        },
    )
    try:
        judge_result = json.loads(judge_text)
    except Exception:
        judge_result = None

    if judge_result:
        propagation_result["judge"] = judge_result

return propagation_result
