"""Deployment topology graph tool."""

from __future__ import annotations

import json

from agentm.tools.observability._builders import _resolve_file
from agentm.tools.observability._core import _enforce_token_limit, _query, _safe_tool


@_safe_tool
async def get_deployment_graph(
    service_name: str | None = None,
) -> str:
    """Infrastructure deployment topology: which services run on which pods and nodes.

    Extracts service -> deployment -> pod -> node relationships from metrics
    resource attributes during the abnormal period.

    Args:
        service_name: Optional filter to show only this service's deployment.

    Returns:
        JSON list of unique deployment records, each with
        service, deployment, pod, node, namespace.
    """
    file = _resolve_file("metrics", "abnormal")

    k8s_filter = '"attr.k8s.pod.name" IS NOT NULL'
    where = f" WHERE {k8s_filter}"
    params: list[object] = []
    if service_name:
        where += " AND service_name = ?"
        params.append(service_name)

    sql = f"""
        SELECT DISTINCT
            service_name AS service,
            "attr.k8s.deployment.name" AS deployment,
            "attr.k8s.pod.name" AS pod,
            "attr.k8s.node.name" AS node,
            "attr.k8s.namespace.name" AS namespace
        FROM read_parquet('{file}'){where}
        ORDER BY service, deployment, pod
    """
    rows = _query(sql, params)

    services_with_pods = {r["service"] for r in rows}
    deploy_where = ' WHERE "attr.k8s.deployment.name" IS NOT NULL AND "attr.k8s.pod.name" IS NULL'
    deploy_params: list[object] = []
    if service_name:
        deploy_where += " AND service_name = ?"
        deploy_params.append(service_name)

    deploy_sql = f"""
        SELECT DISTINCT
            service_name AS service,
            "attr.k8s.deployment.name" AS deployment,
            NULL AS pod, NULL AS node,
            "attr.k8s.namespace.name" AS namespace
        FROM read_parquet('{file}'){deploy_where}
    """
    deploy_rows = _query(deploy_sql, deploy_params)
    for r in deploy_rows:
        if r["service"] not in services_with_pods:
            rows.append(r)

    if not rows:
        return json.dumps({"warning": "No Kubernetes deployment info found in metrics data."})
    rows = [{k: v for k, v in r.items() if v is not None and v != ""} for r in rows]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return _enforce_token_limit(payload, "get_deployment_graph")
