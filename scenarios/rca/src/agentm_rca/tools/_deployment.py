"""Deployment topology graph tool."""

from __future__ import annotations

import json

from ._builders import _resolve_file
from ._core import _query, enforce_token_budget, obs_safe_tool


@obs_safe_tool
async def get_deployment_graph(
    request: str,
    service_name: str | None = None,
) -> str:
    del request
    file = _resolve_file("metrics", "abnormal")

    where = ' WHERE "attr.k8s.pod.name" IS NOT NULL'
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

    services_with_pods = {row["service"] for row in rows}
    deploy_where = (
        ' WHERE "attr.k8s.deployment.name" IS NOT NULL AND "attr.k8s.pod.name" IS NULL'
    )
    deploy_params: list[object] = []
    if service_name:
        deploy_where += " AND service_name = ?"
        deploy_params.append(service_name)

    deploy_sql = f"""
        SELECT DISTINCT
            service_name AS service,
            "attr.k8s.deployment.name" AS deployment,
            NULL AS pod,
            NULL AS node,
            "attr.k8s.namespace.name" AS namespace
        FROM read_parquet('{file}'){deploy_where}
    """
    for row in _query(deploy_sql, deploy_params):
        if row["service"] not in services_with_pods:
            rows.append(row)

    if not rows:
        return json.dumps(
            {"warning": "No Kubernetes deployment info found in metrics data."}
        )

    compact_rows = [
        {key: value for key, value in row.items() if value is not None and value != ""}
        for row in rows
    ]
    payload = json.dumps(compact_rows, ensure_ascii=False, separators=(",", ":"))
    return enforce_token_budget(payload, "get_deployment_graph")
