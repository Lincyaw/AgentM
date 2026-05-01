from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm_rca import build_observability_tools
from agentm_rca.tools import MANIFEST


@pytest.mark.asyncio
async def test_build_observability_tools_executes_metrics_query(
    observability_data_dir: Path,
) -> None:
    tools = build_observability_tools(data_dir=str(observability_data_dir))

    tool_names = {tool.name for tool in tools}
    assert "query_metrics_ohlc_abnormal" in tool_names
    assert "search_logs_abnormal" in tool_names
    assert MANIFEST.name == "rca_observability"

    metrics_tool = next(tool for tool in tools if tool.name == "query_metrics_ohlc_abnormal")
    result = await metrics_tool.execute(
        {
            "metric_name": "cpu.utilization",
            "interval": "5m",
            "filters": json.dumps({"service_name": "checkout"}),
        }
    )

    assert result.is_error is False
    assert len(result.content) == 1

    payload = json.loads(result.content[0].text)
    assert isinstance(payload, list)
    assert payload == [
        {
            "time_bucket": "2025-08-28T20:45:00",
            "open": 1.0,
            "high": 3.0,
            "low": 1.0,
            "close": 3.0,
            "count": 2,
        }
    ]
