"""Regression tests for headless structured response normalization."""

from __future__ import annotations

from agentm.cli.run import _normalize_structured_response


def test_normalize_structured_response_coerces_empty_raw_text_to_graph_shape() -> None:
    """raw_text fallback that cannot be parsed still yields graph-compatible payload."""
    normalized = _normalize_structured_response({"raw_text": ""})

    assert normalized == {
        "nodes": [],
        "edges": [],
        "root_causes": [],
        "component_to_service": {},
    }


def test_normalize_structured_response_unwraps_and_converts_mapping() -> None:
    """Valid graph JSON in raw_text is unwrapped and mapping is converted to dict."""
    raw = (
        '{"nodes":[{"component":"svc-a","state":[],"timestamp":""}],'
        '"edges":[],"root_causes":[],"component_to_service":'
        '[{"component_name":"pod-a","service_name":"svc-a"}]}'
    )
    normalized = _normalize_structured_response({"raw_text": raw})

    assert normalized["nodes"][0]["component"] == "svc-a"
    assert normalized["edges"] == []
    assert normalized["root_causes"] == []
    assert normalized["component_to_service"] == {"pod-a": "svc-a"}
