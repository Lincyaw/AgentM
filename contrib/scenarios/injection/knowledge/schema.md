# Verifier-Compatible Injection Case Library

The injection workflow consumes historical cases in a shape that should be
compatible with verifier outputs. Injection does not own propagation truth: it
uses verifier-produced or verifier-like evidence to choose harder future faults.
When verifier artifacts are unavailable, the injection worker may write a
`simulated` evidence record as a temporary proxy; later verifier imports should
supersede or calibrate it.

## Location

Per-system campaign state may keep a JSONL case library at:

```text
~/.aegisctl/injection-author/<system>/case_library.jsonl
```

Each line is one case record. Records may come from completed injection rounds,
verifier exports, or temporary injection-author simulations.

## Minimal Required Fields for Verifier Producers

Verifier-side producers only need to emit these fields for injection to use the
record:

```json
{
  "case_id": "033437be-da7b-41f3-ad91-ecbb3d6dd485",
  "system": "ts",
  "fault": {
    "family": "http",
    "chaos_type": "HTTPResponseDelay",
    "service": "ts-order-service",
    "scope": "POST /api/v1/orderservice/order"
  },
  "observations": {
    "detector_verdict": "fired",
    "slo_impact": "violated",
    "injection_landed": true,
    "entrypoint_symptoms": []
  },
  "verifier_evidence": {
    "status": "verified",
    "propagation_path": ["ui-dashboard", "ts-preserve-service", "ts-order-service"],
    "root_cause": {
      "service": "ts-order-service",
      "scope": "POST /api/v1/orderservice/order",
      "fault_type": "HTTPResponseDelay"
    },
    "decoy_hypotheses": ["ts-payment-service", "mysql"],
    "attribution_confidence": "medium",
    "reasoning_summary": "Checkout latency propagates through preserve into order creation."
  }
}
```

Accepted `verifier_evidence.status` values:

- `verified`: produced by verifier or an equivalent offline grader.
- `simulated`: temporary injection-author proxy, not authoritative.
- `pending`: trace exists but verifier data has not landed yet.
- `failed`: verifier/platform failure; preserve the reason in `reasoning_summary`.

## Optional Injection-Control Fields

The injection workflow may enrich records with planning metadata:

```json
{
  "puzzle_score": {
    "estimated_difficulty": "medium",
    "inference_chain_length": 3,
    "decoy_count": 2,
    "rca_solved": null
  },
  "reuse_guidance": {
    "reuse_shape": true,
    "avoid_exact_repeat": true,
    "suggested_variations": ["same path with a different chaos family"]
  },
  "source": {
    "producer": "injection-workflow",
    "artifact_id": "round-167-1782618033.json"
  }
}
```

## Planning Semantics

The injection planner should treat `verified` evidence as authoritative,
`simulated` evidence as a weak prior, and `pending` evidence as unavailable for
positive claims. Hard-case reuse should vary at least one of family, service,
scope, or path segment to avoid exact-repeat memorization.
