"""OTel infrastructure for session observability.

File-backed OTLP exporters, event-to-OTel translators, the OTel dispatch
registry, OTLP parsing utilities, and message redaction for privacy.
Atoms reach the dispatch entry point via
:func:`agentm.core.observability.otel_dispatch.dispatch_otel`; everything
else is internal to the substrate.
"""
