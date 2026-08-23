"""Shared helpers for observability extension implementations.

This package intentionally sits outside ``agentm.core``: OTel exporters,
collector discovery, local trace files, and operational-log bridging are
backend/policy implementations, not SDK substrate invariants.
"""
