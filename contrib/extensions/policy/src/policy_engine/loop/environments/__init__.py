"""Environment adapters: how the machine an attempt ran on is brought back.

Kept apart from the benchmark adapters because the two are orthogonal. One
benchmark can run on a sandbox cluster or in a local container, and one cluster
can host several benchmarks.
"""
