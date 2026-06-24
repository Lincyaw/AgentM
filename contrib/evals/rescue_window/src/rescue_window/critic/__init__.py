"""Bounded critic baselines (doc §3.4 / E4).

A critic maps the visible prefix h_t to one intervention (or abstains) and is
just one column in the experiment matrix — evaluated through the same harness and
read back as G^C_t. llmharness is deliberately not a dependency.
"""

from .critics import AbstainCritic, AlwaysVerifyCritic, Critic, as_treatment

__all__ = ["AbstainCritic", "AlwaysVerifyCritic", "Critic", "as_treatment"]
