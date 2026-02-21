"""
littlebrain_rlcard — Dealer algorithms (arXiv:2505.01287) integrated with RLCard.

Importing this package automatically registers the custom RLCard environments
``shuffle-guess`` and ``no-limit-holdem-dealerlab``.
"""

from littlebrain_rlcard.rlcard_register import register_envs as _register_envs

_register_envs()

__all__ = [
    "cards",
    "dealers",
    "envs",
]
