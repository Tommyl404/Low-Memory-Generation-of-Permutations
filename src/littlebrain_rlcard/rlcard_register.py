"""
Register custom RLCard environments.

Called automatically when ``littlebrain_rlcard`` is imported.
"""

from __future__ import annotations


def register_envs() -> None:
    """Register ``shuffle-guess`` and ``no-limit-holdem-dealerlab`` with RLCard."""
    from rlcard.envs.registration import register

    _envs = [
        (
            "shuffle-guess",
            "littlebrain_rlcard.envs.shuffle_guess.env:ShuffleGuessEnv",
        ),
        (
            "no-limit-holdem-dealerlab",
            "littlebrain_rlcard.envs.holdem_dealerlab.env_nolimit:DealerLabNolimitholdemEnv",
        ),
    ]
    for env_id, entry_point in _envs:
        try:
            register(env_id=env_id, entry_point=entry_point)
        except ValueError:
            # Already registered (e.g. multiple imports)
            pass
