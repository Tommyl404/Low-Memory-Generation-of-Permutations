"""
Integration tests — RLCard no-limit-holdem-dealerlab environment.

Tests that the wrapped holdem env runs and produces deterministic results
under seed.
"""

from __future__ import annotations

import pytest
import rlcard
from rlcard.agents import RandomAgent


@pytest.mark.parametrize("dealer_algo", ["bitmap", "fisher_yates", "adaptive", "perfect"])
def test_holdem_determinism(dealer_algo: str) -> None:
    """Same seed → identical payoffs for holdem-dealerlab."""
    seed = 42
    results = []

    for _ in range(2):
        config = {
            "seed": seed,
            "game_num_players": 2,
            "game_dealer_algo": dealer_algo,
            "game_m_bits": 32,
        }
        env = rlcard.make("no-limit-holdem-dealerlab", config=config)
        rlcard.utils.set_seed(seed)
        env.set_agents([
            RandomAgent(num_actions=env.num_actions),
            RandomAgent(num_actions=env.num_actions),
        ])
        _, payoffs = env.run(is_training=False)
        results.append(payoffs)

    assert results[0].tolist() == results[1].tolist(), (
        f"Payoffs differ: {results[0]} vs {results[1]}"
    )


def test_holdem_basic_run() -> None:
    """Smoke test: holdem-dealerlab runs without crashing."""
    env = rlcard.make("no-limit-holdem-dealerlab", config={
        "seed": 0,
        "game_num_players": 2,
        "game_dealer_algo": "fisher_yates",
    })
    env.set_agents([
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
    ])
    rlcard.utils.set_seed(0)
    _, payoffs = env.run(is_training=False)
    assert payoffs.shape == (2,)
    # Zero-sum check (NL Holdem payoffs should sum to 0)
    assert abs(payoffs.sum()) < 1e-6
