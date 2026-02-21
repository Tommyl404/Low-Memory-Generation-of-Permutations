"""
Integration tests — RLCard shuffle-guess environment.

Tests determinism under seed: same config yields identical payoffs and drawn_ids.
"""

from __future__ import annotations

import pytest
import rlcard
from rlcard.agents import RandomAgent

_DEALER_CONFIGS = [
    {"game_dealer": "bitmap"},
    {"game_dealer": "fisher_yates"},
    {"game_dealer": "adaptive", "game_m_bits": 64},
    {"game_dealer": "adaptive", "game_m_bits": 16},
    {"game_dealer": "perfect"},
]


@pytest.mark.parametrize("extra_config", _DEALER_CONFIGS)
def test_determinism(extra_config: dict) -> None:
    """Same seed → identical payoffs and drawn_ids."""
    seed = 42
    results = []

    for _ in range(2):
        config = {
            "seed": seed,
            "game_n_cards": 104,
            **extra_config,
        }
        env = rlcard.make("shuffle-guess", config=config)
        rlcard.utils.set_seed(seed)
        env.set_agents([RandomAgent(num_actions=env.num_actions)])
        _, payoffs = env.run(is_training=False)
        info = env.get_perfect_information()
        results.append((payoffs, info))

    payoffs1, info1 = results[0]
    payoffs2, info2 = results[1]

    assert payoffs1.tolist() == payoffs2.tolist(), (
        f"Payoffs differ: {payoffs1} vs {payoffs2}"
    )
    assert info1["drawn_ids"][:10] == info2["drawn_ids"][:10], (
        "First 10 drawn_ids differ"
    )
    assert info1["drawn_ids"] == info2["drawn_ids"]


def test_env_basic_run() -> None:
    """Basic smoke test: env runs without crashing."""
    env = rlcard.make("shuffle-guess", config={
        "seed": 0,
        "game_dealer": "fisher_yates",
        "game_n_cards": 52,
    })
    env.set_agents([RandomAgent(num_actions=env.num_actions)])
    rlcard.utils.set_seed(0)
    trajectories, payoffs = env.run(is_training=False)
    assert payoffs.shape == (1,)
    assert payoffs[0] >= 0


def test_env_score_nonnegative() -> None:
    """Score must be non-negative."""
    for dealer_name in ["bitmap", "fisher_yates", "adaptive", "perfect"]:
        env = rlcard.make("shuffle-guess", config={
            "seed": 7,
            "game_dealer": dealer_name,
            "game_m_bits": 32,
        })
        rlcard.utils.set_seed(7)
        env.set_agents([RandomAgent(num_actions=env.num_actions)])
        _, payoffs = env.run(is_training=False)
        assert payoffs[0] >= 0


def test_env_perfect_information_keys() -> None:
    """get_perfect_information() returns expected keys."""
    env = rlcard.make("shuffle-guess", config={"seed": 1})
    env.set_agents([RandomAgent(num_actions=env.num_actions)])
    rlcard.utils.set_seed(1)
    env.run(is_training=False)
    info = env.get_perfect_information()
    assert "drawn_ids" in info
    assert "score" in info
    assert "turn" in info
    assert "dealer_name" in info
    assert len(info["drawn_ids"]) == 104  # default n


def test_env_obs_shape() -> None:
    """Observation vector has correct shape (num_actions + 1)."""
    env = rlcard.make("shuffle-guess", config={
        "seed": 0,
        "game_dealer": "fisher_yates",
        "game_n_cards": 52,
    })
    env.set_agents([RandomAgent(num_actions=env.num_actions)])
    rlcard.utils.set_seed(0)
    env.reset()
    # After reset, examine initial state
    state, _ = env.game.init_game()
    extracted = env._extract_state(state)
    obs = extracted["obs"]
    assert obs.shape == (53,)  # 52 counts + 1 norm_turn
