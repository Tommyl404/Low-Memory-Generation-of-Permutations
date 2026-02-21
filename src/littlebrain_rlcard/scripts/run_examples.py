"""
run_examples.py — Quick demo of littlebrain_rlcard.

Demonstrates:
  1. shuffle-guess with the default adaptive dealer (n=104, m_bits=64, seed=42)
  2. First-card distribution table for FisherYates over 1000 runs (n=52)

Example smoke output::

    === Shuffle-Guess demo (adaptive, n=104, m_bits=64, seed=42) ===
    Score (correct guesses): 3.0
    First 10 drawn IDs: [37, 82, 5, ...]

    === First-card distribution (FisherYates, n=52, 1000 runs) ===
    card_id  count
    0        18
    1        21
    ...
    Chi2=49.2  df=51

Usage::

    python -m littlebrain_rlcard.scripts.run_examples
"""

from __future__ import annotations

import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils.seeding import np_random as _np_random

from littlebrain_rlcard.dealers import get_dealer


def demo_shuffle_guess() -> None:
    """Run one episode of shuffle-guess with an adaptive dealer."""
    print("=== Shuffle-Guess demo (adaptive, n=104, m_bits=64, seed=42) ===")
    env = rlcard.make("shuffle-guess", config={
        "seed": 42,
        "game_dealer": "adaptive",
        "game_m_bits": 64,
        "game_n_cards": 104,
    })
    env.set_agents([RandomAgent(num_actions=env.num_actions)])
    rlcard.utils.set_seed(42)
    trajectories, payoffs = env.run(is_training=False)
    info = env.get_perfect_information()
    print(f"Score (correct guesses): {payoffs[0]}")
    print(f"First 10 drawn IDs: {info['drawn_ids'][:10]}")
    print()


def demo_first_card_distribution() -> None:
    """Print first-card frequency table for FisherYates (n=52, 1000 runs)."""
    print("=== First-card distribution (FisherYates, n=52, 1000 runs) ===")
    n = 52
    num_runs = 1000
    counts = np.zeros(n, dtype=np.int64)
    dealer = get_dealer("fisher_yates")
    for i in range(num_runs):
        rng, _ = _np_random(i)
        dealer.reset(n, rng)
        first = dealer.draw()
        counts[first] += 1

    expected = num_runs / n
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    df = n - 1

    print(f"{'card_id':>8}  {'count':>5}")
    for cid in range(n):
        print(f"{cid:>8}  {counts[cid]:>5}")
    print(f"Chi2={chi2:.1f}  df={df}")
    print()


def main() -> None:
    demo_shuffle_guess()
    demo_first_card_distribution()


if __name__ == "__main__":
    main()
