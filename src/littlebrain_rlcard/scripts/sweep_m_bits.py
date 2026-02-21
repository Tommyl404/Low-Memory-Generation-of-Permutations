"""
sweep_m_bits.py — Sweep over m_bits values for AdaptiveThresholdDealer.

Runs N episodes of shuffle-guess with:
  1. A RandomAgent baseline
  2. A simple myopic guesser using ``dealer.peek_next_distribution()``

Plots average score vs m_bits and saves to ``artifacts/plots/``.

Usage::

    python -m littlebrain_rlcard.scripts.sweep_m_bits
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rlcard
from rlcard.agents import RandomAgent

M_BITS_VALUES = [8, 16, 32, 64, 128]
N_EPISODES = 50
N_CARDS = 104
SEED_BASE = 100


class MyopicGuesser:
    """Agent that picks the most probable next card type via peek_next_distribution."""

    def __init__(self, num_actions: int, game_ref=None):
        self.num_actions = num_actions
        self.game_ref = game_ref
        self.use_raw = True  # needed by RLCard

    def step(self, state):
        """Choose best action if distribution available, else random."""
        return self.eval_step(state)

    def eval_step(self, state):
        dealer = None
        if self.game_ref is not None:
            dealer = self.game_ref.dealer

        if dealer is not None:
            dist = dealer.peek_next_distribution()
            if dist:
                # Aggregate by type_id
                type_probs: dict[int, float] = {}
                for cid, p in dist.items():
                    tid = cid % 52
                    type_probs[tid] = type_probs.get(tid, 0.0) + p
                best = max(type_probs, key=type_probs.get)
                return best, {}

        # Fallback: uniform random over legal actions
        legal = list(state.get("legal_actions", {}).keys())
        if legal:
            idx = np.random.choice(len(legal))
            return legal[idx], {}
        return 0, {}


def run_sweep() -> dict[int, dict[str, float]]:
    """Run the sweep.  Returns {m_bits: {'random': avg_score, 'myopic': avg_score}}."""
    results: dict[int, dict[str, float]] = {}

    for m_bits in M_BITS_VALUES:
        random_scores = []
        myopic_scores = []

        for ep in range(N_EPISODES):
            seed = SEED_BASE + ep

            # --- Random agent ---
            env = rlcard.make("shuffle-guess", config={
                "seed": seed,
                "game_dealer": "adaptive",
                "game_m_bits": m_bits,
                "game_n_cards": N_CARDS,
            })
            rlcard.utils.set_seed(seed)
            env.set_agents([RandomAgent(num_actions=env.num_actions)])
            _, payoffs = env.run(is_training=False)
            random_scores.append(payoffs[0])

            # --- Myopic agent ---
            env2 = rlcard.make("shuffle-guess", config={
                "seed": seed,
                "game_dealer": "adaptive",
                "game_m_bits": m_bits,
                "game_n_cards": N_CARDS,
            })
            rlcard.utils.set_seed(seed)
            myopic = MyopicGuesser(
                num_actions=env2.num_actions,
                game_ref=env2.game,
            )
            env2.set_agents([myopic])
            _, payoffs2 = env2.run(is_training=False)
            myopic_scores.append(payoffs2[0])

        results[m_bits] = {
            "random": float(np.mean(random_scores)),
            "myopic": float(np.mean(myopic_scores)),
        }
        print(
            f"m_bits={m_bits:>4}  "
            f"random_avg={results[m_bits]['random']:.2f}  "
            f"myopic_avg={results[m_bits]['myopic']:.2f}"
        )

    return results


def plot_results(results: dict[int, dict[str, float]]) -> None:
    """Save a plot of avg score vs m_bits."""
    m_vals = sorted(results.keys())
    random_avgs = [results[m]["random"] for m in m_vals]
    myopic_avgs = [results[m]["myopic"] for m in m_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m_vals, random_avgs, "o-", label="RandomAgent")
    ax.plot(m_vals, myopic_avgs, "s--", label="MyopicGuesser")
    ax.set_xlabel("m_bits")
    ax.set_ylabel("Avg correct guesses")
    ax.set_title(f"Shuffle-Guess score vs m_bits (n={N_CARDS}, {N_EPISODES} eps)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "plots"
    )
    # Normalize the path
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sweep_m_bits.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {path}")


def main() -> None:
    print(f"Sweeping m_bits={M_BITS_VALUES}, {N_EPISODES} episodes each, n={N_CARDS}\n")
    results = run_sweep()
    plot_results(results)


if __name__ == "__main__":
    main()
