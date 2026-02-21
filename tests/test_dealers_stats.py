"""
Statistical sanity checks for dealer algorithms.

* Uniform dealers (Bitmap, FisherYates, Perfect): chi-square test on first-card
  distribution over K=5000 independent shuffles.
* AdaptiveThreshold: first card must be one of the d mini-decks' top cards.
"""

from __future__ import annotations

import numpy as np
import pytest
from rlcard.utils.seeding import np_random as _np_random

from littlebrain_rlcard.dealers import get_dealer

K = 5000  # number of shuffles for chi-square


@pytest.mark.parametrize("dealer_name", ["bitmap", "fisher_yates", "perfect"])
@pytest.mark.parametrize("n", [52, 104])
def test_first_card_uniform(dealer_name: str, n: int) -> None:
    """First-card frequencies should pass a lenient chi-square test."""
    counts = np.zeros(n, dtype=np.int64)
    dealer = get_dealer(dealer_name)

    for i in range(K):
        rng, _ = _np_random(i)
        dealer.reset(n, rng)
        first = dealer.draw()
        counts[first] += 1

    expected = K / n
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    df = n - 1
    threshold = 10 * df  # very lenient — catches degenerate bias

    assert chi2 < threshold, (
        f"{dealer_name} n={n}: chi2={chi2:.1f} exceeds {threshold} (df={df})"
    )


def test_adaptive_first_card_support() -> None:
    """AdaptiveThreshold first card must be a top card of some mini-deck."""
    n = 104
    m_bits = 64
    dealer = get_dealer("adaptive")

    for seed in range(200):
        rng, _ = _np_random(seed)
        dealer.reset(n, rng, m_bits=m_bits)

        # Compute expected top cards (start of each mini-deck)
        d = dealer._d
        top_cards = set()
        for i in range(d):
            top_cards.add(dealer._starts[i])  # ell[i]==0 initially

        first = dealer.draw()
        assert first in top_cards, (
            f"seed={seed}: first card {first} not in top-cards {top_cards}"
        )


@pytest.mark.parametrize("dealer_name", ["bitmap", "fisher_yates", "perfect"])
def test_peek_distribution_sums_to_one(dealer_name: str) -> None:
    """peek_next_distribution() probabilities must sum to ~1."""
    rng, _ = _np_random(0)
    dealer = get_dealer(dealer_name)
    dealer.reset(52, rng)

    for _ in range(10):
        dist = dealer.peek_next_distribution()
        assert dist is not None
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9, f"sum={total}"
        dealer.draw()


def test_adaptive_peek_drawable_options() -> None:
    """peek_drawable_options() returns valid (card_id, prob) tuples."""
    rng, _ = _np_random(7)
    dealer = get_dealer("adaptive")
    dealer.reset(104, rng, m_bits=64)

    for _ in range(10):
        opts = dealer.peek_drawable_options()
        assert len(opts) > 0
        total_prob = sum(p for _, p in opts)
        assert abs(total_prob - 1.0) < 1e-9
        drawn = dealer.draw()
        # drawn must have been in the drawable options
        drawn_ids = {cid for cid, _ in opts}
        assert drawn in drawn_ids, f"drew {drawn}, options were {drawn_ids}"


@pytest.mark.parametrize("dealer_name", ["bitmap", "fisher_yates", "perfect"])
def test_determinism_same_seed(dealer_name: str) -> None:
    """Same seed produces identical permutations."""
    n = 52
    for seed in [0, 42, 999]:
        rng1, _ = _np_random(seed)
        rng2, _ = _np_random(seed)
        d1 = get_dealer(dealer_name)
        d2 = get_dealer(dealer_name)
        d1.reset(n, rng1)
        d2.reset(n, rng2)
        perm1 = [d1.draw() for _ in range(n)]
        perm2 = [d2.draw() for _ in range(n)]
        assert perm1 == perm2, f"seed={seed}: permutations differ"
