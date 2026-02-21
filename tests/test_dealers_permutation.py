"""
Unit tests — permutation validity for each dealer.

For each dealer, generate a full n-draw permutation and assert:
  * len(set(outputs)) == n
  * sorted(outputs) == list(range(n))
  * calling draw() after n draws raises RuntimeError
"""

from __future__ import annotations

import pytest
from rlcard.utils.seeding import np_random as _np_random

from littlebrain_rlcard.dealers import get_dealer

# Test parameters: (dealer_name, n, extra_params)
_DEALER_CONFIGS = [
    ("bitmap", 104, {}),
    ("bitmap", 52, {}),
    ("fisher_yates", 104, {}),
    ("fisher_yates", 52, {}),
    ("adaptive", 104, {"m_bits": 64}),
    ("adaptive", 52, {"m_bits": 32}),
    ("adaptive", 104, {"m_bits": 8}),
    ("adaptive", 10, {"m_bits": 8}),
    ("perfect", 104, {}),
    ("perfect", 52, {}),
    ("perfect", 7, {}),
]


@pytest.mark.parametrize("dealer_name,n,params", _DEALER_CONFIGS)
def test_full_permutation(dealer_name: str, n: int, params: dict) -> None:
    """Drawing n cards yields a permutation of {0..n-1}."""
    rng, _ = _np_random(42)
    dealer = get_dealer(dealer_name)
    dealer.reset(n, rng, **params)

    outputs = [dealer.draw() for _ in range(n)]
    assert len(set(outputs)) == n, f"Duplicates found: {len(set(outputs))} unique of {n}"
    assert sorted(outputs) == list(range(n))
    assert dealer.remaining() == 0


@pytest.mark.parametrize("dealer_name,n,params", _DEALER_CONFIGS)
def test_exhaustion_raises(dealer_name: str, n: int, params: dict) -> None:
    """draw() after exhaustion must raise RuntimeError."""
    rng, _ = _np_random(123)
    dealer = get_dealer(dealer_name)
    dealer.reset(n, rng, **params)

    for _ in range(n):
        dealer.draw()

    with pytest.raises(RuntimeError, match="[Ee]xhausted"):
        dealer.draw()


@pytest.mark.parametrize("dealer_name,n,params", _DEALER_CONFIGS)
def test_remaining_decrements(dealer_name: str, n: int, params: dict) -> None:
    """remaining() counts down correctly."""
    rng, _ = _np_random(7)
    dealer = get_dealer(dealer_name)
    dealer.reset(n, rng, **params)

    for t in range(n):
        assert dealer.remaining() == n - t
        dealer.draw()
    assert dealer.remaining() == 0


@pytest.mark.parametrize("dealer_name,n,params", _DEALER_CONFIGS)
def test_state_summary(dealer_name: str, n: int, params: dict) -> None:
    """state_summary() returns required keys."""
    rng, _ = _np_random(1)
    dealer = get_dealer(dealer_name)
    dealer.reset(n, rng, **params)
    summary = dealer.state_summary()
    assert "theoretical_bits" in summary
    assert "python_bytes" in summary
    assert isinstance(summary["theoretical_bits"], int)


@pytest.mark.parametrize("dealer_name", ["bitmap", "fisher_yates", "perfect"])
def test_next_card_alias(dealer_name: str) -> None:
    """next_card() is an alias for draw()."""
    rng1, _ = _np_random(99)
    rng2, _ = _np_random(99)
    d1 = get_dealer(dealer_name)
    d2 = get_dealer(dealer_name)
    d1.reset(52, rng1)
    d2.reset(52, rng2)
    assert d1.draw() == d2.next_card()


def test_multiple_resets() -> None:
    """A dealer can be reset and reused."""
    dealer = get_dealer("fisher_yates")
    for seed in range(5):
        rng, _ = _np_random(seed)
        dealer.reset(52, rng)
        out = [dealer.draw() for _ in range(52)]
        assert sorted(out) == list(range(52))
