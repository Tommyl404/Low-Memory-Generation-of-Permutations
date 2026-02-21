"""
DealerSwap — drop-in replacement for RLCard's card dealer that uses
one of the four littlebrain dealer algorithms for shuffling.

Only single-deck (52 cards) is supported to maintain compatibility with
standard poker hand evaluators.
"""

from __future__ import annotations

from typing import Any

# RLCard utilities
from rlcard.utils.utils import init_standard_deck

from littlebrain_rlcard.dealers import get_dealer


class DealerSwapDealer:
    """Replacement card dealer for RLCard poker games.

    Mimics ``rlcard.games.limitholdem.dealer.LimitHoldemDealer`` but uses a
    selectable shuffle algorithm from :mod:`littlebrain_rlcard.dealers`.

    Parameters
    ----------
    np_random : numpy.random.RandomState
        RNG instance (passed through from the game).
    dealer_name : str
        One of ``"bitmap"``, ``"fisher_yates"``, ``"adaptive"``, ``"perfect"``.
    dealer_params : dict
        Extra kwargs forwarded to ``dealer.reset()``.
    """

    def __init__(
        self,
        np_random,
        dealer_name: str = "fisher_yates",
        dealer_params: dict[str, Any] | None = None,
    ) -> None:
        self.np_random = np_random
        self._dealer_name = dealer_name
        self._dealer_params = dealer_params or {}
        self._algo = get_dealer(dealer_name)
        self.deck: list = []
        self.pot = 0
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle a fresh 52-card deck using the chosen algorithm."""
        base_deck = init_standard_deck()
        n = len(base_deck)  # 52
        self._algo.reset(n, self.np_random, **self._dealer_params)
        perm = [self._algo.draw() for _ in range(n)]
        # RLCard deals via deck.pop() => store reversed so first deal = perm[0]
        self.deck = [base_deck[i] for i in perm[::-1]]

    def deal_card(self):
        """Deal one card from the deck (same interface as RLCard)."""
        return self.deck.pop()

    def deal_cards(self, player, num: int) -> None:
        """Deal *num* cards to *player* (same interface as RLCard)."""
        for _ in range(num):
            player.hand.append(self.deck.pop())
