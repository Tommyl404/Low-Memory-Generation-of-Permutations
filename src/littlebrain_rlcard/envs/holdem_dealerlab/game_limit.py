"""
Adapted Limit Hold'em game that uses our selectable dealer.
"""

from __future__ import annotations

from typing import Any

from rlcard.games.limitholdem.game import LimitHoldemGame as _BaseGame

from littlebrain_rlcard.envs.holdem_dealerlab.dealer import DealerSwapDealer


class DealerLabLimitHoldemGame(_BaseGame):
    """Limit Hold'em game with a pluggable shuffle algorithm."""

    def __init__(
        self,
        allow_step_back: bool = False,
        num_players: int = 2,
    ) -> None:
        super().__init__(allow_step_back=allow_step_back, num_players=num_players)
        self._dealer_algo: str = "fisher_yates"
        self._dealer_params: dict[str, Any] = {}

    def configure(self, game_config: dict[str, Any]) -> None:
        """Accept extra config keys for the dealer algorithm."""
        super().configure(game_config)
        self._dealer_algo = game_config.get("dealer_algo", "fisher_yates")
        self._dealer_params = game_config.get("dealer_params", {})

    def init_game(self):
        """Override to inject our custom dealer."""
        # Create our dealer *before* the base class builds the round
        from rlcard.games.limitholdem import game as lh_game_module

        # Store original Dealer class
        orig_dealer = lh_game_module.Dealer
        outer = self

        # Monkey-patch temporarily so base init_game uses our dealer
        class _PatchedDealer:  # noqa: N801
            def __init__(self, np_random):  # noqa: N805
                wrapped = DealerSwapDealer(
                    np_random, outer._dealer_algo, outer._dealer_params
                )
                self.__dict__ = wrapped.__dict__
                self.__class__ = wrapped.__class__

        lh_game_module.Dealer = _PatchedDealer
        try:
            result = super().init_game()
        finally:
            lh_game_module.Dealer = orig_dealer
        return result
