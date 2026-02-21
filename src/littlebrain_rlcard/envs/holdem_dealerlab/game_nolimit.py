"""
Adapted No-Limit Hold'em game that uses our selectable dealer.
"""

from __future__ import annotations

from typing import Any

from rlcard.games.nolimitholdem.game import NolimitholdemGame as _BaseGame

from littlebrain_rlcard.envs.holdem_dealerlab.dealer import DealerSwapDealer


class DealerLabNolimitHoldemGame(_BaseGame):
    """No-Limit Hold'em game with a pluggable shuffle algorithm."""

    def __init__(
        self,
        allow_step_back: bool = False,
        num_players: int = 2,
    ) -> None:
        super().__init__(allow_step_back=allow_step_back, num_players=num_players)
        self._dealer_algo: str = "fisher_yates"
        self._dealer_params: dict[str, Any] = {}

    def configure(self, game_config: dict[str, Any]) -> None:
        """Accept extra config keys for the dealer algorithm.

        Merges in RLCard's required defaults before calling super().
        """
        self._dealer_algo = game_config.get("dealer_algo", "fisher_yates")
        self._dealer_params = game_config.get("dealer_params", {})

        # Ensure RLCard-required keys are present
        full_config = {
            "game_num_players": game_config.get("game_num_players", 2),
            "chips_for_each": game_config.get("chips_for_each", 100),
            "dealer_id": game_config.get("dealer_id", None),
        }
        full_config.update(game_config)
        super().configure(full_config)

    def init_game(self):
        """Override to inject our custom dealer."""
        from rlcard.games.nolimitholdem import game as nlh_game_module

        orig_dealer = nlh_game_module.Dealer
        outer = self

        class _PatchedDealer:  # noqa: N801
            def __init__(self, np_random):  # noqa: N805
                wrapped = DealerSwapDealer(
                    np_random, outer._dealer_algo, outer._dealer_params
                )
                self.__dict__ = wrapped.__dict__
                self.__class__ = wrapped.__class__

        nlh_game_module.Dealer = _PatchedDealer
        try:
            result = super().init_game()
        finally:
            nlh_game_module.Dealer = orig_dealer
        return result
