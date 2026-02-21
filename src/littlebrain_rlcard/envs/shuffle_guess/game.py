"""
ShuffleGuessGame — core game logic for the *shuffle-guess* RLCard environment.

Single-player card-guessing game that measures the predictability of a dealer
algorithm.  At each step the agent guesses the next card type (0..51) and
receives reward 1 for a correct guess.

Default configuration: n=104 (two standard decks), action_mode="type" (52
actions corresponding to card types).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from littlebrain_rlcard.cards import NUM_TYPES
from littlebrain_rlcard.dealers import get_dealer


class ShuffleGuessGame:
    """RLCard-compatible game class for the shuffle-guess environment."""

    def __init__(self) -> None:
        self.np_random: np.random.RandomState | None = None
        self.num_players: int = 1
        # Configured later via .configure()
        self._n: int = 104
        self._num_decks: int = 2
        self._action_mode: str = "type"
        self._dealer_name: str = "adaptive"
        self._m_bits: int = 64
        self._dealer_params: dict[str, Any] = {}
        # Per-episode state
        self._dealer = get_dealer(self._dealer_name)
        self._t: int = 0
        self._counts: np.ndarray = np.zeros(NUM_TYPES, dtype=np.int32)
        self._score: int = 0
        self._drawn_ids: list[int] = []
        self._last_drawn_id: int | None = None
        self._done: bool = False

    # -- RLCard game interface -----------------------------------------------

    def configure(self, game_config: dict[str, Any]) -> None:
        """Apply configuration keys (called before ``init_game``)."""
        self._n = game_config.get("n_cards", 104)
        self._num_decks = game_config.get("num_decks", 2)
        self._action_mode = game_config.get("action_mode", "type")
        self._dealer_name = game_config.get("dealer", "adaptive")
        self._m_bits = game_config.get("m_bits", 64)
        self._dealer_params = game_config.get("dealer_params", {})
        self._dealer = get_dealer(self._dealer_name)

    def init_game(self) -> tuple[dict, int]:
        """Start a new episode.  Returns ``(state_dict, player_id)``."""
        params: dict[str, Any] = {"m_bits": self._m_bits, **self._dealer_params}
        self._dealer.reset(self._n, self.np_random, **params)
        self._t = 0
        self._counts = np.zeros(NUM_TYPES, dtype=np.int32)
        self._score = 0
        self._drawn_ids = []
        self._last_drawn_id = None
        self._done = False
        return self.get_state(0), 0

    def step(self, action: int) -> tuple[dict, int]:
        """Agent guesses *action* (card type id); dealer draws next card.

        Returns ``(state_dict, player_id)``.
        """
        if self._done:
            raise RuntimeError("Game is already over")

        drawn_id = self._dealer.draw()
        drawn_type = drawn_id % NUM_TYPES

        if action == drawn_type:
            self._score += 1

        self._counts[drawn_type] += 1
        self._drawn_ids.append(drawn_id)
        self._last_drawn_id = drawn_id
        self._t += 1

        if self._t >= self._n:
            self._done = True

        return self.get_state(0), 0

    def get_state(self, player_id: int) -> dict:
        """Return the current observation dict for *player_id*."""
        return {
            "counts": self._counts.copy(),
            "turn": self._t,
            "n": self._n,
            "last_drawn_id": self._last_drawn_id,
            "drawn_ids": list(self._drawn_ids),
            "dealer_name": self._dealer_name,
            "dealer_params": {**self._dealer_params, "m_bits": self._m_bits},
            "score": self._score,
            "legal_actions": self.get_legal_actions(),
        }

    def get_num_players(self) -> int:
        return 1

    def get_num_actions(self) -> int:
        if self._action_mode == "type":
            return NUM_TYPES  # 52
        return self._n  # full card-id space

    def get_legal_actions(self) -> list[int]:
        if self._action_mode == "type":
            return list(range(NUM_TYPES))
        return list(range(self._n))

    def get_player_id(self) -> int:
        return 0

    def is_over(self) -> bool:
        return self._done

    def get_payoffs(self) -> list[float]:
        """Return ``[total_correct_guesses]``."""
        return [float(self._score)]

    # -- extra ---------------------------------------------------------------

    @property
    def dealer(self):
        """Direct access to the underlying dealer (for scripts)."""
        return self._dealer
