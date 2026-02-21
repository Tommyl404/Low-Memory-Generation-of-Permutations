"""
ShuffleGuessEnv — RLCard Env wrapper for the shuffle-guess game.

Env ID: ``"shuffle-guess"``

Custom config keys (prefix ``game_`` in the config dict):

* ``game_n_cards``        — total cards, default 104
* ``game_num_decks``      — number of standard decks, default 2
* ``game_action_mode``    — ``"type"`` (52 actions) or ``"id"`` (n actions)
* ``game_dealer``         — dealer algorithm name
* ``game_m_bits``         — memory-budget parameter for AdaptiveThresholdDealer
* ``game_dealer_params``  — extra dealer params dict
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

# RLCard imports
from rlcard.envs import Env

from littlebrain_rlcard.envs.shuffle_guess.game import ShuffleGuessGame


class ShuffleGuessEnv(Env):
    """RLCard environment for the shuffle-guess game."""

    def __init__(self, config: dict[str, Any] | None = None):
        # Defaults
        defaults: dict[str, Any] = {
            "game_n_cards": 104,
            "game_num_decks": 2,
            "game_action_mode": "type",
            "game_dealer": "adaptive",
            "game_m_bits": 64,
            "game_dealer_params": {},
            "seed": 0,
            "allow_step_back": False,
        }
        if config:
            defaults.update(config)
        config = defaults

        # Extract game-specific keys
        game_config: dict[str, Any] = {
            "n_cards": config.pop("game_n_cards", 104),
            "num_decks": config.pop("game_num_decks", 2),
            "action_mode": config.pop("game_action_mode", "type"),
            "dealer": config.pop("game_dealer", "adaptive"),
            "m_bits": config.pop("game_m_bits", 64),
            "dealer_params": config.pop("game_dealer_params", {}),
        }

        self.name = "shuffle-guess"
        self.game = ShuffleGuessGame()
        self.game.configure(game_config)
        self._game_config = game_config

        # Number of actions / players
        self.num_actions = self.game.get_num_actions()
        self.num_players = 1
        self.state_shape = [[self.num_actions + 1]]  # counts(52) + norm_turn(1)
        self.action_shape = [None]

        super().__init__(config)

    def _extract_state(self, state: dict) -> dict:
        """Convert game state dict to the form expected by agents."""
        counts = state["counts"].astype(np.float32)
        norm_turn = np.array(
            [state["turn"] / max(state["n"], 1)], dtype=np.float32
        )
        obs = np.concatenate([counts, norm_turn])

        legal_actions = OrderedDict(
            {a: None for a in state["legal_actions"]}
        )
        return {
            "obs": obs,
            "legal_actions": legal_actions,
            "raw_obs": state,
            "raw_legal_actions": state["legal_actions"],
            "action_record": self.action_recorder,
        }

    def _decode_action(self, action_id: int) -> int:
        """Decode action_id to game action (identity mapping)."""
        return action_id

    def get_payoffs(self) -> np.ndarray:
        return np.array(self.game.get_payoffs(), dtype=np.float64)

    def get_perfect_information(self) -> dict:
        """Return full game state for analysis / determinism tests."""
        state = self.game.get_state(0)
        return {
            "drawn_ids": state["drawn_ids"],
            "score": state["score"],
            "turn": state["turn"],
            "counts": state["counts"].tolist(),
            "dealer_name": state["dealer_name"],
            "dealer_params": state["dealer_params"],
        }
