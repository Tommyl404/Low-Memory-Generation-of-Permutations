"""
DealerLabNolimitholdemEnv — No-Limit Hold'em with selectable shuffler.

Env ID: ``"no-limit-holdem-dealerlab"``

Behaves identically to RLCard's ``no-limit-holdem`` environment but lets
you swap the shuffle algorithm via config keys:

* ``game_dealer_algo``          — dealer name (default ``"fisher_yates"``)
* ``game_dealer_params_json``   — JSON string of extra dealer params
* ``game_m_bits``               — shorthand for m_bits param
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import rlcard
from rlcard.envs import Env
from rlcard.games.nolimitholdem import Action

from littlebrain_rlcard.envs.holdem_dealerlab.game_nolimit import (
    DealerLabNolimitHoldemGame,
)


class DealerLabNolimitholdemEnv(Env):
    """No-Limit Hold'em env with pluggable dealer algorithm."""

    def __init__(self, config: dict[str, Any] | None = None):
        defaults: dict[str, Any] = {
            "game_num_players": 2,
            "game_dealer_algo": "fisher_yates",
            "game_m_bits": 64,
            "game_dealer_params_json": "{}",
            "seed": 0,
            "allow_step_back": False,
        }
        if config:
            defaults.update(config)
        config = defaults

        # Parse dealer params
        dealer_algo = config.pop("game_dealer_algo", "fisher_yates")
        m_bits = config.pop("game_m_bits", 64)
        dp_json = config.pop("game_dealer_params_json", "{}")
        dealer_params = json.loads(dp_json) if isinstance(dp_json, str) else dp_json
        dealer_params["m_bits"] = m_bits

        num_players = config.get("game_num_players", 2)

        self.name = "no-limit-holdem-dealerlab"
        self.game = DealerLabNolimitHoldemGame(
            allow_step_back=config.get("allow_step_back", False),
            num_players=num_players,
        )
        game_config = {
            "dealer_algo": dealer_algo,
            "dealer_params": dealer_params,
            "game_num_players": num_players,
            "chips_for_each": config.get("game_chips_for_each", 100),
            "dealer_id": config.get("game_dealer_id", None),
        }
        self.game.configure(game_config)

        # Standard NL Holdem action space
        self.num_actions = 6
        self.num_players = num_players
        self.state_shape = [[54] for _ in range(num_players)]
        self.action_shape = [None for _ in range(num_players)]

        self.actions = Action

        # Load card2index mapping (same as RLCard's NL holdem env)
        with open(
            os.path.join(
                rlcard.__path__[0], "games/limitholdem/card2index.json"
            ),
        ) as f:
            self.card2index = json.load(f)

        super().__init__(config)

    def _extract_state(self, state: dict) -> dict:
        """Convert raw game state to agent-ready dict.

        Replicates the encoding from RLCard's nolimitholdem env.
        """
        extracted_state = {}

        # Convert Action enums to integer values for legal_actions
        legal_actions = OrderedDict(
            {action.value: None for action in state["legal_actions"]}
        )
        extracted_state["legal_actions"] = legal_actions

        public_cards = state["public_cards"]
        hand = state["hand"]
        my_chips = state["my_chips"]
        all_chips = state["all_chips"]
        cards = public_cards + hand
        idx = [self.card2index[card] for card in cards]
        obs = np.zeros(54)
        obs[idx] = 1
        obs[52] = float(my_chips)
        obs[53] = float(max(all_chips))
        extracted_state["obs"] = obs

        extracted_state["raw_obs"] = state
        extracted_state["raw_legal_actions"] = [a for a in state["legal_actions"]]
        extracted_state["action_record"] = self.action_recorder

        return extracted_state

    def _decode_action(self, action_id: int) -> Action:
        """Decode integer action_id to Action enum."""
        legal = self.game.get_legal_actions()
        # Try direct match
        for a in legal:
            if a.value == action_id:
                return a
        # Fallback to first legal action
        if legal:
            return legal[0]
        return Action(action_id)

    def get_payoffs(self) -> np.ndarray:
        return np.array(self.game.get_payoffs(), dtype=np.float64)

    def get_perfect_information(self) -> dict:
        """Return full game state."""
        return self.game.get_state(0) if hasattr(self.game, "get_state") else {}
