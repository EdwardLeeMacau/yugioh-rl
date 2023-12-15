###############
#   Package   #
###############
import random
from itertools import chain
from typing import Dict, List, Set, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

# customized package
from env.game import (CARDS2IDX, DECK, OPTIONS, POSSIBLE_ACTIONS, Action,
                      GameState, Policy)

################
#   Variable   #
################

ckt_dict = {}

#############
#   Class   #
#############

class RandomPolicy(Policy):
    def react(
            self,
            state: GameState,
            options: List[Action],
            targets: Dict[Action, List[Action]]
        ) -> Action:
        return random.choice(options + list(targets.values()))

# the following class is the old version of the pseudo-self-play opponent
class PseudoSelfPlayPolicy(Policy):
    # option (string) -> option (int)
    OPTIONS2DIGITS = { option: i for i, option in enumerate(OPTIONS, 1) }
    OPTIONS2DIGITS[None] = 0

    # action (string | CardID) -> action (int)
    ACTION2DIGITS = { action: i for i, action in enumerate(POSSIBLE_ACTIONS) }

    # action (int) -> action (string | CardID)
    DIGITS2ACTION = { value: key for key, value in ACTION2DIGITS.items() }

    # phase (int) -> phase (int)
    PHASE2DIGITS = { 1: 0, 2: 1, 4: 2, 8: 3, 256: 4, 512: 5 }

    def __init__(self, model_path: str = None):
        self._model_path = model_path
        if not any(ckt_dict):
            ckt_dict['0'] = MaskablePPO.load(model_path)

    def _encode_state(self, game_state: Dict, actions: Tuple[List[Action], Dict[Action, Set[str]]]):
        """ Encode the game state into the tensor.

        This function changes the interval variables `_action_mask`, `_state` and `_spec_map`.

        Arguments
        ---------
        game_state: Dict
            The game state dictionary, which is queried from the Game() instance.

        actions: Tuple[List[Action], List[Action]]
            The list of valid actions, which is composed of parts of actions and cards.
        """
        # Utilities
        options, cards = actions
        player, opponent = game_state['player'], game_state['opponent']

        # Prepare `self._spec_unmap` for future usage.
        self._spec_unmap = cards

        # Compose the action mask.
        # Mark as True if the action is valid.
        mask = np.zeros(shape=(len(POSSIBLE_ACTIONS), ), dtype=np.int8)
        for opt in map(lambda x: self.ACTION2DIGITS[x], chain(options, cards.keys())):
            mask[opt] = 1

        self._action_masks = mask
        self._state = {

            # --------------------------------- Games information ---------------------------------

            "phase": self.PHASE2DIGITS[game_state['phase']],
            "turn": np.array([game_state['turn']], dtype=np.float32),

            # -------------------------------- Players information --------------------------------

            "agent_LP": np.array([player['lp']]) / 8000.,
            "agent_hand": self._IDList_to_MultiHot(player['hand']),
            "agent_deck": np.array([player['deck']]) / 40.,
            "agent_grave": self._IDList_to_MultiHot(player['grave']),
            "agent_removed": self._IDList_to_MultiHot(player['removed']),

            # ------------------------------ Valid action information -----------------------------

            "last_option": self.OPTIONS2DIGITS[game_state['last_option']],
            "action_masks": self._action_masks.astype(np.float32),

            # ------------------------------- Opponents information -------------------------------

            "oppo_LP": np.array([opponent['lp']]) / 8000.,
            "oppo_hand": np.array([opponent['hand']]) / 40.,
            "oppo_deck": np.array([opponent['deck']]) / 40.,
            "oppo_grave": self._IDList_to_MultiHot(opponent['grave']),
            "oppo_removed": self._IDList_to_MultiHot(opponent['removed']),

            # --------------------------------- Table information ---------------------------------

            "t_agent_m": self._IDStateList_to_vector(player['monster']).flatten(),
            "t_agent_s": self._IDStateList_to_vector(player['spell']).flatten(),
            "t_oppo_m": self._IDStateList_to_vector(opponent['monster']).flatten(),
            "t_oppo_s": self._IDStateList_to_vector(opponent['spell']).flatten(),
        }

    @staticmethod
    def _IDStateList_to_vector(id_state_list: List[Tuple[int, int]]) -> np.ndarray:
        assert len(id_state_list) == 5, "The card list is padded to 5 with empty card (None)"

        frame_array = np.zeros(shape=(5, len(DECK) + 2), dtype=np.float32)
        for i, id_state in enumerate(id_state_list):
            card_id, pos = id_state[0], id_state[1]

            # One-hot encoding for the card ID.
            frame_array[i, CARDS2IDX[card_id]] = 1

            # One-hot encoding for the position (UP/DOWN, ATK/DEF).
            frame_array[i, len(DECK):] = [pos & 0b101, pos & 0b011]

        return frame_array

    @classmethod
    def _IDList_to_MultiHot(cls, id_list: List[int]) -> np.ndarray:
        multi_hot = np.zeros(shape=(len(DECK), ), dtype=np.float32)

        for card_id in id_list:
            multi_hot[CARDS2IDX[card_id]] += 1

        return multi_hot

    def decode_action(self, action: int) -> str:
        # Decode the action as server's format first.
        # If the action is related to a card (int), then further decode it as a position code.
        action: str | int = self.DIGITS2ACTION[action]
        if self._spec_unmap.get(action, None) is not None:
            action = self._spec_unmap[action]
        return action

    def react(
            self,
            state: GameState,
            options: List[Action],
            targets: Dict[Action, Set[str]]
        ) -> Action:
        self._encode_state(state, (options, targets))
        action, _state = ckt_dict['0'].predict(self._state, action_masks=self._action_masks, deterministic=True)
        action = self.decode_action(action.item())
        return action
