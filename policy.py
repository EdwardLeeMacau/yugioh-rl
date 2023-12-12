###############
#   Package   #
###############
import random
import gymnasium as gym
import os
import numpy as np
import torch

from typing import List
from itertools import chain, combinations
from sb3_contrib import MaskablePPO
from typing import Dict, List, Tuple, Optional, Set

# costumized package
from env.game import GameState, Action, Policy

#############
#   Class   #
#############
class RandomPolicy(Policy):
    def react(self, state: GameState, actions: List[Action]) -> Action:
        return random.choice(actions)

class PseudoSelfPlayPolicy(Policy):
    def __init__(self, ckt_model_dir: str = None, env: gym.Env = None):
        # check the
        assert os.path.isdir(ckt_model_dir), AssertionError("Wrong Model Dictionary.")
        self.ckt_dir = ckt_model_dir
        self._ckt_list = os.listdir(self.ckt_dir)
        self.env = env

# the following class is the old version of the pseudo-self-play opponent
class PseudoSelfPlayPolicy_0(Policy):
    _RACES = [
        "Warrior",
        "Spellcaster",
        "Fairy",
        "Fiend",
        "Zombie",
        "Machine",
        "Aqua",
        "Pyro",
        "Rock",
        "Winged Beast",
        "Plant",
        "Insect",
        "Thunder",
        "Dragon",
        "Beast",
        "Beast-Warrior",
        "Dinosaur",
        "Fish",
        "Sea Serpent",
        "Reptile",
        "Psychic",
        "Divine-Beast",
        "Creator God",
        "Wyrm",
        "Cyberse",
    ]

    # List of all cards in the YGO04 format.
    # Use for assigning the cards into the one-hot encoding / multi-hot encoding.
    #
    # TODO: Accelerate index queries with hash map
    _DECK_LIST = [
        None,       # empty space
        72989439,
        77585513,
        18036057,
        63749102,
        88240808,
        33184167,
        39507162,
        71413901,
        76922029,
        74131780,
        78706415,
        79575620,
        23205979,
        8131171,
        19613556,
        32807846,
        55144522,
        42829885,
        17375316,
        4031928,
        45986603,
        69162969,
        71044499,
        72302403,
        5318639,
        70828912,
        29401950,
        53582587,
        56120475,
        60082869,
        83555666,
        97077563,
        7572887,
        74191942,
        73915051,
        44095762,
        31560081,
        73915052,   # Scapegoat (should map to the same encoding)
        73915053,   # Scapegoat
        73915054,   # Scapegoat
        73915055,   # Scapegoat
        0,          # hidden card
    ]

    # Enumerate actions
    _ACTIONS = [
        # *_DECK_LIST,
        *_DECK_LIST,
        *_RACES,    # Announce races
        "e", # enter end phase
        "z", # back
        "s", # summon this card in face-up attack position
        "m", # summon this card in face-down defense position/ enter main phase
        "t", # set this card (Trap/Magic)
        "v", # activate this card
        "c", # cancel
        "b", # enter battle phase
        "y", # yes
        "n", # no
        "a", # attack
        "r", # reposition
        '1', # select option of Don Zaloog
        '2',
        '3', # FACEUP.DEFENSE
        '4',
    ]

    # action (string | CardID) -> action (int)
    ACTION2DIGITS = { action: i for i, action in enumerate(_ACTIONS) }

    # action (int) -> action (string | CardID)
    DIGITS2ACTION = { value: key for key, value in ACTION2DIGITS.items() }

    # phase (int) -> phase (int)
    PHASE2DIGITS = { 1: 0, 2: 1, 4: 2, 8: 3, 256: 4, 512: 5 }

    def __init__(self, model_path: str = None):
        self._model_path = model_path

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
        mask = np.zeros(shape=(len(self._ACTIONS), ), dtype=np.int8)
        for opt in map(lambda x: self.ACTION2DIGITS[x], chain(options, cards.keys())):
            mask[opt] = 1

        self._action_mask = mask
        self._state = {

            # --------------------------------- Games information ---------------------------------

            "phase": self.PHASE2DIGITS[game_state['phase']],

            # -------------------------------- Players information --------------------------------

            "agent_LP": np.array([player['lp']]) / 8000.,
            "agent_hand": self._IDList_to_MultiHot(player['hand']),
            "agent_deck": np.array([player['deck']]) / 40.,
            "agent_grave": self._IDList_to_MultiHot(player['grave']),
            "agent_removed": self._IDList_to_MultiHot(player['removed']),

            # ------------------------------ Valid action information -----------------------------

            "action_mask": self._action_mask.astype(np.float32),

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

    @classmethod
    def _IDStateList_to_vector(cls, id_state_list: List[Tuple[int, int]]) -> np.ndarray:
        assert len(id_state_list) == 5, "The card list is padded to 5 with empty card (None)"

        frame_array = np.zeros(shape=(5, 45), dtype=np.float32)
        # frame_array[:, 0] = cls._DECK_LIST.index(None)
        # frame_array[:, 1] = 4

        # One-hot encoding for the card ID.
        for i in range(len(id_state_list)):
            frame_array[i, cls._DECK_LIST.index(id_state_list[i][0])] = 1
            # frame_array[i, 1] = id_state_list[i][1]

        return frame_array

    @classmethod
    def _IDList_to_MultiHot(cls, id_list: List[int]) -> np.ndarray:
        multi_hot = np.zeros(shape=(40, ), dtype=np.float32)

        for card_id in id_list:
            multi_hot[cls._DECK_LIST.index(card_id)] += 1

        return multi_hot

    def _decode_action(self, action: int) -> str:
        # Decode the action as server's format first.
        # If the action is related to a card (int), then further decode it as a position code.
        action: str | int = self.DIGITS2ACTION[action]
        if self._spec_unmap.get(action, None) is not None:
            action = self._spec_unmap[action]
        return action

    def react(self, state: GameState, actions: Tuple[List[Action], Dict[Action, Set[str]]]) -> Action:
        model = MaskablePPO.load(self._model_path)
        self._encode_state(state, actions)
        action, _state = model.predict(self._state, deterministic=True)
        action = self._decode_action(action.item())
        del model
        return action
