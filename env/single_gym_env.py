###############
#   Package   #
###############
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register

import numpy as np
import pysnooper

import logging
import sys
import os
import torch
from torch import Tensor

from itertools import chain, combinations
from six import StringIO
from datetime import datetime
from multiprocessing import Process
from threading import Thread, Event
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
from telnetlib import Telnet

# insert path for package
sys.path.insert(0, os.path.abspath(os.getcwd()))

from env.game import Action, Game, GameState, Player, Policy
from policy import RandomPolicy

#######################
#   Global Variable   #
#######################

GameInfo = Dict
CardID = int

################
#   Function   #
################

def game_loop(player: Player, policy: Policy) -> None:
    terminated, state = player.decode(player.wait())

    while not terminated:
        options, targets = player.list_valid_actions()
        action = policy.react(None, options + list(targets.values()))
        terminated, state = player.step(action)

    return

#############
#   Class   #
#############
class YGOEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 2,
    }

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

    # Type hinting for public fields.

    action_space: spaces.Discrete
    observation_space: spaces.Dict

    # Type hinting for private fields.

    _game: Game
    _opponent: Policy
    _process: Process

    # Field for caching the state of the game.

    _action_mask: np.ndarray | Tensor
    _spec_map: Dict[str, int]
    _spec_unmap: Dict[int, str]
    _state: Tuple[Dict[str, Tensor], GameInfo]
    _step: int

    def __init__(self, opponent: Policy = RandomPolicy()):
        super(YGOEnv, self).__init__()
        # define the Game and the opponent object
        self._game = None
        self._opponent = opponent
        self._process = None

        self._state = None
        self._step = 0

        # define the action space and the observation space
        self.action_space = spaces.Discrete(len(self.ACTION2DIGITS.keys()), start=0)
        # trap card have not been implemented.
        self.observation_space = spaces.Dict({

            # Current game phase (affect the valid actions)
            "phase": spaces.Discrete(6),

            # -------------------------------- Players information --------------------------------

            # Player's life points (normalized to [0, 1])
            "agent_LP": spaces.Box(low=-1., high=1., shape=(1, ), dtype=np.float32),
            # Player's hand (multi-hot encoding, number of cards in hand <= 3)
            "agent_hand": spaces.Box(low=0., high=3., shape=(40, ), dtype=np.float32),
            # Player's deck (number of cards in deck), can infer the remaining cards in deck
            "agent_deck": spaces.Box(low=0, high=40, shape=(1, ), dtype=np.float32),
            # Player's grave (multi-hot encoding)
            "agent_grave": spaces.Box(low=0., high=4., shape=(40, ), dtype=np.float32),
            # Player's banished cards (multi-hot encoding)
            "agent_removed": spaces.Box(low=0., high=4., shape=(40, ), dtype=np.float32),
            # Valid action information
            "action_mask": spaces.MultiBinary(len(self.ACTION2DIGITS.keys())),

            # ------------------------------- Opponents information -------------------------------

            "oppo_LP": spaces.Box(low=-1, high=1., shape=(1, ), dtype=np.float32),
            "oppo_hand": spaces.Box(low=0., high=40., shape=(1, ), dtype=np.float32),
            "oppo_deck": spaces.Box(low=0, high=40, shape=(1, ), dtype=np.float32),
            "oppo_grave": spaces.Box(low=0., high=4., shape=(40, ), dtype=np.float32),
            "oppo_removed": spaces.Box(low=0., high=4., shape=(40, ), dtype=np.float32),

            # --------------------------------- Table information ---------------------------------

            "t_agent_m": spaces.MultiBinary(225),
            "t_oppo_m": spaces.MultiBinary(225),
            "t_agent_s": spaces.MultiBinary(225),
            "t_oppo_s": spaces.MultiBinary(225),
        })

        # Set negative reward (penalty) for illegal moves (optional)
        self.set_illegal_move_reward(-0.2)

        # Ready for a game
        self.reset()

    @property
    def player(self) -> Player:
        return self._game._player1

    def action_masks(self) -> np.ndarray:
        """ Return True if the action is valid. """
        return self._action_mask

    def set_illegal_move_reward(self, penalty: float=0) -> None:
        self._illegal_move_reward = penalty

    def _decode_action(self, action: int) -> str:
        # Decode the action as server's format first.
        # If the action is related to a card (int), then further decode it as a position code.
        action: str | int = self.DIGITS2ACTION[action]
        if self._spec_unmap.get(action, None) is not None:
            action = self._spec_unmap[action]
        return action

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

    def seed(self, seed=None) -> None:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[Dict[str, Tensor], float, bool, bool, GameInfo]:
        reward = 0.

        # TODO: Figure out method to guarantee the action is valid from the policy model.
        if self._action_mask[action] == 1:
            action = self._decode_action(action)

            terminated, next_state_dict = self.player.step(action)
            if terminated:
                reward = next_state_dict.get('score', 0.0)
                # self.reset()
            else:
                self._encode_state(next_state_dict, self.player.list_valid_actions())

            return self._state, reward, terminated, False, {}
        else:
            # Illegal move. Nothing happens but the agent will be punished.
            return self._state, self._illegal_move_reward, False, False, {}

    def last(self) -> Tuple[Dict[str, Tensor], GameInfo]:
        return self._state, {}

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Tensor], GameInfo]:
        """ Reset the game.

        Arguments
        ---------
        seed: int | None
            The random seed for the game.
            ! NOT work because the randomness comes from both the game and the agent.

        options: dict | None
            The options for the game.

        Returns
        -------
        state: GameState
            The initial state of the game.

        info: Info
            Additional information.
        """
        # Halt the previous launched thread.
        #
        # Assume that all resources are released after the instance
        # is no longer referenced by any variables.
        if self._process is not None:
            self._process.terminate()

        if self._game is not None:
            self._game.close()

        # Re-create the game instance.
        # Launch a new thread for the opponent's decision making.
        self._game = Game().start()
        self._process = Process(target=game_loop, args=(self._game._player2, self._opponent))
        self._process.start()
        self._step = 0

        # Wait until server acknowledges the player to make a decision.
        _, state = self.player.decode(self.player.wait())

        # Encode the game state into the tensor.
        self._encode_state(state, self.player.list_valid_actions())

        return self._state, {}

    def finalize(self):
        """ Finalize the game.

        A workaround for the bug of the multiprocessing module.
        """
        # Halt the previous launched thread.
        if self._process is not None:
            self._process.terminate()

        # Assume that all resources are released after the instance
        # is no longer referenced by any variables.
        if self._game is not None:
            self._game.close()

    def render(self, mode='human', close=False):
        raise NotImplementedError()
