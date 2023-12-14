###############
#   Package   #
###############
import os
import sys
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding
from torch import Tensor
from torch.multiprocessing import Process as Process

# insert path for package
sys.path.insert(0, os.path.abspath(os.getcwd()))

from .game import (CARDS2IDX, DECK, OPTIONS, POSSIBLE_ACTIONS, Action, Game,
                      GameState, Player, Policy)
from .state import StateMachine

#######################
#   Global Variable   #
#######################

GameInfo = Dict[str, Any]
CardID = int

################
#   Function   #
################

def game_loop(player: Player, policy: Policy) -> None:
    terminated, state, action = player.decode_server_msg(player.wait())
    player._sm = StateMachine.from_dict(action)
    player._state = state

    while not terminated:
        options, targets = player.list_valid_actions()
        if type(policy).__name__ == "RandomPolicy":
            action = policy.react(state, options + list(targets.values()))
        else:
            action = policy.react(state, (options, targets))
        terminated, state, _ = player.step(action)

    return

#############
#   Class   #
#############

class YGOEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 2,
    }

    # option (string) -> option (int)
    OPTIONS2DIGITS = { option: i for i, option in enumerate(OPTIONS, 1) }
    OPTIONS2DIGITS[None] = 0

    # action (string | CardID) -> action (int)
    ACTION2DIGITS = { action: i for i, action in enumerate(POSSIBLE_ACTIONS) }

    # action (int) -> action (string | CardID)
    DIGITS2ACTION = { value: key for key, value in ACTION2DIGITS.items() }

    # phase (int) -> phase (int)
    PHASE2DIGITS = { 1: 0, 2: 1, 4: 2, 8: 3, 256: 4, 512: 5 }

    # Type hinting for public fields.

    action_space: spaces.Discrete = spaces.Discrete(len(ACTION2DIGITS.keys()), start=0)
    observation_space: spaces.Dict = spaces.Dict({

        # Current game phase (affect the valid actions)
        "phase": spaces.Discrete(6),

        # Is current player's turn
        "turn": spaces.MultiBinary(1),

        # -------------------------------- Players information --------------------------------

        # Player's life points (normalized to [0, 1])
        "agent_LP": spaces.Box(low=-1., high=1., shape=(1, ), dtype=np.float32),
        # Player's hand (multi-hot encoding, number of cards in hand <= 3)
        "agent_hand": spaces.Box(low=0., high=3., shape=(len(DECK), ), dtype=np.float32),
        # Player's deck (number of cards in deck), can infer the remaining cards in deck
        "agent_deck": spaces.Box(low=0, high=40, shape=(1, ), dtype=np.float32),
        # Player's grave (multi-hot encoding)
        "agent_grave": spaces.Box(low=0., high=4., shape=(len(DECK), ), dtype=np.float32),
        # Player's banished cards (multi-hot encoding)
        "agent_removed": spaces.Box(low=0., high=4., shape=(len(DECK), ), dtype=np.float32),

        # ------------------------------ Valid action information -----------------------------

        # Previous option (one-hot encoding, extra 1-dim for none)
        "last_option": spaces.Discrete(1 + len(OPTIONS)),

        # Valid action information
        "action_masks": spaces.MultiBinary(len(ACTION2DIGITS.keys())),

        # ------------------------------- Opponents information -------------------------------

        "oppo_LP": spaces.Box(low=-1, high=1., shape=(1, ), dtype=np.float32),
        "oppo_hand": spaces.Box(low=0., high=40., shape=(1, ), dtype=np.float32),
        "oppo_deck": spaces.Box(low=0, high=40, shape=(1, ), dtype=np.float32),
        "oppo_grave": spaces.Box(low=0., high=4., shape=(len(DECK), ), dtype=np.float32),
        "oppo_removed": spaces.Box(low=0., high=4., shape=(len(DECK), ), dtype=np.float32),

        # --------------------------------- Table information ---------------------------------

        "t_agent_m": spaces.MultiBinary(5 * (len(DECK) + 2)),
        "t_oppo_m": spaces.MultiBinary(5 * (len(DECK) + 2)),
        "t_agent_s": spaces.MultiBinary(5 * (len(DECK) + 2)),
        "t_oppo_s": spaces.MultiBinary(5 * (len(DECK) + 2)),
    })


    # Type hinting for private fields.

    _game: Game
    _opponent: Policy
    _process: Process

    # Field for caching the state of the game.

    _action_masks: np.ndarray | Tensor
    _spec_map: Dict[str, int]
    _spec_unmap: Dict[int, str]
    _state: Dict[str, Tensor]
    _info: GameInfo

    # No advantages for both players, just for demonstration.
    DEFAULT_ADVANTAGES = {
        'player1': {},
        'player2': { 'lifepoints': 8000 },
    }

    def __init__(self, reward_type: str = "", opponent: Policy = None, advantages: Dict = DEFAULT_ADVANTAGES):
        super(YGOEnv, self).__init__()
        # define the Game and the opponent object
        self._game = None
        self._opponent = opponent
        self._process = None
        self._advantages = advantages
        self._state = None

        # `reward_type` should be "win/loss reward", "LP reward", or "step count reward"
        match reward_type:
            case "win/loss":
                self._reward_type = 0
            case "LP":
                self._reward_type = 1
            case "step count reward":
                self._reward_type = -1
            case _:
                raise ValueError(f"Invalid reward type: {reward_type}")

        # Set negative reward (penalty) for illegal moves (optional)
        self.set_illegal_move_reward(-0.2)

        # ! DO NOT call reset() here.
        #   Otherwise the connection will NOT be closed and lead to infinite waiting.
        # self.reset()

    @property
    def player(self) -> Player:
        return self._game._player1

    def action_masks(self) -> np.ndarray:
        """ Return True if the action is valid. """
        return self._action_masks

    def set_illegal_move_reward(self, penalty: float=0) -> None:
        self._illegal_move_reward = penalty

    def decode_action(self, action: int) -> str:
        # Decode the action as server's format first.
        # If the action is related to a card (int), then further decode it as a position code.
        action: str | int = self.DIGITS2ACTION[action]
        if self._spec_unmap.get(action, None) is not None:
            action = self._spec_unmap[action]
        return action

    def _encode_state(
            self,
            game_state: Dict,
            actions: Tuple[List[Action], Dict[Action, Set[str]]]
        ):
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

    @classmethod
    def _IDStateList_to_vector(cls, id_state_list: List[Tuple[int, int]]) -> np.ndarray:
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

    def _reward_shaping(self, state: GameState) -> float:
        return (state['player']['lp'] - state['opponent']['lp']) / (16000. * self._info['steps'] ** (self._reward_type < 0 ))

    def seed(self, seed=None) -> None:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[Dict[str, Tensor], float, bool, bool, GameInfo]:
        # Illegal move. Nothing happens but the agent will be punished.
        if not self._action_masks[action]:
            return self._state, self._illegal_move_reward, False, False, self._info

        # Otherwise, interact with the environment.
        reward = 0.
        action = self.decode_action(action)
        terminated, next_state_dict, concrete_action = self.player.step(action)
        next_state_dict['last_option'] = self.player.last_option()
        self._info['steps'] += 1

        # * Win/Lose reward
        outcome = next_state_dict.get('score', .0)
        self._info['outcome'] = outcome
        reward += outcome

        # * reward shaping
        reward += self._reward_shaping(next_state_dict) * np.abs(self._reward_type)

        valid_actions = self.player.list_valid_actions()
        self._encode_state(next_state_dict, valid_actions)

        return self._state, reward, terminated, False, self._info

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

        if self._game is None:
            self._game = Game(self._advantages)

        # Re-create the game instance.
        self._game.start()
        self._info = { 'steps': 0, 'outcome': 0.0 }

        # Launch a new thread for the opponent's decision making.
        torch.multiprocessing.set_start_method('spawn', force=True)
        self._process = Process(target=game_loop, args=(self._game._player2, self._opponent))
        self._process.start()

        # Wait until server acknowledges the player to make a decision.
        terminated, state, action = self.player.decode_server_msg(self.player.wait())
        self.player._sm = StateMachine.from_dict(action)
        self.player._state = state
        state['last_option'] = self.player.last_option()

        # Encode the game state into the tensor.
        self._encode_state(state, self.player.list_valid_actions())

        return self._state, self._info

    def close(self):
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
