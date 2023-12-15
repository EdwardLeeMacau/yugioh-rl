###############
#   Package   #
###############
import os
import sys
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import gymnasium as gym
import numpy as np
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
    _opponent_policy: Policy

    # Field for caching the state of the game.

    _action_masks: np.ndarray | Tensor
    _reward_kwargs: Dict[str, Any]
    _spec_map: Dict[str, int]
    _spec_unmap: Dict[int, str]
    _state: Dict[str, Tensor]
    _info: GameInfo

    # No advantages for both players, just for demonstration.
    DEFAULT_ADVANTAGES = {
        'player1': {},
        'player2': { 'lifepoints': 8000 },
    }

    def __init__(
            self,
            reward_kwargs: Dict = { 'type': 'win/loss', 'factor': 1 },
            opponent_policy: Policy = None,
            advantages: Dict = DEFAULT_ADVANTAGES,
        ):
        super(YGOEnv, self).__init__()
        # define the Game and the opponent object
        self._game = None
        self._advantages = advantages
        self._opponent_policy = opponent_policy
        self._state = None

        if reward_kwargs['type'] not in ["win/loss", "LP", "LP_linear_step", "LP_exp_step"]:
            raise ValueError(f"Invalid reward type: {reward_kwargs}")

        self._reward_kwargs = reward_kwargs

        # ! DO NOT call reset() here.
        #   Otherwise the connection will NOT be closed and lead to infinite waiting.
        # self.reset()

    @property
    def player(self) -> Player:
        return self._game._players[0]

    def decode_action(self, action: int) -> str:
        # Decode the action as server's format first.
        # If the action is related to a card (int), then further decode it as a position code.
        action: str | int = self.DIGITS2ACTION[action]
        if self._spec_unmap.get(action, None) is not None:
            action = self._spec_unmap[action]
        return action

    # --------------------------------------------------------------------------
    # Private functions
    # --------------------------------------------------------------------------

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

            # Games information

            "phase": self.PHASE2DIGITS[game_state['phase']],
            "turn": np.array([game_state['turn']], dtype=np.float32),

            # Players information

            "agent_LP": np.array([player['lp']]) / 8000.,
            "agent_hand": self._card_to_vec(player['hand']),
            "agent_deck": np.array([player['deck']]) / 40.,
            "agent_grave": self._card_to_vec(player['grave']),
            "agent_removed": self._card_to_vec(player['removed']),

            # Valid action information

            "last_option": self.OPTIONS2DIGITS[game_state['last_option']],
            "action_masks": self._action_masks.astype(np.float32),

            # Opponents information

            "oppo_LP": np.array([opponent['lp']]) / 8000.,
            "oppo_hand": np.array([opponent['hand']]) / 40.,
            "oppo_deck": np.array([opponent['deck']]) / 40.,
            "oppo_grave": self._card_to_vec(opponent['grave']),
            "oppo_removed": self._card_to_vec(opponent['removed']),

            # Table information

            "t_agent_m": self._card_state_to_vec(player['monster']).flatten(),
            "t_agent_s": self._card_state_to_vec(player['spell']).flatten(),
            "t_oppo_m": self._card_state_to_vec(opponent['monster']).flatten(),
            "t_oppo_s": self._card_state_to_vec(opponent['spell']).flatten(),
        }

    @staticmethod
    def _card_state_to_vec(card_state: List[Tuple[int, int]]) -> np.ndarray:
        assert len(card_state) == 5, "The card list is padded to 5 with empty card (None)"

        frame_array = np.zeros(shape=(5, len(DECK) + 2), dtype=np.float32)
        for i, id_state in enumerate(card_state):
            card_id, pos = id_state[0], id_state[1]

            # One-hot encoding for the card ID.
            frame_array[i, CARDS2IDX[card_id]] = 1

            # One-hot encoding for the position (UP/DOWN, ATK/DEF).
            frame_array[i, len(DECK):] = [pos & 0b101, pos & 0b011]

        return frame_array

    @staticmethod
    def _card_to_vec(cards: List[int]) -> np.ndarray:
        multi_hot = np.zeros(shape=(len(DECK), ), dtype=np.float32)

        for card_id in cards:
            multi_hot[CARDS2IDX[card_id]] += 1

        return multi_hot

    def _wait(self) -> Dict:
        """ Wait until the server ask player to decide an action. """

        # Message communicates through the player's connection.
        # But not all of them should be read by the player.
        player, opponent = self._game._players[0], self._game._players[1]
        while (embed := player.wait()):
            ### recv
            while True:
                _, state, action, _ = self._decode_server_msg(embed)
                receiver: Player = self._game._players[embed['recv']]

                if action is None or action['requirement'] != 'PLACE':
                    break

                n = action['min']
                response = ' '.join(action['options'][:n])
                receiver._write(response.encode() + b'\r\n')
                embed = player.wait()

            if receiver is player:
                break

            opponent._sm = StateMachine.from_dict(action)
            opponent._state = state
            while True:
                state['last_option'] = opponent.last_option()

                ### predict
                options, targets = opponent.list_valid_actions()
                action = self._opponent_policy.react(state, options, targets)

                ### write
                if (is_sent := opponent.step(action)) is not None:
                    break

        return embed

    @staticmethod
    def _decode_server_msg(
            embed: Dict
        ) -> Tuple[bool, Dict, Dict | None, float | None]:
        """
        Returns
        -------
        terminated : bool
            Whether the game is over.

        state : Dict
            The state in Dict format.

        actions : Dict | None
            The valid actions in Dict format.

        score: float | None
            The result of the game.
        """
        return (
            'terminated' in embed,
            embed.get('state', None),
            embed.get('actions', None),
            embed.get('score', None)
        )

    # --------------------------------------------------------------------------
    # Gym's API
    # --------------------------------------------------------------------------

    def seed(self, seed=None) -> None:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[Dict[str, Tensor], float, bool, bool, GameInfo]:
        # Illegal move. Nothing happens but the agent will be punished.
        if not self._action_masks[action]:
            return self._state, -0.2, False, False, self._info

        # Otherwise, interact with the environment.
        action = self.decode_action(action)

        # Request the next state
        if self.player.step(action) is not None:
            terminated, next_state, action, score = self._decode_server_msg(self._wait())
            self.player._sm = StateMachine.from_dict(action)
            self.player._state = next_state
        else:
            terminated, next_state, action, score = False, self.player._state, None, None

        next_state['last_option'] = self.player.last_option()

        # Maintain reward function
        reward = 0.

        # * Win/Lose reward
        reward += score if score is not None else 0.

        # * Reward shaping
        LP_diff = (next_state['player']['lp'] - next_state['opponent']['lp']) / 16000.
        match self._reward_kwargs['type']:
            case 'win/loss':
                pass
            case 'LP':
                reward += LP_diff * self._reward_kwargs['weight']
            case 'LP_linear_step':
                reward += (LP_diff * self._reward_kwargs['weight']) / self._info['steps']
            case 'LP_exp_step':
                reward += (LP_diff * self._reward_kwargs['weight']) / np.exp(self._info['steps'] / self._reward_kwargs['temperature'])

        # Prepare the next state tensor.
        self._encode_state(next_state, self.player.list_valid_actions())

        # Copy player states for evaluation.
        self._info['steps'] += 1
        self._info['state'] = next_state
        self._info['score'] = score

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
        if self._game is None:
            self._game = Game(self._advantages)

        # Re-start the game instance.
        self._game.start()

        # Wait until server acknowledges the player to make a decision.
        _, state, action, _ = self._decode_server_msg(self._wait())
        self.player._sm = StateMachine.from_dict(action)
        self.player._state = state
        state['last_option'] = None

        # Encode the game state into the tensor.
        self._encode_state(state, self.player.list_valid_actions())

        self._info = { 'steps': 0, 'score': 0.0, 'state': state }

        return self._state, self._info

    def close(self):
        """ Finalize the game.

        A workaround for the bug of the multiprocessing module.
        """
        # Assume that all resources are released after the instance
        # is no longer referenced by any variables.
        if self._game is not None:
            self._game.close()

    def render(self, mode='human', close=False):
        raise NotImplementedError()

    def action_masks(self) -> np.ndarray:
        """ Return True if the action is valid. """
        return self._action_masks
