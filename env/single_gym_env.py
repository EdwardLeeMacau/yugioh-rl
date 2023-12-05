###############
#   Package   #
###############
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register

import numpy as np

import logging
import sys
import os
import torch

from itertools import combinations
from six import StringIO
from datetime import datetime
from multiprocessing import Process
from threading import Thread, Event
from typing import Dict, List, Tuple
from tqdm import tqdm
from telnetlib import Telnet

# insert path for package
sys.path.insert(0, os.path.abspath(os.getcwd()))

from env.game import Action, Game, GameState, Player, Policy
from policy import RandomPolicy

#######################
#   Global Variable   #
#######################
Info = Dict

################
#   Function   #
################
def game_loop(player: Player, policy: Policy) -> None:
    terminated, state = player.decode(player.wait())
    while not terminated:
        actions = player.list_valid_actions()
        action = policy.react(state, actions)
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


    action2digit = {
        "": 0, # no action
        "e": 1, # enter end phase
        "z": 2, # back
        "s": 3, # summon this card in face-up attack position
        "m": 4, # summon this card in face-down defense position/ enter main phase
        "t": 5, # set this card (Trap/Magic)
        "v": 6, #activate this card
        "c": 7, # cancel
        "b": 8, # enter battle phase
        "y": 9, # yes
        "n": 10, # no
        "a": 11, # attack

        "s1": 12,
        "s2": 13,
        "s3": 14,
        "s4": 15,
        "s5": 16,

        "m1": 17,
        "m2": 18,
        "m3": 19,
        "m4": 20,
        "m5": 21,

        "h1": 22,
        "h2": 23,
        "h3": 24,
        "h4": 25,
        "h5": 26,
        "h6": 27,
        "h7": 28,
        "h8": 29,
        "h9": 30,
        "h10": 31,

        "1": 32,
        "2": 33,
        "3": 34,
        "4": 35,
        "5": 36,
        "6": 37,
        "7": 38,
        "8": 39,
        "9": 40,
        "10": 41,

        "r": 42,

        "g1": 43,
        "g2": 44,
        "g3": 45,
        "g4": 46,
        "g5": 47,
        "g6": 48,
        "g7": 49,
        "g8": 50,
        "g9": 51,
        "g10": 52,
        "g11": 53,
        "g12": 54,
        "g13": 55,
        "g14": 56,
        "g15": 57,
        "g16": 58,
        "g17": 59,
        "g18": 60,
        "g19": 61,
        "g20": 62,
        "g21": 63,
        "g22": 64,
        "g23": 65,
        "g24": 66,
        "g25": 67,
        "g26": 68,
        "g27": 69,
        "g28": 70,
        "g29": 71,
        "g30": 72,
        "g31": 73,
        "g32": 74,
        "g33": 75,
        "g34": 76,
        "g35": 77,
        "g36": 78,
        "g37": 79,
        "g38": 80,
        "g39": 81,
        "g40": 82,

        "11": 83,
        "12": 84,
        "13": 85,
        "14": 86,
        "15": 87,
        "16": 88,
        "17": 89,
        "18": 90,
        "19": 91,
        "20": 92,
        "21": 93,
        "22": 94,
        "23": 95,
        "24": 96,
        "25": 97,
        "26": 98,
        "27": 99,
        "28": 100,
        "29": 101,
        "30": 102,
        "31": 103,
        "32": 104,
        "33": 105,
        "34": 106,
        "35": 107,
        "36": 108,
        "37": 109,
        "38": 110,
        "39": 111,
        "40": 112,
        
        "r1": 113,
        "r2": 114,
        "r3": 115,
        "r4": 116,
        "r5": 117,
        "r6": 118,
        "r7": 119,
        "r8": 120,
        "r9": 121,
        "r10": 122,
        "r11": 123,
        "r12": 124,
        "r13": 125,
        "r14": 126,
        "r15": 127,
        "r16": 128,
        "r17": 129,
        "r18": 130,
        "r19": 131,
        "r20": 132,
        "r21": 133,
        "r22": 134,
        "r23": 135,
        "r24": 136,
        "r25": 137,
        "r26": 138,
        "r27": 139,
        "r28": 140,
        "r29": 141,
        "r30": 142,
        "r31": 143,
        "r32": 144,
        "r33": 145,
        "r34": 146,
        "r35": 147,
        "r36": 148,
        "r37": 149,
        "r38": 150,
        "r39": 151,
        "r40": 152,

        "os1": 153,
        "os2": 154,
        "os3": 155,
        "os4": 156,
        "os5": 157,

        "om1": 158,
        "om2": 159,
        "om3": 160,
        "om4": 161,
        "om5": 162,
    }

    digit2action = {value: key for key, value in action2digit.items()}

    deck_list = [
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
            0,
            "empty",
            "token"
            ]

    _game: Game
    _opponent: Policy
    _process: Process

    _state: Tuple[GameState, float, bool, bool, Info]

    def __init__(self, opponent: Policy = RandomPolicy()):
        super(YGOEnv, self).__init__()
        # define the Game and the opponent object
        self._game = None
        self._opponent = opponent
        self._process = None

        self._state = None

        # define the action space and the observation space
        self.action_space = spaces.Discrete(len(self.action2digit.keys()), start=0)
        # trap card have not been implemented.
        self.observation_space = spaces.Dict({
                                            "phase": spaces.Discrete(6, start=1),
                                            "agent_LP": spaces.Box(low=0., high=1., shape=(1, ), dtype=np.float32),
                                            "agent_hand": spaces.MultiDiscrete([4 for i in range(40)]),
                                            "agent_deck": spaces.Discrete(41, start=0),
                                            "agent_grave": spaces.MultiDiscrete([4 for i in range(40)]),
                                            "agent_removed": spaces.MultiDiscrete([4 for i in range(40)]),
                                            "oppo_LP": spaces.Box(low=0., high=1., shape=(1, ), dtype=np.float32),
                                            "oppo_hand": spaces.Discrete(7, start=0),
                                            "oppo_deck": spaces.Discrete(41, start=0),
                                            "oppo_grave": spaces.MultiDiscrete([4 for i in range(40)]),
                                            "oppo_removed": spaces.MultiDiscrete([4 for i in range(40)]),
                                            "t_agent_m": spaces.MultiDiscrete([40, 5, 40, 5, 40, 5, 40, 5, 40, 5], dtype=np.int64),
                                            "t_oppo_m": spaces.MultiDiscrete([40, 5, 40, 5, 40, 5, 40, 5, 40, 5], dtype=np.int64),
                                            "t_agent_s": spaces.MultiDiscrete([40, 5, 40, 5, 40, 5, 40, 5, 40, 5], dtype=np.int64),
                                            "t_oppo_s": spaces.MultiDiscrete([40, 5, 40, 5, 40, 5, 40, 5, 40, 5], dtype=np.int64),
                                            })
        
        # Set negative reward (penalty) for illegal moves (optional)
        self.set_illegal_move_reward(-0.2)

        # Reset ready for a game
        #self.reset()

    @property
    def player(self) -> Player:
        return self._game._player1

    def list_valid_actions(self) -> List[Action]:
        return self.player.list_valid_actions()
    def set_illegal_move_reward(self, penalty: float=0) -> None:
        self._illegal_move_reward = penalty

    def _dict_to_state_vector(self, game_state: dict) -> spaces.Dict:
        try:
            tmp = game_state['phase']
        except:
            breakpoint()
        frame_dict = {
            "phase": np.array([game_state['phase']]),
            "agent_LP": np.array([game_state['score']['player']['lp'] / 8000.]),
            "agent_hand": self._IDList_to_MultiHot(game_state['hand']),
            "agent_deck": np.array([game_state['score']['player']['deck']]),
            "agent_grave": self._IDList_to_MultiHot(game_state['score']['player']['grave']),
            "agent_removed": self._IDList_to_MultiHot(game_state['score']['player']['removed']),
            "oppo_LP": np.array([game_state['score']['opponent']['lp'] / 8000.]),
            "oppo_hand": np.array([game_state['score']['opponent']['hand']]),
            "oppo_deck": np.array([game_state['score']['opponent']['deck']]),
            "oppo_grave": self._IDList_to_MultiHot(game_state['score']['opponent']['grave']),
            "oppo_removed": self._IDList_to_MultiHot(game_state['score']['opponent']['removed']),
            "t_agent_m": self._IDStateList_to_vector(game_state['table']['player']['monster']),
            "t_agent_s": self._IDStateList_to_vector(game_state['table']['player']['spell']),
            "t_oppo_m": self._IDStateList_to_vector(game_state['table']['opponent']['monster']),
            "t_oppo_s": self._IDStateList_to_vector(game_state['table']['opponent']['spell']),
        }
        return frame_dict

    def _digit_to_action(self, action: np.int64) -> str:
        return self.digit2action[action]

    def _IDStateList_to_vector(self, id_state_list: List) -> np.ndarray:
        frame_array = np.zeros(shape=(5, 2))
        frame_array[:, 0] = self.deck_list.index("empty")
        frame_array[:, 1] = 4
        for i in range(len(id_state_list)):
            try:
                frame_array[i, 0] = self.deck_list.index(id_state_list[i][0])
            except:
                frame_array[i, 0] = self.deck_list.index("token")
            frame_array[i, 1] = id_state_list[i][1]
        return frame_array.reshape(1, -1)

    def _IDList_to_MultiHot(self, id_list: List) -> np.ndarray:
        multi_hot = np.zeros(shape=(40, ), dtype=np.int64)
        # naive method
        # can be accelarated
        for ID in id_list:
            multi_hot[self.deck_list.index(ID)] += 1
        return multi_hot

    def _check_valid_action(self, action: Action) -> bool:
        return (action in self.list_valid_actions())

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(shape=(len(self.digit2action.keys()), ))
        for valid_action in self.list_valid_actions():
            try:
                mask[self.action2digit[valid_action]] = 1
            except:
                breakpoint()
        return mask.astype(np.int8)

    def seed(self, seed=None) -> None:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: Action) -> Tuple[spaces.Dict, float, bool, bool, Dict]:
        truncate = False
        reward = 0.
        action = self._digit_to_action(action)
        if self._check_valid_action(action):
            terminated, next_state_dict = self.player.step(action)

            self._state = next_state_dict

            if terminated:
                reward = next_state_dict.get('score', 0.0)
                next_state, _ = self.reset()
            else:
                next_state = self._dict_to_state_vector(next_state_dict)
            return next_state, reward, terminated, truncate, {}
        else:
            next_state = self._dict_to_state_vector(self._state)
            return next_state, self._illegal_move_reward, False, False, {}

    def last(self) -> Tuple[GameState, Dict]:
        return self._state, {}

    def reset(self, seed=None, options=None):
        if self._process is not None:
            self._process.terminate()

        if self._game is not None:
            self._game.close()

        self._game = Game().start()

        self._process = Process(target=game_loop, args=(self._game._player2, self._opponent))
        self._process.start()

        _, state = self.player.decode(self.player.wait())
        self._state = state

        return self._dict_to_state_vector(self._state), {}

    def render(self, mode='human', close=False):
        raise NotImplementedError()

if __name__ =="__main__":
    register(
        id="single_ygo",
        entry_point="env.single_gym_env:YGOEnv"
    )
    env = gym.make('single_ygo')
    env.reset()
    n = 0
    pbar = tqdm(total = 3000)
    while True:
        action = env.action_space.sample(mask=env.get_action_mask())
        obs, reward,  done, _, info = env.step(action)
        breakpoint()
        if done:
            pbar.update(1)
        if n > 1000:
            breakpoint()
