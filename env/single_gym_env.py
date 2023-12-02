###############
#   Package   #
###############
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register

import numpy as np

import itertools
import logging
from six import StringIO
import sys
import os
from datetime import datetime
from multiprocessing import Process
from threading import Thread, Event
from typing import Dict, List, Tuple

# insert path for package
sys.path.insert(0, os.path.abspath(os.getcwd()))

from env.game import Action, Game, GameState, Player, Policy
from policy import RandomPolicy
import torch

from telnetlib import Telnet

################
#   Function   #
################
def game_loop(player: Player, agent: Policy) -> None:
    while True:
        terminated, state = player.wait_action()
        if terminated:
            break

        action = agent.react(state)
        player.interact(action)

    return

#############
#   Class   #
#############
class YGOEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 2,
    }

    digit2action = {
        0: "", # no action
        1: "e", # enter end phase
        2: "z", # back
        3: "s", # summon this card in face-up attack position
        4: "m", # summon this card in face-down defense position/ enter main phase
        5: "t", # set this card (Trap/Magic)
        6: "v", #activate this card
        7: "c", # cancel
        8: "b", # enter battle phase
        9: "y", # yes
        10: "n", # no
        11: "a", # attack

        12: "s1",
        13: "s2",
        14: "s3",
        15: "s4",
        16: "s5",

        17: "m1",
        18: "m2",
        19: "m3",
        20: "m4",
        21: "m5",

        22: "h1",
        23: "h2",
        24: "h3",
        25: "h4",
        26: "h5",
        27: "h6",
        28: "h7",
        29: "h8",
        30: "h9",
        31: "h10",

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

    }

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

    def __init__(self, opponent: Policy = RandomPolicy()):
        super(YGOEnv, self).__init__()
        # define the players for agent and its opponents
        # player1 is for the RL agent
        #self._player1 = YGOPlayer(HOST, PORT, "player7", "player7")
        #self._player2 = YGOPlayer(HOST, PORT, "player8", "player8")
        self._game = None
        self._opponent = opponent
        self._process = None

        # define the action space and the observation space
        self.action_space = spaces.Discrete(32, start=0)
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
                                            "t_agent_m": spaces.MultiDiscrete([[40, 5] for i in range(5)], dtype=np.int32),
                                            "t_oppo_m": spaces.MultiDiscrete([[40, 5] for i in range(5)], dtype=np.int32),
                                            "t_agent_s": spaces.MultiDiscrete([[40, 5] for i in range(5)], dtype=np.int32),
                                            "t_oppo_s": spaces.MultiDiscrete([[40, 5] for i in range(5)], dtype=np.int32),
                                            })
        
        # Set negative reward (penalty) for illegal moves (optional)
        self.set_illegal_move_reward(-0.2)

        # Reset ready for a game
        #self.reset()

    def set_illegal_move_reward(self, penalty: float=0) -> None:
        self._illegal_move_reward = penalty

    def _dict_to_state_vector(self, game_state: dict) -> spaces.Dict:
        frame_dict = {
            "phase": game_state['state']['phase'],
            "agent_LP": game_state['state']['score']['player']['lp'] / 8000.,
            "agent_hand": self._IDList_to_MultiHot(game_state['state']['hand']),
            "agent_deck": game_state['state']['score']['player']['deck'],
            "agent_grave": self._IDList_to_MultiHot(game_state['state']['score']['player']['grave']),
            "agent_removed": self._IDList_to_MultiHot(game_state['state']['score']['player']['removed']),
            "oppo_LP": game_state['state']['score']['opponent']['lp'] / 8000.,
            "oppo_hand": game_state['state']['score']['opponent']['hand'],
            "oppo_deck": game_state['state']['score']['opponent']['deck'],
            "oppo_grave": self._IDList_to_MultiHot(game_state['state']['score']['opponent']['grave']),
            "oppo_removed": self._IDList_to_MultiHot(game_state['state']['score']['opponent']['removed']),
            "t_agent_m": self._IDStateList_to_vector(game_state['state']['table']['player']['monster']),
            "t_agent_s": self._IDStateList_to_vector(game_state['state']['table']['player']['spell']),
            "t_oppo_m": self._IDStateList_to_vector(game_state['state']['table']['opponent']['monster']),
            "t_oppo_s": self._IDStateList_to_vector(game_state['state']['table']['opponent']['spell']),
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
            frame_array[i, 1] = id_state_list[i][0]
        return frame_array

    def _IDList_to_MultiHot(self, id_list: List) -> np.ndarray:
        multi_hot = np.zeros(shape=(40, ), dtype=np.int32)
        # naive method
        # can be accelarated
        for ID in id_list:
            multi_hot[self.deck_list.index(ID)] += 1
        return multi_hot

    def _check_valid_action(self, action: str) -> bool:
        for valid_action in Policy.list_valid_actions(self.current_state_dict):
            if action in valid_action.split("\r\n"):
                return True
        return False

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(shape=(len(self.digit2action.keys()), ))
        for valid_action in Policy.list_valid_actions(self.current_state_dict):
            for sub_valid_action in valid_action.split("\r\n"):
                mask[self.action2digit[sub_valid_action]] = 1
        return mask.astype(np.int8)

    def step(self, action: Action) -> Tuple[spaces.Dict, float, bool, bool, Dict]:
        truncate = False
        reward = 0.
        action = self._digit_to_action(action)
        if self._check_valid_action(action):
            self._game._player1.interact(action)
            terminated, next_state_dict = self._game._player1.wait_action()
            self.current_state_dict = next_state_dict
            next_state = self._dict_to_state_vector(next_state_dict)
            if terminated:
                reward = state.get('score', 0.0)
                self._game._player1.interact("")
                next_state, _ = self.reset()
            return next_state, reward, terminated, truncate, {}
        else:
            next_state = self._dict_to_state_vector(self.current_state_dict)
            return next_state, self._illegal_move_reward, False, False, {}
        """
        #OLD VERSION
        done = False
        truncate = False
        reward = 0

        terminated, pre_state_dict = self._game._player1.wait_action()
        pre_state = self._dict_to_state_vector(pre_state_dict)
        # I am not sure if the "terminated" represents the end of the whole game.
        # this line should be checked!!!!!!!
        if not terminated:
            action = self._digit_to_action(action)
            self._game._player1.interact(action)
            _, next_state_dict = self._game._player1.wait_action()
            
            if pre_state_dict["state"] == next_state_dict["state"]:
                # this if loop checks if the action is meaningless.
                reward = self._illegal_move_reward

                return pre_state, reward, done, truncate, {}
            next_state = self._dict_to_state_vector(next_state_dict)
            return next_state, reward, done, truncate, {}

        else:
            reward, done, truncate = self.last(action)
            next_state, _ = self.reset()
            return next_state, reward, terminated, False, {}
        """

    def last(self) -> Tuple[float, bool, bool]:
        terminated, state = self._game._player1.wait_action()

        reward = state.get('score', 0.0)

        return reward, terminated, False


    def reset(self, seed=None, options=None):
        if self._process is not None:
            self._process.terminate()

        if self._game is not None:
            self._game.close()
        self._game = Game().start()
        self._process = Process(target=game_loop, args=(self._game._player2, self._opponent))
        self._process.start()
        _, next_state = self._game._player1.wait_action()
        self.current_state_dict = next_state
        return self._dict_to_state_vector(next_state), {}


    def render(self, mode='human', close=False):
        raise NotImplementedError()

if __name__ =="__main__":
    register(
        id="single_ygo",
        entry_point="env.single_gym_env:YGOEnv"
    )
    env = gym.make('single_ygo')
    env.reset()
    while True:
        action = env.action_space.sample(mask=env.get_action_mask())
        obs, reward,  done, _, info = env.step(action)
        if done:
            breakpoint()
