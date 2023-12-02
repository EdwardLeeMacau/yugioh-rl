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
from threading import Thread, Event
from typing import Dict, List, Tuple

# insert path for package
sys.path.insert(0, os.path.abspath(os.getcwd()))

from env.game import Action, Game, GameState, Player, Policy
import torch

from telnetlib import Telnet

if __name__ =="__main__":
    register(
        id="single_ygo",
        entry_point="env.single_gym_env:YGOEnv"
    )
    env = gym.make('single_ygo')
    env.reset()
    while True:
        obs, reward,  done, _, info = env.step(action)
