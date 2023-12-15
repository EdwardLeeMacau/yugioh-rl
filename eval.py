import copy
import warnings
from itertools import chain
from tqdm import tqdm
from typing import Dict, List, Tuple

from queue import Queue
from joblib import Parallel, delayed

import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env_config import ENV_CONFIG
from env.single_gym_env import YGOEnv, GameInfo
from env.game import GameState, Action

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv",
    kwargs=ENV_CONFIG
)

# observation, action, decoded action, reward, game info,
Trajectory = Tuple[GameState, Action, str, float, GameInfo]

def play_game(env: YGOEnv, model: MaskablePPO) -> Trajectory:
    done = False
    trajectories = []

    obs, info = env.reset()
    trajectories.append((obs, None, None, None, info))

    while not done:
        mask = get_action_masks(env)
        action, state = model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)

        decoded_action = env.decode_action(action)
        obs, reward, done, _, info = env.step(action)

        trajectories.append((obs, action, decoded_action, reward, info))

    # Unwrap env to access the raw state
    # * env.player._state to access observation
    # * env.player._sm._current to access valid actions
    env = env.unwrapped
    lp = env.player._state['player']['lp'], env.player._state['opponent']['lp']
    deck = env.player._state['player']['deck'], env.player._state['opponent']['deck']

    if all(map(lambda x: x > 0, chain(lp, deck))):
        raise ValueError(f"Game is terminated unexpectedly. recv: {env.player._state}")

    return trajectories

def play_game_for_multi_process(resource_queue: Queue) -> Trajectory:
    env, model = resource_queue.get()
    trajectories = play_game(env, model)
    resource_queue.put((env, model))
    return trajectories

def play_game_multiple_times(num_game: int, env: YGOEnv, model, multi_process=True, num_resource=32, nums_worker=16) -> List[Trajectory]:
    if multi_process:
        # multi-process
        resource_queue = Queue()
        for _ in range(num_resource):
            resource_queue.put((copy.deepcopy(env), model))

        games_trajectories = Parallel(n_jobs=nums_worker, require='sharedmem')(delayed(play_game_for_multi_process)(resource_queue) for _ in tqdm(range(num_game), desc="Evaluation"))
        while not resource_queue.empty():
            env, model = resource_queue.get()
            env.close()

    else:
        # single process
        games_trajectories = []
        for _ in tqdm(range(num_game), desc="Evaluation"):
            trajectories = play_game(env, model)
            games_trajectories.append(trajectories)

    return games_trajectories

def calc_winning_rate(trajectories: List[Trajectory]):
    winning_times = 0
    for trajectory in trajectories:
        obs, action, decoded_action, reward, info = trajectory[-1]

        # only 3 case for score
        # score equal to 1 => wining
        # score equal to 0 => drawn
        # score equal to -1 => lossing
        score = info['score']

        assert score in [-1, 0, 1]

        if score > 0.0:
            winning_times += 1

    return winning_times / len(trajectories)

def calc_avg_step(trajectories: List[Trajectory]):
    total_step = 0
    for trajectory in trajectories:
        total_step += len(trajectory)
    return total_step / len(trajectories)

def evaluate(trajectories: List[Trajectory]) -> Dict:
    return {
        "winning_rate": calc_winning_rate(trajectories),
        "avg_step": calc_avg_step(trajectories),
    }

def main():
    # config
    num_game = 10
    model_path = "/home/bee/HW/RL/final/new_yugioh_rl/yugioh-rl/models/20231215-170036/2.zip"

    env = gym.make('single_ygo')
    model = MaskablePPO.load(model_path)

    game_trajectories = play_game_multiple_times(num_game, env, model)

    winning_rate = calc_winning_rate(game_trajectories)
    avg_step = calc_avg_step(game_trajectories)

    print(f"winning_rate: {winning_rate}")
    print(f"avg_step: {avg_step}")


if __name__ == '__main__':
    main()