import argparse
import copy
import pysnooper
import warnings
from queue import Queue
from typing import List, Tuple

import gymnasium as gym
import pandas as pd
from gymnasium.envs.registration import register
from joblib import Parallel, delayed
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from tqdm import tqdm

from env.game import Action, GameState
from env.single_gym_env import GameInfo, YGOEnv
from env_config import ENV_CONFIG
from policy import RandomPolicy, PseudoSelfPlayPolicy

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv",
    # Remove all optional arguments to keep the environment consistent
    kwargs={
        'opponent': RandomPolicy(),
    },
)

register(
    id="ygo",
    entry_point="env.single_gym_env:Duel",
)

# observation, action, decoded action, reward, game info,
Trajectory = Tuple[GameState, Action, str, float, GameInfo]

def play_game(env: YGOEnv, model: MaskablePPO) -> Trajectory:
    done = False
    trajectories = []

    obs, info = env.reset()
    trajectories.append((None, None, info))

    while not done:
        mask = get_action_masks(env)
        action, state = model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)

        decoded_action = env.decode_action(action)
        obs, reward, done, _, info = env.step(action)

        trajectories.append((decoded_action, reward, info))

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

def evaluate(trajectories: List[Trajectory]) -> pd.DataFrame:
    steps = list(map(lambda x: len(x), trajectories))

    last = list(map(lambda x: x[-1][-1], trajectories))
    scores = list(map(lambda x: x['score'], last))
    remain = list(map(lambda x: x['state']['player']['lp'], last))
    win = list(map(lambda x: x == 1, scores))

    return pd.DataFrame({
        "steps": steps,
        "scores": scores,
        "win": win,
        "remain_lp": remain,
    })

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', "--num-game", type=int, default=1000)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "human"])
    return parser.parse_args()

def main():
    args = parse_args()

    model = MaskablePPO.load(args.model_path)
    match args.opponent:
        case "random":
            env: YGOEnv = gym.make('single_ygo')
            trajectories = play_game_multiple_times(args.num_game, env, model)
            metric = evaluate(trajectories)

            print(f'step: {metric["steps"].mean()}')
            print(f'win rate: {metric["win"].mean()}')
            print(f'remain lp: {metric["remain_lp"].mean()}')

        case "human":
            env: YGOEnv = gym.make('ygo')

            done = False
            obs, _ = env.reset()
            while not done:
                mask = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)

                print(f"Action: {env.decode_action(action)}")
                obs, _, done, _, _ = env.step(action)


    env.close()

if __name__ == '__main__':
    main()