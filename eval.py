import warnings
from itertools import chain
from tqdm import tqdm
from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env.single_gym_env import YGOEnv
from env.game import GameState, Action

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv"
)

# observation, action, decoded action, reward
Trajectory = Tuple[GameState, Action, str, float]

def play_game(env: YGOEnv, model) -> Trajectory:
    done = False
    trajectories = []

    obs, info = env.reset()
    trajectories.append((obs, None, None, None))

    while not done:
        if type(model) == MaskablePPO:
            mask = get_action_masks(env)
            action, state = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, state = model.predict(obs, deterministic=True)
        action = int(action)

        decoded_action = env.decode_action(action)
        obs, reward, done, _, info = env.step(action)

        trajectories.append((obs, action, decoded_action, reward))

    # Unwrap env to access the raw state
    # * env.player._state to access observation
    # * env.player._sm._current to access valid actions
    env = env.unwrapped
    lp = env.player._state['player']['lp'], env.player._state['opponent']['lp']
    deck = env.player._state['player']['deck'], env.player._state['opponent']['deck']

    if all(map(lambda x: x > 0, chain(lp, deck))):
        raise ValueError(f"Game is terminated unexpectedly. recv: {env.player._state}")

    return trajectories

def play_game_multiple_times(num_game: int, env: YGOEnv, model) -> List[Trajectory]:
    games_trajectories = []
    for _ in tqdm(range(num_game), desc="Evaluation"):
        trajectories = play_game(env, model)
        games_trajectories.append(trajectories)

    return games_trajectories

def calc_winning_rate(trajectories: List[Trajectory]):
    winning_times = 0
    for trajectory in trajectories:
        obs, action, decoded_action, reward = trajectory[-1]
        if reward > 0.0:
            winning_times += 1

    return winning_times / len(trajectories)

def evaluate(trajectories: List[Trajectory]) -> Dict:
    return {
        "winning_rate": calc_winning_rate(trajectories)
    }

def main():
    # config
    num_game = 100
    model_path = "models/sample_model/0.zip"

    env = gym.make('single_ygo')
    model = MaskablePPO.load(model_path)

    game_trajectories = play_game_multiple_times(num_game, env, model)
    winning_rate = calc_winning_rate(game_trajectories)
    print(f"winning_rate: {winning_rate}")


if __name__ == '__main__':
    main()