import json
import os
from datetime import datetime
from typing import List, Tuple

from env.env import SinglePlayerEnv
from env.game import GameState, Action
from policy import RandomPolicy
from policy import Policy
from tqdm import tqdm

def play_game(env: SinglePlayerEnv, policy: Policy) -> Tuple[List[Tuple[GameState, Action]], bool]:
    terminated = False
    trajectories: List[Tuple[GameState, Action]] = []
    env.reset()
    while not terminated:
        state, reward, terminated, truncation, info = env.last()

        if terminated:
            break

        action = policy.react(state)
        trajectories.append((state, action))
        env.step(action)
    
    is_win = (reward == 1.0)
    
    return trajectories, is_win

def main():
    # config
    num_rounds = 10
    env = SinglePlayerEnv(opponent=RandomPolicy())
    policy = RandomPolicy()

    # Play some rounds for the games
    info: List[List[Tuple[GameState, Action]], bool] = []
    for _ in tqdm(range(num_rounds)):
        trajectories, is_win = play_game(env, policy)
        info.append((trajectories, is_win))
    
    # calculate the winning rate and print the result
    winning_rate = (sum([ int(info[i][1]) for i in range(num_rounds)]) / num_rounds) * 100
    print(f"Winning rate: {winning_rate:.2f}%")

if __name__ == '__main__':
    main()