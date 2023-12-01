import json
import os
from datetime import datetime
from typing import List, Tuple

from env import SinglePlayerEnv
from game import GameState, Action
from policy import RandomPolicy
from tqdm import tqdm

def main():
    env = SinglePlayerEnv(opponent=RandomPolicy())
    # env.reset()

    # ---------------------- TODO: Implement the policy -----------------------
    policy = RandomPolicy()

    # -------------------------------------------------------------------------

    # Stress test: run 10000 games.
    for _ in tqdm(range(10000)):
        start = datetime.now().strftime('%Y%m%d-%H%M%S')
        terminated = False
        trajectories: List[Tuple[GameState, Action]] = []
        env.reset()
        while not terminated:
            state, reward, terminated, truncation, info = env.last()
            if terminated:
                break

            action = policy.react(state)
            env.step(action)

        # with open(os.path.join('logs', f'{start}-{env._game._player1.username}.json'), 'w') as f:
        #     json.dump(trajectories, f, indent=4)


if __name__ == '__main__':
    main()