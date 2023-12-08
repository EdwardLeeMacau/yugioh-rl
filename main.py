
import gymnasium as gym
from gymnasium.envs.registration import register
from tqdm import tqdm

from env.single_gym_env import YGOEnv

def main():
    register(
        id="single_ygo",
        entry_point="env.single_gym_env:YGOEnv"
    )

    env: YGOEnv = gym.make('single_ygo')
    env.reset()
    n = 0
    pbar = tqdm(total = 1001)
    # player = env.player
    # policy = env._opponent
    for _ in (pbar := tqdm(range(100000))):
        action = env.action_space.sample(mask=env.action_masks)
        obs, reward, done, _, info = env.step(action)
        if done:
            # pbar.update(1)
            env.reset()
        # if n > 1000:
        #     breakpoint()

    env.finalize()

if __name__ == "__main__":
    main()
