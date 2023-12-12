import os
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from model import MultiFeaturesExtractor
from env.single_gym_env import YGOEnv
from policy import *

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv",
    kwargs={
        "opponent": PseudoSelfPlayPolicy_0(model_path="./models/sample_model/50"),
    }
)



# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "policy_network": "MultiInputPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 100,
    "timesteps_per_epoch": 102400,
    "n_steps": 128,
    "parallel": 8,
    "eval_episode_num": 1,
}

def make_env():
    env = gym.make('single_ygo')
    return env

def train(model, config):
    current_best_ = 0
    current_best = 0
    outcome_list = []

    for epoch in range(config["epoch_num"]):
        ### Train agent using SB3
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            log_interval=1,
            progress_bar=True
        )

        # Reconstruct the environment to avoid the issue of threading
        env = DummyVecEnv([make_env])

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score = 0
        avg_highest = 0
        for seed in range(config["eval_episode_num"]):
            done = False

            # Set seed using old Gym API
            env.seed(seed)

            # Interact with env using old Gym API
            obs = env.reset()
            while not done:
                mask = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, done, info = env.step(action)
        outcome = "win" if info[0]['outcome'] > 0 else 'loss'
        outcome_list.append(info[0]['outcome'])
        moving_win_rate = np.sum(np.array(outcome_list[-10:]) > 0)/len(outcome_list[-10:])
        print("『{:s}』 with {:d} steps, moving average win rate (last 10 eval result) = {:.4f}".format(outcome, info[0]['steps'], moving_win_rate))

        # Manually close the connection to the server to ensure the resources are released
        env.envs[0].unwrapped.finalize()

        ### Save best model
        # model.save() encounters error because the environment utilizes threading.
        if epoch % 10 == 0:
            print("Saving Model")
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")
        print("---------------")

    # Workaround for terminating the background threads
    ...


if __name__ == "__main__":
    train_env = DummyVecEnv([make_env for _ in range(my_config["parallel"])])
    model = MaskablePPO(
        my_config["policy_network"],
        train_env,
        learning_rate=0.0003,
        gamma=0.90,
        n_steps=my_config["n_steps"],
        tensorboard_log=os.path.join("logs", my_config["run_id"]),
        policy_kwargs={
            "features_extractor_class": MultiFeaturesExtractor,
            "features_extractor_kwargs":{},
        }
    )
    train(model, my_config)
