import os
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from model import MultiFeaturesExtractor
from env.single_gym_env import YGOEnv

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv"
)


# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "policy_network": "MultiInputPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 1,
    "timesteps_per_epoch": 100000,
    "eval_episode_num": 1,
}

def make_env():
    env = gym.make('single_ygo')
    return env

def train(env: YGOEnv, model, config):
    current_best_ = 0
    current_best = 0

    for epoch in range(config["epoch_num"]):
        ### Train agent using SB3
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            log_interval=1,
        )

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
                obs, reward, done, _, info = env.step(action)

        ### Save best model
        # model.save() encounters error because the environment utilizes threading.
        if epoch % 10 == 0:
            print("Saving Model")
            save_path = config["save_path"]
            # model.save(f"{save_path}/{epoch}")
        print("---------------")

    # Workaround for terminating the background threads
    ...


if __name__ == "__main__":
    train_env = DummyVecEnv([make_env for _ in range(32)])
    env = DummyVecEnv([make_env])
    model = MaskablePPO(
        my_config["policy_network"],
        train_env,
        learning_rate=0.0003,
        gamma=0.90,
        tensorboard_log=os.path.join("logs", my_config["run_id"]),
        policy_kwargs={
            "features_extractor_class": MultiFeaturesExtractor,
            "features_extractor_kwargs":{},
        }
    )
    train(env, model, my_config)
