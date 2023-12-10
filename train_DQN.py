import os
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

#import wandb
#from wandb.integration.sb3 import WandbCallback

import torch as th
import pysnooper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.dqn.policies import MultiInputPolicy, QNetwork

from model import MultiFeaturesExtractor
from env.single_gym_env import YGOEnv

from eval import play_game_multiple_times, calc_winning_rate

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv"
)

# Inherit from QNetwork to implement a masked DQN

# TODO: Double check the impl. to ensure the action mask works during
#       both training and inference time.
class MaskedQNetwork(QNetwork):
    def forward(self, obs: PyTorchObs) -> th.Tensor:
        mask = obs['action_mask'].clone().to(th.bool)

        q_values = super().forward(obs)
        q_values[~mask] = -th.inf

        return q_values

class MaskedPolicy(MultiInputPolicy):
    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": DQN,
    "policy_network": MaskedPolicy,
    "save_path": "models/sample_model",

    "epoch_num": 100,
    "timesteps_per_epoch": 25000,
    "eval_episode_num": 10,

    #"normalize_images": False,
}

def make_env():
    env = gym.make('single_ygo')
    return env

def train(env: YGOEnv, model, config):
    current_best_ = 0
    current_best = 0

    max_winning_rate = 0.0

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
        # for seed in range(config["eval_episode_num"]):
        #     done = False

        #     # Set seed using old Gym API
        #     env.seed(seed)
        #     obs = env.reset()

        #     # Interact with env using old Gym API
        #     while not done:
        #         action, _state = model.predict(obs, deterministic=False)
        #         obs, reward, done, info = env.step(action)

        ### Save best model
        # model.save() encounters error because the environment utilizes threading.
        # if epoch % 10 == 0:
        #     print("Saving Model")
        #     save_path = config["save_path"]
            # model.save(f"{save_path}/{epoch}")

        ### Evaluation time and save the model with higher winning rate
        games_trajectories = play_game_multiple_times(config['eval_episode_num'], env, model)
        winning_rate = calc_winning_rate(games_trajectories)        
        print(f"Winning rate: {winning_rate}")

        if winning_rate > max_winning_rate:
            print("Saving Model")
            max_winning_rate = winning_rate
            model.save(f"{config['save_path']}/{epoch}")

        print("---------------")

    # Workaround for terminating the background threads
    ...


if __name__ == "__main__":
    train_env = DummyVecEnv([make_env for _ in range(32)])
    env = make_env()
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        learning_rate=0.0007,# 0.00007,
        gamma=0.90,
        verbose=0,
        seed=1109,
        buffer_size=1000,
        tensorboard_log=os.path.join("logs", my_config["run_id"]),
        policy_kwargs={
            "features_extractor_class": MultiFeaturesExtractor,
            "features_extractor_kwargs":{},
        }
    )
    train(env, model, my_config)
