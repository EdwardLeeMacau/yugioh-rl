import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import torch as th
import torch.nn as nn

#import wandb
#from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import stable_baselines3

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv"
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": DQN,
    "policy_network": "MultiInputPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 100,
    "timesteps_per_epoch": 25000,
    "eval_episode_num": 10,
    
    #"normalize_images": False,
}

def make_env():
    env = gym.make('single_ygo')
    return env

def train(env, model, config):
    current_best_ = 0
    current_best = 0

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
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
            obs = env.reset()

            # Interact with env using old Gym API
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
        
        ### Save best model
        if epoch % 10 == 0:
            print("Saving Model")
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")
        print("---------------")

if __name__ == "__main__":

    env = DummyVecEnv([make_env])
    model = my_config["algorithm"](
        my_config["policy_network"], 
        env, 
        learning_rate=0.0007,#0.00007,
        gamma=0.099,
        verbose=0,
        seed=1109,
        tensorboard_log=my_config["run_id"],
    )
    train(env, model, my_config)